"""Measures the built npm package in tessera/pkg in Chrome, Firefox and Safari through WebDriver:
compressed file sizes, cold createTessera time, warm parseAddress and detect latency (rules only,
and with the model on a 10,000-character document), and, in Chrome, the memory the page gains.
Every browser configuration runs with the SIMD wasm and with the baseline, with the worker off
and on. Prints markdown tables. Run from the repository root
after `just wasm`:

    python3 bench/web/measure.py [chrome] [firefox] [safari] [--json out.json]

macOS only: it reads the hardware with sysctl and finds the browsers in /Applications. Chrome
uses the chromedriver on PATH or in wasm-pack's cache that matches the installed Chrome's major
version, else the newest; Firefox uses `geckodriver` from PATH, and Safari
`/usr/bin/safaridriver`, which must be enabled with `safaridriver --enable`. Python standard
library only; brotli sizes appear when the `brotli` module or command exists.
"""

import functools
import glob
import hashlib
import http.server
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAGE = "/bench/web/measure.html"
PKG = "tessera/pkg"
BUNDLE = "models/tessera-v1.safetensors"
SHIPPED = ["tessera.js", "index.js", "worker.js", "tessera_simd_bg.wasm", "tessera_bg.wasm"]
COLD_RUNS = 10
# Warm calls cycle through the fixture addresses this many times, so each weighs the same.
WARM_PASSES = 3
# Model-backed detect runs on one document of this many characters, this many times.
LONG_CHARS = 10_000
MODEL_RUNS = 20
CONFIGS = [(variant, worker) for variant in ("simd", "baseline") for worker in ("off", "on")]

requested = []


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".safetensors": "application/octet-stream",
    }

    def log_message(self, *args):
        pass

    def end_headers(self):
        # Cross-origin isolation enables measureUserAgentSpecificMemory and finer timers; no-store
        # keeps every load cold, including the wasm the worker fetches.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.endswith(".wasm"):
            requested.append(os.path.basename(self.path.split("?")[0]))
        super().do_GET()


class Driver:
    def __init__(self, name, command, port, capabilities):
        self.name = name
        self.base = f"http://127.0.0.1:{port}"
        self.capabilities = capabilities
        self.session = None
        self.proc = subprocess.Popen(command + [f"--port={port}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    self.call("GET", "/status")
                    break
                except OSError:
                    time.sleep(0.1)
            self.start()
        except BaseException:
            self.proc.terminate()
            raise

    def start(self):
        """A new session, which is a new browser process with a new profile."""
        value = self.call("POST", "/session", {"capabilities": {"alwaysMatch": self.capabilities}})
        self.session = value["sessionId"]
        self.version = value["capabilities"].get("browserVersion", "?")
        self.call("POST", f"/session/{self.session}/timeouts", {"script": 300000})

    def restart(self):
        self.call("DELETE", f"/session/{self.session}")
        self.session = None
        # safaridriver can refuse a new session for a moment after the last one ends.
        for attempt in range(20):
            try:
                self.start()
                return
            except RuntimeError:
                if attempt == 19:
                    raise
                time.sleep(0.5)

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=400) as res:
                return json.load(res)["value"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.name} {method} {path}: {e.code} {e.read()[:600]!r}") from None

    def measure(self, url, cfg):
        requested.clear()
        # about:blank first so the next page is a new document even where the URL repeats.
        self.call("POST", f"/session/{self.session}/url", {"url": "about:blank"})
        self.call("POST", f"/session/{self.session}/url", {"url": url})
        script = (
            "const [cfg, done] = arguments;"
            "(function go() { window.measure ? window.measure(cfg).then(done, (e) => done({ error: `${e.name}: ${e.message} ${e.code ?? ''}` }))"
            " : setTimeout(go, 20); })();"
        )
        result = self.call("POST", f"/session/{self.session}/execute/async", {"script": script, "args": [cfg]})
        if "error" in result:
            raise RuntimeError(f"{self.name} {cfg['mode']}: {result['error']}")
        # Every request, repeats included: a worker-backed instance fetches the wasm a second time.
        result["wasm"] = list(requested)
        return result

    def quit(self):
        try:
            if self.session:
                self.call("DELETE", f"/session/{self.session}")
        finally:
            self.proc.terminate()


def version_of(command):
    out = subprocess.run(command + ["--version"], capture_output=True, text=True).stdout
    match = re.search(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", out)
    return tuple(map(int, match.groups())) if match else (0,)


def chromedriver():
    """The chromedriver matching the installed Chrome's major version, else the newest found."""
    candidates = [shutil.which("chromedriver")] + glob.glob(os.path.expanduser("~/Library/Caches/.wasm-pack/chromedriver-*/chromedriver"))
    candidates = [c for c in candidates if c]
    if not candidates:
        return None
    chrome = version_of(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"])
    versions = {c: version_of([c]) for c in candidates}
    matching = [c for c in candidates if versions[c][0] == chrome[0]]
    return max(matching or candidates, key=lambda c: versions[c])


def browsers():
    return {
        # ForceEagerMeasureMemory makes measureUserAgentSpecificMemory answer at once instead of
        # after the next scheduled garbage collection.
        "chrome": (
            [chromedriver()],
            9541,
            {"browserName": "chrome", "goog:chromeOptions": {"args": ["--headless=new", "--enable-blink-features=ForceEagerMeasureMemory"]}},
        ),
        "firefox": ([shutil.which("geckodriver")], 4471, {"browserName": "firefox", "moz:firefoxOptions": {"args": ["-headless"]}}),
        "safari": (["/usr/bin/safaridriver"], 4481, {"browserName": "safari"}),
    }


def read(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        return f.read()


def brotli_size(data):
    try:
        import brotli

        return len(brotli.compress(data, quality=11))
    except ImportError:
        if shutil.which("brotli"):
            return len(subprocess.run(["brotli", "-q", "11", "-c"], input=data, capture_output=True, check=True).stdout)
        return None


def sizes():
    rows = []
    for name in SHIPPED + [BUNDLE]:
        path = name if name == BUNDLE else f"{PKG}/{name}"
        data = read(path)
        # -n leaves the file name out of the header, as a server compressing a response would.
        gz = len(subprocess.run(["gzip", "-9", "-n", "-c"], input=data, capture_output=True, check=True).stdout)
        rows.append({"file": path, "raw": len(data), "gzip": gz, "brotli": brotli_size(data)})
    return rows


def email_sample():
    source = read("site/src/samples.rs").decode()
    match = re.search(r'name: "email",.*?country_hint: "(\w+)",\s*text: r#"(.*?)"#', source, re.S)
    if not match:
        raise RuntimeError("email sample not found in site/src/samples.rs")
    return match.group(2), match.group(1)


def addresses():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "fixtures/parser/*.json"))):
        with open(path, encoding="utf-8") as f:
            out += [case["input"] for case in json.load(f)["cases"]]
    return out


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def run_browser(name, url, base_cfg):
    command, port, caps = browsers()[name]
    if not command[0]:
        raise RuntimeError(f"no driver found for {name}")
    driver = Driver(name, command, port, caps)
    out = {"version": driver.version, "cold": {}, "cold_process": {}, "parse": {}, "detect": {}, "detect_model": {}}
    try:
        # Round robin, so drift spreads evenly across configurations. "cold" loads a new document
        # in the same browser process; "cold_process" starts a new session first, so nothing the
        # engine keeps in memory between documents, such as compiled code, can carry over.
        for fresh in (False, True):
            for _ in range(COLD_RUNS):
                for variant, worker in CONFIGS:
                    if fresh:
                        driver.restart()
                    cfg = {**base_cfg, "mode": "cold", "baseline": variant == "baseline", "worker": worker == "on", "memory": False}
                    r = driver.measure(url, cfg)
                    out["cold_process" if fresh else "cold"].setdefault(f"{variant} {worker}", []).append(r)
        for mode in ("parse", "detect", "detect_model"):
            for variant, worker in CONFIGS:
                cfg = {**base_cfg, "mode": mode, "baseline": variant == "baseline", "worker": worker == "on", "memory": True}
                out[mode][f"{variant} {worker}"] = driver.measure(url, cfg)
    finally:
        driver.quit()
    return out


def check(results):
    problems = []
    for name, r in results.items():
        for mode in ("cold", "cold_process", "parse", "detect", "detect_model"):
            for key, runs in r[mode].items():
                for run in runs if isinstance(runs, list) else [runs]:
                    variant, worker = key.split()
                    # One fetch by the page, and with the worker on one more by the worker; a
                    # worker row with one fetch ran inline because the worker failed to start.
                    want = ["tessera_bg.wasm" if variant == "baseline" else "tessera_simd_bg.wasm"] * (2 if worker == "on" else 1)
                    if run["wasm"] != want:
                        problems.append(f"{name} {mode} {key}: requested {run['wasm']}, expected {want}")
                    if not mode.startswith("detect") and run["components"] == 0:
                        problems.append(f"{name} {mode} {key}: parse returned no components")
                    if mode.startswith("detect") and run["entities"] == 0:
                        problems.append(f"{name} {mode} {key}: detect found nothing")
                    if not run["isolated"]:
                        problems.append(f"{name} {mode} {key}: page not cross-origin isolated")
    return problems


def ms(v):
    return f"{v:.2f}"


def mb(v):
    return "n/a" if v is None else f"{v:,} ({v / 1e6:.2f} MB)"


def report(size_rows, results, environment):
    lines = ["## Sizes", "", "`gzip -9 -n`; brotli at quality 11.", "", "| File | Raw | gzip -9 | brotli -q 11 |", "| --- | ---: | ---: | ---: |"]
    for row in size_rows:
        br = "n/a" if row["brotli"] is None else f"{row['brotli']:,}"
        lines.append(f"| {row['file']} | {row['raw']:,} | {row['gzip']:,} | {br} |")
    # A page fetches one wasm variant. The worker requests the same URL again, which a cacheable
    # response serves from the HTTP cache; this harness disables caching, so it downloads twice.
    download = [r for r in size_rows if not r["file"].endswith("tessera_bg.wasm")]
    total_br = None if any(r["brotli"] is None for r in download) else f"{sum(r['brotli'] for r in download):,}"
    lines.append(
        f"| everything a web user downloads (simd wasm, js, bundle) | {sum(r['raw'] for r in download):,} "
        f"| {sum(r['gzip'] for r in download):,} | {total_br or 'n/a'} |"
    )
    lines += ["", "## Environment", ""] + [f"- {k}: {v}" for k, v in environment.items()]
    for name, r in results.items():
        lines.append(f"- {name}: {r['version']}")
    lines += [
        "",
        f"## Cold init ({COLD_RUNS} runs per row)",
        "",
        "From calling `createTessera({kinds: [\"address\"], modelUrl, integrity, worker})` to its resolution; the package module is imported first and not timed.",
        "",
        "- same process: a new document in a browser process that already ran earlier measurements. The engine may keep compiled code or warm internal state across documents, so these rows can be optimistic.",
        "- new process: a new WebDriver session, so a new browser process and profile, before every run.",
        "",
        "Nothing is cached (`Cache-Control: no-store`), so with the worker on, the worker downloads the wasm a second time. A deployment serving the wasm cacheably would answer that second request from the HTTP cache, so the worker-on rows include a cost it would not pay. Over 127.0.0.1 that download is short, but it is not separated out.",
        "",
        "| Browser | Cold | wasm | Worker | Median ms | Min | Q1 | Q3 | Max |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        for mode, label in (("cold", "same process"), ("cold_process", "new process")):
            for key, runs in r[mode].items():
                variant, worker = key.split()
                init = [x["init"] for x in runs]
                q1, q2, q3 = statistics.quantiles(init, n=4, method="inclusive")
                lines.append(
                    f"| {name} | {label} | {variant} | {worker} | {ms(q2)} | {ms(min(init))} | {ms(q1)} | {ms(q3)} | {ms(max(init))} |"
                )
    titles = (
        ("parse", "Warm parseAddress"),
        ("detect", "Warm detect, emails and phones"),
        ("detect_model", f"Warm detect with the model, every kind, {LONG_CHARS:,}-character document"),
    )
    for mode, title in titles:
        calls = len(next(iter(next(iter(results.values()))[mode].values()))["times"])
        lines += [
            "",
            f"## {title} ({calls} calls after one warm-up pass)",
            "",
            "p95 is nearest-rank: the value at position ceil(0.95 n) of the n sorted times.",
            "",
            "| Browser | wasm | Worker | Median ms | p95 ms | Max ms | Mean ms (total / runs) | Timer resolution ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, r in results.items():
            for key, run in r[mode].items():
                variant, worker = key.split()
                t = run["times"]
                lines.append(
                    f"| {name} | {variant} | {worker} | {ms(statistics.median(t))} | {ms(p95(t))} | {ms(max(t))} "
                    f"| {ms(run['total'] / len(t))} | {run['resolution']:.3f} |"
                )
    lines += [
        "",
        "## Memory added to the page (measureUserAgentSpecificMemory)",
        "",
        "Bytes over a sample taken before the package is imported; worker memory is included. n/a where the browser lacks the API.",
        "",
        "| Browser | Mode | wasm | Worker | After load | After the runs |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for name, r in results.items():
        for mode in ("parse", "detect", "detect_model"):
            for key, run in r[mode].items():
                variant, worker = key.split()
                lines.append(f"| {name} | {mode} | {variant} | {worker} | {mb(run.get('memLoad'))} | {mb(run.get('memParsed'))} |")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1]
        del args[i : i + 2]
    names = args or ["chrome", "firefox", "safari"]

    handler = functools.partial(Handler, directory=ROOT)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}{PAGE}"

    text, hint = email_sample()
    inputs = addresses()
    long_text = text
    while len(long_text) < LONG_CHARS:
        long_text += "\n\n" + text
    long_text = long_text[:LONG_CHARS]
    base_cfg = {
        "modelUrl": f"/{BUNDLE}",
        # The digest format Tessera::load verifies: sha256- and lowercase hex.
        "integrity": "sha256-" + hashlib.sha256(read(BUNDLE)).hexdigest(),
        "addresses": inputs,
        "runs": WARM_PASSES * len(inputs),
        "text": text,
        "longText": long_text,
        "modelRuns": MODEL_RUNS,
        "countryHint": [hint],
    }
    environment = {
        "machine": subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip(),
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "memory": f"{int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 2**30:.0f} GiB",
        "os": f"macOS {platform.mac_ver()[0]}",
        "form factor": "desktop only; no mobile device measured",
        "server": "127.0.0.1, so download time is local and not a network estimate",
    }
    size_rows = sizes()
    results = {}
    for name in names:
        print(f"measuring {name}", file=sys.stderr)
        results[name] = run_browser(name, url, base_cfg)
    server.shutdown()

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"sizes": size_rows, "environment": environment, "browsers": results}, f, indent=1)
    print(report(size_rows, results, environment))
    problems = check(results)
    for p in problems:
        print(p, file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
