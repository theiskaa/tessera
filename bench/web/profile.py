"""Runs the profiling page bench/web/index.html in Chrome, Firefox and Safari through WebDriver
and prints each browser's per-stage medians as a markdown table and the raw JSON. Build the
profile variant first, with the shipped size profile:

    just profile-web          # simd128 build in bench/web/pkg
    python3 bench/web/profile.py [chrome] [firefox] [safari] [--label TEXT]

Reuses measure.py's server and drivers; macOS only, like it.
"""

import functools
import http.server
import json
import sys
import threading

from measure import Driver, Handler, ROOT, browsers

STAGES = ["tokenize_ms", "featurize_ms", "rules_ms", "detect_ms", "parse_ms", "group_ms", "total_ms"]


def run(name, url):
    command, port, capabilities = browsers()[name]
    if not command[0]:
        raise RuntimeError(f"{name}: no driver found")
    driver = Driver(name, command, port, capabilities)
    try:
        driver.call("POST", f"/session/{driver.session}/url", {"url": url})
        script = "const done = arguments[0]; (function go() { window.profileResult ? window.profileResult.then(done) : setTimeout(go, 20); })();"
        result = driver.call("POST", f"/session/{driver.session}/execute/async", {"script": script, "args": []})
        if "error" in result:
            raise RuntimeError(f"{name}: {result['error']}")
        result["browser_version"] = driver.version
        return result
    finally:
        driver.quit()


def table(result):
    rows = ["| input | tokenize | featurize | rules | detect | parse | group | total | p95 total |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, key in [("address-gb", "address"), ("document-10k", "document")]:
        m = result[key]["median"]
        cells = " | ".join(f"{m[s]:.3f}" for s in STAGES)
        rows.append(f"| {name} | {cells} | {result[key]['p95_total']:.3f} |")
    return "\n".join(rows)


def main():
    args = sys.argv[1:]
    label = ""
    if "--label" in args:
        i = args.index("--label")
        label = args[i + 1]
        del args[i : i + 2]
    names = args or ["chrome", "firefox", "safari"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=ROOT))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/bench/web/index.html"
    for name in names:
        try:
            result = run(name, url)
        except Exception as e:  # a browser that cannot start is reported, not fatal
            print(f"## Browser: {name}{label}\n\nnot run: {e}\n")
            continue
        print(f"## Browser: {name} {result['browser_version']}{label}\n")
        print(f"simd128 supported: {result['simd']}; wasm memory {result['wasm_memory_bytes_before']} -> {result['wasm_memory_bytes_after']} bytes\n")
        print(table(result) + "\n")
        print("```json\n" + json.dumps(result, indent=1) + "\n```\n")
    server.shutdown()


if __name__ == "__main__":
    main()
