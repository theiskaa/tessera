// Per-stage timings of the `wasm,profile` build in bench/web/pkg, on the two profiling fixtures.
// Open /bench/web/ from a server at the repository root; profile.py drives it through WebDriver.
import init, { Tessera } from "./pkg/tessera.js";

const ITERATIONS = 200;
const STAGES = ["tokenize_ms", "featurize_ms", "rules_ms", "detect_ms", "parse_ms", "group_ms", "total_ms"];
// The smallest module using a simd128 instruction; validates only where SIMD is supported.
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

async function fetched(path, as) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return as === "bytes" ? new Uint8Array(await r.arrayBuffer()) : r.text();
}

function row(name, t, p95) {
  const cells = STAGES.map((k) => `<td>${t[k].toFixed(3)}</td>`).join("");
  return `<tr><td>${name}</td>${cells}<td>${p95.toFixed(3)}</td></tr>`;
}

function simd() {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return null;
  }
}

async function profile() {
  await init();
  const tessera = Tessera.load(await fetched("/models/tessera-v1.safetensors", "bytes"));
  const address = (await fetched("/fixtures/profile/address-gb.txt")).trimEnd();
  const doc = await fetched("/fixtures/profile/document-10k.txt");
  const now = performance.now.bind(performance);
  const before = tessera.memoryBytes();
  const a = tessera.profileAddress(address, ITERATIONS, now);
  const d = tessera.profileStages(doc, ITERATIONS, now);
  const after = tessera.memoryBytes();
  tessera.free();
  return {
    ua: navigator.userAgent,
    simd: simd(),
    iterations: ITERATIONS,
    address: a,
    document: d,
    wasm_memory_bytes_before: before,
    wasm_memory_bytes_after: after,
  };
}

const status = document.getElementById("status");
window.profileResult = profile().then(
  (result) => {
    document.getElementById("table").innerHTML =
      `<tr><th>input</th>${STAGES.map((k) => `<th>${k.replace("_ms", "")}</th>`).join("")}<th>p95 total</th></tr>` +
      row("address-gb", result.address.median, result.address.p95_total) +
      row("document-10k", result.document.median, result.document.p95_total);
    document.getElementById("json").textContent = JSON.stringify(result, null, 2);
    status.textContent = "done";
    return result;
  },
  (e) => {
    status.textContent = `error: ${e && e.message ? e.message : e}`;
    return { error: String(e && e.message ? e.message : e) };
  },
);
