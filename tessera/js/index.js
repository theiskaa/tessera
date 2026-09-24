// Package entry: instantiates the wasm module once, then hands everything else to the Rust
// `createInstance`: option parsing, fetching, verification, and the optional worker. Each wasm
// variant has its own glue, since wasm-bindgen names functions in it by index.

// A 31-byte module with one function returning a v128: validate() is false on engines without
// SIMD and instantiates nothing.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

// Not a literal, so browser bundlers leave the Node-only import alone instead of failing to
// resolve the `node:` scheme; webpack and vite are told to skip it so they do not warn either.
const NODE_FS = "node:fs/promises";

let ready;

export async function createTessera(options = {}) {
  if (typeof WebAssembly !== "object") {
    throw unsupported("WebAssembly is not available in this runtime");
  }
  // Both URLs are written out in full: bundlers emit an asset only for a literal
  // `new URL("…", import.meta.url)`, and would drop a module named by an expression.
  const simd = WebAssembly.validate(SIMD_PROBE);
  const wasmUrl = simd
    ? new URL("./tessera_simd_bg.wasm", import.meta.url)
    : new URL("./tessera_bg.wasm", import.meta.url);
  // Node cannot fetch a file: URL, so the module is read from disk there.
  const wasmSource = () =>
    wasmUrl.protocol === "file:"
      ? import(/* webpackIgnore: true */ /* @vite-ignore */ NODE_FS).then((fs) => fs.readFile(wasmUrl))
      : wasmUrl;
  ready ??= (simd ? import("./tessera_simd.js") : import("./tessera.js"))
    .then(async (glue) => {
      await glue.default({ module_or_path: wasmSource() });
      return glue;
    })
    .catch((e) => {
      ready = undefined;
      throw unsupported(`could not load the WebAssembly module: ${e?.message ?? e}`);
    });
  const { createInstance } = await ready;
  return createInstance({
    ...options,
    wasmUrl: wasmUrl.href,
    workerUrl: workerUrl(simd),
  });
}

// The worker loads the same variant's glue; a bundler may rename the wasm, so it is told which.
function workerUrl(simd) {
  const url = new URL("./worker.js", import.meta.url);
  if (simd) url.searchParams.set("simd", "");
  return url.href;
}

function unsupported(message) {
  return Object.assign(new Error(message), { name: "TesseraError", code: "UNSUPPORTED_RUNTIME" });
}
