// Package entry: instantiates the wasm module once, then hands everything else to the Rust
// `createInstance`: option parsing, fetching, verification, and the optional worker.
import init, { createInstance } from "./tessera.js";

// A 31-byte module with one function returning a v128: validate() is false on engines without
// SIMD and instantiates nothing.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

let ready;

export async function createTessera(options = {}) {
  if (typeof WebAssembly !== "object") {
    throw unsupported("WebAssembly is not available in this runtime");
  }
  const variant = WebAssembly.validate(SIMD_PROBE) ? "./tessera_simd_bg.wasm" : "./tessera_bg.wasm";
  const wasmUrl = new URL(variant, import.meta.url);
  // Node cannot fetch a file: URL, so the module is read from disk there.
  const wasmSource = () =>
    wasmUrl.protocol === "file:" ? import("node:fs/promises").then((fs) => fs.readFile(wasmUrl)) : wasmUrl;
  ready ??= init({ module_or_path: wasmSource() }).catch((e) => {
    ready = undefined;
    throw unsupported(`could not load the WebAssembly module: ${e?.message ?? e}`);
  });
  await ready;
  return createInstance({
    ...options,
    wasmUrl: wasmUrl.href,
    workerUrl: new URL("./worker.js", import.meta.url).href,
  });
}

function unsupported(message) {
  return Object.assign(new Error(message), { name: "TesseraError", code: "UNSUPPORTED_RUNTIME" });
}
