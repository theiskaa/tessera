// Package entry: instantiates the wasm module once, then hands everything else to the Rust
// `createInstance`: option parsing, fetching, verification, and the optional worker.
import init, { createInstance } from "./tessera.js";

let ready;

export async function createTessera(options = {}) {
  if (typeof WebAssembly !== "object") {
    throw unsupported("WebAssembly is not available in this runtime");
  }
  const wasmUrl = new URL("./tessera_bg.wasm", import.meta.url);
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
