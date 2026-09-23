// Starts the inference worker. A failed wasm load is rethrown as an uncaught error, which the
// browser re-raises on the page's window; a rejected promise would stay inside the worker.
importScripts("./worker.js");
wasm_bindgen({ module_or_path: "./worker_bg.wasm" }).catch((e) => setTimeout(() => { throw e; }));
