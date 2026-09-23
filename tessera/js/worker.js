// Worker bootstrap: instantiates the same wasm module as the page, holds one Tessera, and answers
// each { id, op, ... } with { id, ok, result | error }. Everything with logic in it is Rust.
import init, { Tessera } from "./tessera.js";

let tessera;

self.onmessage = async ({ data: { id, op, text, options, bytes, wasmUrl } }) => {
  try {
    let result;
    if (op === "load") {
      await init({ module_or_path: wasmUrl });
      tessera = Tessera.load(bytes, options);
    } else {
      result = await tessera[op](text, options);
    }
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    const error = { code: e?.code, message: e?.message ?? String(e), stage: e?.stage };
    self.postMessage({ id, ok: false, error });
  }
};
