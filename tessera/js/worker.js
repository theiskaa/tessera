// Worker bootstrap: instantiates the same wasm module as the page through that module's glue,
// holds one Tessera, and answers each { id, op, ... } with { id, ok, result | error }.
// Everything with logic in it is Rust.
let tessera;

self.onmessage = async ({ data: { id, op, text, options, bytes, wasmUrl } }) => {
  try {
    let result;
    if (op === "load") {
      const simd = new URL(self.location.href).searchParams.has("simd");
      const glue = await (simd ? import("./tessera_simd.js") : import("./tessera.js"));
      await glue.default({ module_or_path: wasmUrl });
      tessera = glue.Tessera.load(bytes, options);
    } else {
      result = await tessera[op](text, options);
    }
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    const error = { code: e?.code, message: e?.message ?? String(e), stage: e?.stage };
    self.postMessage({ id, ok: false, error });
  }
};
