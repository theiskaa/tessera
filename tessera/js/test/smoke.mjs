// Loads the built package the way a server would and checks that every offset slices the
// original string to the reported text. Run with node or bun after `just wasm`.
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { createTessera } from "../../pkg/index.js";

const root = new URL("../../../", import.meta.url);
const read = (path) => readFile(new URL(path, root), "utf8");
const modelBytes = await readFile(new URL("models/tessera-v1.safetensors", root));
const integrity = (await read("models/tessera-v1.sha256")).trim();

function assertSlices(input, entity, name) {
  assert.equal(input.slice(entity.start, entity.end), entity.text, name);
  for (const c of entity.components ?? []) {
    assert.equal(input.slice(c.start, c.end), c.text, `${name}: ${c.label}`);
  }
}

// A module that fails to instantiate rejects, and the next call loads it afresh. Only our module
// is blocked, told apart by its glue's import namespace (`./tessera_bg.js` or
// `./tessera_simd_bg.js`): the runtime may instantiate modules of its own, such as its HTTP parser.
const { instantiate } = WebAssembly;
const ours = (imports) => Object.keys(imports ?? {}).some((name) => name.startsWith("./tessera"));
WebAssembly.instantiate = (source, imports) =>
  ours(imports) ? Promise.reject(new Error("blocked by the smoke test")) : instantiate(source, imports);
await assert.rejects(createTessera({ kinds: ["email"] }), { name: "TesseraError", code: "UNSUPPORTED_RUNTIME" });
WebAssembly.instantiate = instantiate;

// Outside a browser main thread `worker: true` is ignored and the instance runs inline.
const tessera = await createTessera({ modelBytes, integrity, kinds: ["address"], worker: true });
assert.deepEqual(tessera.kinds, ["address"]);

let cases = 0;
let components = 0;
for (const file of (await readdir(new URL("fixtures/parser/", root))).filter((f) => f.endsWith(".json"))) {
  for (const c of JSON.parse(await read(`fixtures/parser/${file}`)).cases) {
    const entity = await tessera.parseAddress(c.input);
    assert.equal(entity.kind, "address", c.name);
    assertSlices(c.input, entity, `${file}: ${c.name}`);
    assert.ok(entity.components.length > 0, `${file}: ${c.name}: no components`);
    cases += 1;
    components += entity.components.length;
  }
}
assert.ok(cases >= 40, `only ${cases} fixture cases`);

// Characters outside the BMP take two UTF-16 units and four UTF-8 bytes.
const astral = "🏠🏠 10 Downing Street, London SW1A 2AA";
const home = await tessera.parseAddress(astral);
assert.ok(home.components.length > 0, "astral: no components");
assertSlices(astral, home, "astral");

await assert.rejects(tessera.parseAddress("x ".repeat(300)), { name: "TesseraError", code: "INPUT_TOO_LARGE" });
tessera.dispose();
await assert.rejects(tessera.parseAddress("221B Baker Street"), { code: "DISPOSED" });
await assert.rejects(createTessera({ modelBytes, integrity: "sha256-0000", kinds: ["address"] }), {
  code: "CHECKSUM_MISMATCH",
});

const rules = await createTessera({ kinds: ["email", "phone"] });
for (const c of JSON.parse(await read("fixtures/rules/documents.json")).cases) {
  const found = await rules.detect(c.input, { countryHint: c.country_hint });
  const shape = (e) => [e.kind, e.text, e.normalized];
  assert.deepEqual(found.map(shape), c.expected.map(shape), c.name);
  for (const e of found) assertSlices(c.input, e, c.name);
}
// The package is built without the `markdown` feature.
await assert.rejects(rules.detect("[a](mailto:a@b.example)", { format: "markdown" }), {
  name: "TesseraError",
  code: "UNSUPPORTED_FORMAT",
});
await assert.rejects(rules.detect("x", { format: "html" }), TypeError);
assert.equal((await rules.detect("a@b.example", { format: "text" })).length, 1);
rules.dispose();

const runtime = typeof Bun === "undefined" ? `node ${process.version}` : `bun ${Bun.version}`;
console.log(`smoke: ${components} address components sliced correctly on ${runtime}`);
