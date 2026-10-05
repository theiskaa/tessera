import { createHash } from "node:crypto";
import { readFile, readdir, stat } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { createTessera } from "../../tessera/pkg/index.js";
import { fromUtf8Case, scoreCases, validateGold } from "./scoring.mjs";

const inputPath = process.argv[2];
if (!inputPath || process.argv.length !== 3) throw new Error("expected one native JSON fixture file, fixture directory, or gold JSONL path");
const root = new URL("../../", import.meta.url);
const path = resolve(inputPath);
const files = (await stat(path)).isDirectory()
  ? (await readdir(path)).filter((name) => name.endsWith(".json")).sort().map((name) => resolve(path, name))
  : [path];
const cases = [];
const inputs = [];
const names = new Set();
for (const file of files) {
  const bytes = await readFile(file);
  const text = bytes.toString("utf8");
  const parsed = file.endsWith(".jsonl") ? text.split(/\r?\n/u).filter((line) => line.trim()).map((line) => JSON.parse(line)) : JSON.parse(text).cases;
  if (!Array.isArray(parsed)) throw new Error(`${file}: no cases`);
  inputs.push({ path: file, sha256: createHash("sha256").update(bytes).digest("hex") });
  for (const c of parsed) {
    const gold = fromUtf8Case({ ...c, name: files.length > 1 ? `${basename(file)}:${c.name}` : c.name });
    validateGold(gold);
    if (names.has(gold.name)) throw new Error(`duplicate case name: ${gold.name}`);
    names.add(gold.name);
    if (gold.country && gold.country !== "US") throw new Error(`${gold.name}: expected US evaluation data`);
    cases.push(gold);
  }
}
if (cases.length === 0) throw new Error("empty evaluation corpus");
const modelBytes = await readFile(new URL("models/tessera-v1.safetensors", root));
const integrity = (await readFile(new URL("models/tessera-v1.sha256", root), "utf8")).trim();
const packageHashes = {};
for (const name of ["index.js", "tessera.js", "tessera_bg.wasm", "tessera_simd.js", "tessera_simd_bg.wasm"]) {
  packageHashes[name] = createHash("sha256").update(await readFile(new URL(`tessera/pkg/${name}`, root))).digest("hex");
}
const query = { countryHint: ["US"], includeUncertain: false, format: "text" };
const model = await createTessera({ modelBytes, integrity });
const rows = [];
try {
  for (const gold of cases) {
    const detected = await model.detect(gold.input, query);
    const extraction = gold.contacts === undefined ? undefined : await model.extractContacts(gold.input, query);
    rows.push({ gold, detected, extraction });
  }
  console.log(JSON.stringify({
    ...scoreCases(rows),
    provenance: { inputs, bundleSha256: createHash("sha256").update(modelBytes).digest("hex"), packageHashes, query, runtime: `node ${process.version}; public JavaScript/WASM API` },
    predictions: rows.map(({ gold, detected, extraction }) => ({ name: gold.name, detected, extraction })),
  }, null, 2));
} finally {
  model.dispose();
}
