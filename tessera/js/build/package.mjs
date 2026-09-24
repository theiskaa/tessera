// Post-build step, never shipped: makes index.js the package entry and index.d.ts its types,
// and leaves tessera/pkg holding exactly the files the package ships plus package.json.
import { copyFile, readFile, rm, writeFile } from "node:fs/promises";

const pkgDir = new URL("../../pkg/", import.meta.url);
const shipped = [
  "index.js",
  "index.d.ts",
  "worker.js",
  "tessera.js",
  "tessera_bg.wasm",
  "tessera_simd.js",
  "tessera_simd_bg.wasm",
];

const path = new URL("package.json", pkgDir);
const pkg = JSON.parse(await readFile(path, "utf8"));
Object.assign(pkg, {
  type: "module",
  main: "index.js",
  module: "index.js",
  types: "index.d.ts",
  exports: { ".": { types: "./index.d.ts", default: "./index.js" }, "./package.json": "./package.json" },
  files: shipped,
  sideEffects: ["./worker.js"],
});
await writeFile(path, JSON.stringify(pkg, null, 2) + "\n");

for (const file of ["index.js", "index.d.ts", "worker.js"]) {
  await copyFile(new URL(`../${file}`, import.meta.url), new URL(file, pkgDir));
}
// wasm-pack's own typings and ignore file; index.d.ts replaces the first, `files` the second.
for (const file of ["tessera.d.ts", "tessera_bg.wasm.d.ts", ".gitignore"]) {
  await rm(new URL(file, pkgDir), { force: true });
}
