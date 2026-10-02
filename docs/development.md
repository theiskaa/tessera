# Development

[README](../README.md) · [Getting started](getting-started.md) · [Model](model.md)

## Repository map

| Path | Purpose |
| --- | --- |
| `tessera/` | Runtime library, native CLI, WebAssembly bindings, and JavaScript wrapper |
| `trainer/` | Data preparation, training, validation, evaluation, and model packaging |
| `site/` | Leptos demo with inference in a browser worker |
| `models/` | Distributed bundle, checksum, and parser/detector golden vectors |
| `fixtures/` | Small checked-in examples for rules, parsing, detection, and grouping |
| `configs/`, `data/manifests/` | Training configuration and data-source metadata |
| `bench/` | Runtime benchmarks and data checks |

Raw data, processed shards, checkpoints, and run evidence stay in ignored local directories. A checkout includes runtime fixtures and the distributed model; it does not include the prepared training corpus.

## Native checks

The toolchain is pinned in [rust-toolchain.toml](../rust-toolchain.toml). Run from the repository root:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace
```

`just` runs the same formatting, lint, and native-test recipes. To inspect the Rust API locally:

```sh
cargo doc -p tessera --no-deps --open
```

Golden vectors bind the tokenizer, features, and network outputs to the bundled weights. When replacing a model, its checksum and golden vectors must be updated together.

## JavaScript and WebAssembly

Install `just`, `wasm-pack`, and Binaryen's `wasm-opt`; the pinned Rust toolchain includes the `wasm32-unknown-unknown` target. Node is needed for package assembly, and Bun is also used by the JavaScript test recipe.

```sh
just wasm
just js-test
just test-web-ci
```

`just wasm` writes `tessera/pkg/`: baseline and SIMD modules, their generated glue, the wrapper, worker, and TypeScript definitions. The wrapper chooses SIMD when the runtime supports it. The generated package directory is ignored by Git.

`just js-test` rebuilds the package and checks it in Node and Bun. `just test-web-ci` runs the headless Chrome tests; `just wasm-test firefox` selects Firefox instead. The browser and its corresponding test driver must be available.

## Run the site

Install Trunk in addition to the Rust/WebAssembly tools.

```sh
just site
```

The output is `site/dist/`, including the page, inference worker, and model bundle. For a local development server:

```sh
cd site
trunk serve
```

Open `http://localhost:8080`. The worker fetches the bundle once and checks it against `models/tessera-v1.sha256`, embedded at build time. Text is analyzed in the browser.

## Training code

The trainer is a separate application; it is not part of the runtime package. Inspect its actual commands before running a workflow:

```sh
cargo run -p trainer -- --help
cargo run -p trainer -- check-detector-data --help
```

`check-detector-data` validates prepared inputs without initializing a model. It needs the data and manifests referenced by its configuration. The public [US data checks](../bench/us/README.md) describe those prerequisites and checks.

The hard-objective lane binds corrected labels, full-document selector masks, planned batches, and completed coefficients through reviewed receipts. Its source checks deliberately reject stale producer bindings. Portable tests cover these invariants; historical experiment scripts and proof-generation fixtures remain archived locally.

In `trainer/src/fullmix_training_seen/`, `mod.rs` contains common membership and metrics code, `legacy_bindings.rs` reads the older plan contract, and `hard_bindings.rs` reads the corrected hard-objective contract. Their schema identifiers remain stable even though filenames describe their roles.

## Licensing

Keep [NOTICE](../NOTICE) with redistributed model weights. The code license and training-source terms are documented separately there and in the two root license files. Publicly accessible source documents still retain their applicable attribution and licensing terms.
