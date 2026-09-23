# wasm-pack builds with its own fixed modes and ignores custom cargo profiles, so the
# size-oriented release settings are applied here as profile overrides for wasm builds only;
# native release builds of the trainer keep their speed-oriented profile.
size_env := "CARGO_PROFILE_RELEASE_OPT_LEVEL=z CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort"
wasm_env := size_env + " CARGO_PROFILE_RELEASE_STRIP=true"

# `strip` drops the target-features section, so wasm-opt assumes MVP unless told what the wasm32
# target enables by default since Rust 1.82. Only the SIMD build adds `--enable-simd`: an enabled
# feature is one binaryen's optimizer may itself emit, which would break the baseline on old engines.
wasm_opt := "wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals --enable-reference-types --enable-multivalue"

# The SIMD build has its own target directory so switching RUSTFLAGS keeps both caches warm.
simd_env := "RUSTFLAGS='-C target-feature=+simd128' CARGO_TARGET_DIR='" + justfile_directory() / "target/wasm-simd'"

# Release npm package in tessera/pkg: the baseline and SIMD wasm, the generated glue, and the
# hand-written entry, worker, and types.
wasm: wasm-baseline wasm-simd wasm-package

# The baseline wasm for engines without SIMD, with the glue both variants share.
wasm-baseline:
    env {{wasm_env}} wasm-pack build --release --target web --out-dir pkg tessera --features wasm
    {{wasm_opt}} -o tessera/pkg/tessera_bg.wasm tessera/pkg/tessera_bg.wasm

# The simd128 wasm. Its glue must be identical to the baseline's, which both variants load
# through, so the build fails if it differs; it is then discarded.
wasm-simd:
    env {{wasm_env}} {{simd_env}} wasm-pack build --release --target web --out-dir pkg-simd tessera --features wasm
    mkdir -p tessera/pkg
    {{wasm_opt}} --enable-simd -o tessera/pkg/tessera_simd_bg.wasm tessera/pkg-simd/tessera_bg.wasm
    cmp tessera/pkg-simd/tessera.js tessera/pkg/tessera.js
    rm -rf tessera/pkg-simd

# Makes index.js the entry and leaves tessera/pkg holding exactly the shipped files.
wasm-package:
    node tessera/js/build/package.mjs

# Gzip size of both release variants with the phone tables, and of the baseline without them.
wasm-size: wasm
    @echo "baseline, with phone tables:    $(gzip -9 -c tessera/pkg/tessera_bg.wasm | wc -c | tr -d ' ') bytes gzip"
    @echo "simd128, with phone tables:     $(gzip -9 -c tessera/pkg/tessera_simd_bg.wasm | wc -c | tr -d ' ') bytes gzip"
    env {{wasm_env}} wasm-pack build --release --target web --out-dir pkg-nophone tessera --no-default-features --features wasm
    {{wasm_opt}} -o tessera/pkg-nophone/tessera_bg.wasm tessera/pkg-nophone/tessera_bg.wasm
    @echo "baseline, without phone tables: $(gzip -9 -c tessera/pkg-nophone/tessera_bg.wasm | wc -c | tr -d ' ') bytes gzip"
    rm -rf tessera/pkg-nophone

# Every integration test in headless browsers, e.g. `just wasm-test firefox` or
# `just wasm-test chrome firefox safari`. The runner's default budget of 20 seconds covers a whole
# test binary including the browser's start, and the worker tests compile the debug test module
# once per worker, which leaves Firefox close to that edge.
wasm-test +browsers="chrome firefox":
    WASM_BINDGEN_TEST_TIMEOUT=120 wasm-pack test --headless {{prepend("--", browsers)}} tessera --features wasm

# The built package in Node and Bun: loading from disk, UTF-16 offsets, and typed errors.
js-test: wasm
    node tessera/js/test/smoke.mjs
    bun tessera/js/test/smoke.mjs

# The showcase site in site/dist: the page, the inference worker, and the bundle. It takes
# `size_env` but not strip: stripping drops the target-features section and Trunk's wasm-opt
# then rejects `memory.copy`. Trunk.toml builds in release.
site:
    cd site && env {{size_env}} trunk build

# Package sizes, cold init, warm latency, and memory in Chrome, Firefox and Safari, e.g.
# `just measure safari` or `just measure --json out.json`.
measure *args: wasm
    python3 bench/web/measure.py {{args}}
