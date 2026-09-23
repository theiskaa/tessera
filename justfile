# wasm-pack builds with its own fixed modes and ignores custom cargo profiles, so the
# size-oriented release settings are applied here as profile overrides for wasm builds only;
# native release builds of the trainer keep their speed-oriented profile.
wasm_env := "CARGO_PROFILE_RELEASE_OPT_LEVEL=z CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort CARGO_PROFILE_RELEASE_STRIP=true"

# Release npm package in tessera/pkg: the size-optimized wasm run through wasm-opt, the
# generated glue, and the hand-written entry, worker, and types.
wasm:
    env {{wasm_env}} wasm-pack build --release --target web --out-dir pkg tessera --features wasm
    node tessera/js/build/package.mjs

# Gzip size of the release wasm with the phone tables, and without them.
wasm-size: wasm
    @echo "with phone tables:    $(gzip -9 -c tessera/pkg/tessera_bg.wasm | wc -c | tr -d ' ') bytes gzip"
    env {{wasm_env}} wasm-pack build --release --target web --out-dir pkg-nophone tessera --no-default-features --features wasm
    @echo "without phone tables: $(gzip -9 -c tessera/pkg-nophone/tessera_bg.wasm | wc -c | tr -d ' ') bytes gzip"
    rm -rf tessera/pkg-nophone

# Every integration test in headless browsers, e.g. `just wasm-test firefox` or
# `just wasm-test chrome firefox safari`. The runner's default budget of 20 seconds covers a whole
# test binary including the browser's start, and the worker tests compile the debug test module
# once per worker, which leaves Firefox close to that edge.
wasm-test +browsers="chrome firefox":
    WASM_BINDGEN_TEST_TIMEOUT=120 wasm-pack test --headless {{prepend("--", browsers)}} tessera --features wasm
