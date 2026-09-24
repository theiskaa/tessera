#!/usr/bin/env bash
# Writes target/sizes.json with the gzip -9 sizes the spec tracks: both wasm variants of the
# package (run `just wasm` first), the weight bundle, and the phone metadata. The phone tables are
# Rust source, so their size is the baseline wasm minus the same build without `phone-metadata`,
# built here with the package's size profile and wasm-opt flags.
set -euo pipefail
cd "$(dirname "$0")/.."

gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }

wasm_simd=$(gz tessera/pkg/tessera_simd_bg.wasm)
wasm_nosimd=$(gz tessera/pkg/tessera_bg.wasm)
bundle=$(gz models/tessera-v1.safetensors)

nophone=target/pkg-nophone
env CARGO_PROFILE_RELEASE_OPT_LEVEL=z CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
  CARGO_PROFILE_RELEASE_PANIC=abort CARGO_PROFILE_RELEASE_STRIP=true \
  wasm-pack --quiet build --release --target web --out-dir "../$nophone" tessera --no-default-features --features wasm >/dev/null
wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory --enable-nontrapping-float-to-int \
  --enable-sign-ext --enable-mutable-globals --enable-reference-types --enable-multivalue \
  -o "$nophone/tessera_bg.wasm" "$nophone/tessera_bg.wasm"
phone=$((wasm_nosimd - $(gz "$nophone/tessera_bg.wasm")))
rm -rf "$nophone"

mkdir -p target
jq -n \
  --argjson wasm_simd_gz "$wasm_simd" \
  --argjson wasm_nosimd_gz "$wasm_nosimd" \
  --argjson bundle_gz "$bundle" \
  --argjson phone_metadata_gz "$phone" \
  '{wasm_simd_gz: $wasm_simd_gz, wasm_nosimd_gz: $wasm_nosimd_gz, bundle_gz: $bundle_gz, phone_metadata_gz: $phone_metadata_gz}' |
  tee target/sizes.json
