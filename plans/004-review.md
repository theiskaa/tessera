# Phase 004 review — artifact-bound export gate

Completed 2026-09-26. No model training, bundle export, or site update ran.

## Changes

- Training initialization removes any old `quantize.json` before writing the effective config. Existing quantized bytes remain available for inspection, but have no approval.
- Quantization removes prior approval before loading inputs, writes weights to a pending path, and scores them before publication. A rejected score writes a failed gate and keeps rejected weights. Passing publication hashes the current `config.toml`, `best.mpk`, and pending safetensors, verifies that config and checkpoint have not changed during validation, renames weights, then atomically renames the versioned gate JSON. Failure before the last rename leaves no passing gate.
- Export checks the gate version, `passed` flag, and all three SHA-256 hashes for each run before reading tensors. Legacy unbound gates require requantization.

## Review and verification

- Burn 0.21's `NamedMpkFileRecorder` sets the `.mpk` extension and writes one file, so `best.mpk` is the complete checkpoint used by `load_best`.
- Focused quantize tests: 9 passed. Export tests: 5 passed. Training initialization test: 1 passed. Full trainer suite: 163 passed.
- `cargo fmt --all --check`, `cargo clippy -p trainer --all-targets -- -D warnings`, and `git diff --check` passed.
- Tests cover passing publication, rejected score, an injected gate-write failure after weight publication, changed checkpoint during quantization, changed config/checkpoint/weights for both parser and detector, a reused run, and a legacy gate.
- `git status --short` showed no modifications in `models/` or `site/`.

## Operational consequence

Existing `quantize.json` files are legacy gates. Any future `trainer export` must first rerun `trainer quantize` for both parser and detector runs against their current checkpoints. This does not retrain weights. Phase 005 must account for the parser run as well as the staged detector run.
