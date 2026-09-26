# Plan 004: Refuse export when quantization approval is stale

> **Executor instructions:** Run the drift check, apply the narrow artifact-gate
> change, and verify the failure cases. Do not run training, re-export into
> `models/`, or update the site. Stop on the conditions below and update
> `plans/README.md` only after verification passes.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- trainer/src/train.rs trainer/src/quantize.rs trainer/src/export.rs plans/README.md`.
> If these files changed, re-read the cited functions before editing.

## Status

- **Priority:** P1
- **Effort:** S
- **Risk:** LOW; old unbound gates will require fresh quantization before export
- **Depends on:** none
- **Category:** correctness, reproducibility
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

`trainer train --name EXISTING` reuses a directory and overwrites its effective
config and `best.mpk`. An earlier `quantize.json` with `passed=true` and
`quantized.safetensors` can remain. `trainer export` checks only that flag, so it
can package weights from a previous run under current metadata. V17's scripted
path quantized freshly and is not implicated, but future comparisons must not
depend on operators remembering that sequence.

## Current state

- `trainer/src/train.rs:118-122` creates a run directory and overwrites
  `config.toml`; `:419-426` overwrites `best.mpk` when validation improves.
- `trainer/src/quantize.rs:389-429` writes `quantized.safetensors` before its
  validation score, then writes `quantize.json` with `passed` and metrics. A
  failure before that JSON write can leave an older passing gate in place.
- `trainer/src/export.rs:199-223` checks `gate["passed"] == true` and loads the
  quantized file without binding it to the current config or checkpoint.
- The trainer already depends on `sha2` (`trainer/Cargo.toml:26`) and has a
  SHA-256 helper pattern in `trainer/src/export.rs:77-83`. Use one shared helper
  rather than a second checksum format.
- Match `CLAUDE.md`: `anyhow` errors in trainer, private helpers, no commit
  unless asked, never push. Keep `models/` and site unchanged without explicit
  user approval.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer quantize` and `cargo test -p trainer export` | Both exit 0 |
| Format and lint | `cargo fmt --all --check && cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Working-tree check | `git status --short` | No change under `models/` or `site/` |

## Scope

**In scope:** `trainer/src/train.rs`, `trainer/src/quantize.rs`,
`trainer/src/export.rs`, tests in those modules, and the status row in
`plans/README.md`.

**Out of scope:** model architecture, the quantization algorithm or allowed
F1 drop, bundle format, training data, `models/`, site, and a real training run.

## Git workflow

Do not commit or push unless asked. If asked later, use a lowercase single-line
commit message with no trailer.

## Steps

### 1. Invalidate old approval as soon as a run is reused

In `train.rs`, remove or mark invalid any existing quantization gate before
writing a new effective config or starting fitting. Preserve old quantized
bytes for forensic inspection if desired, but make `export` unable to accept
them. An aborted retrain must leave the run non-exportable until a new gate
passes.

**Verify:** A focused test using a temporary run directory with a previously
passing gate shows the gate is invalidated at training initialization, even if
fitting is not run. `cargo test -p trainer quantize` exits 0.

### 2. Write an artifact-bound gate atomically

Quantize into a temporary safetensors path, validate it, and only on success
publish `quantized.safetensors` and a `quantize.json` containing SHA-256 of the
effective `config.toml`, `best.mpk`, and quantized file. Keep the existing score
fields. If quantization or validation fails, no `passed=true` gate may remain;
keep rejected bytes under the existing rejected naming convention if useful.
Use temp paths in the same run directory so rename is atomic on one filesystem.

**Verify:** Tests cover success, validation failure, and an injected failure
between writing temporary weights and publishing the gate. Each failure leaves
export unable to pass. `cargo test -p trainer quantize` exits 0.

### 3. Verify hashes during export

In `export.rs::load_run`, require the gate version, `passed=true`, and all three
hashes equal to the current file bytes before reading quantized tensors.
Reject legacy gates without hashes with a clear “requantize this run” error.
Check both parser and detector runs separately. Do not silently accept by
mtime, filename, or run name.

**Verify:** Tests mutate config, `best.mpk`, and `quantized.safetensors` one at
a time in temporary runs. Each mutation makes the gate fail; untouched files
pass. `cargo test -p trainer export` and format/lint commands exit 0.

## Test plan

Use temporary files with small known bytes for hash-gate unit tests; the
tests need not create a full network checkpoint. Factor hash verification so
it can be tested without loading Burn. Test a legacy gate and a reused-run
gate explicitly. Do not call GPU training from tests.

## Done criteria

- [x] A reused run directory cannot export on an old quantization gate.
- [x] Quantization failures cannot leave a passing gate for mismatched bytes.
- [x] Export verifies config, checkpoint, and quantized-file hashes for both nets.
- [x] Legacy unbound gates fail with actionable error text.
- [x] Focused tests, format, and lint commands pass.
- [x] No training ran; shipped model and site remain unchanged.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- The actual checkpoint filename is not `best.mpk` or the recorder writes
  multiple files; identify and hash the complete checkpoint instead of guessing.
- A change would require altering the shipped bundle's binary layout.
- Existing tooling requires exporting legacy runs without requantization; report
  the exact workflow rather than adding an unchecked bypass.

## Maintenance notes

Export provenance must bind the bytes that were actually validated, not only
the run directory's name. A later change to the quantization metric should
increment the gate version and keep old gates fail-closed.
