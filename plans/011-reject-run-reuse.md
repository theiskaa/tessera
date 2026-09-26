# Plan 011: Refuse a training run directory that already holds a checkpoint

> **Executor instructions**: Make the run start fail before changing any artifact when the named directory already contains a trained checkpoint. Preserve the current V18 process and its run directory. If the live code differs from the excerpt below, stop and report.
>
> **Drift check (run first)**: `git diff --stat 2361304..HEAD -- trainer/src/train.rs trainer/src/quantize.rs trainer/src/export.rs` and `git diff -- trainer/src/train.rs trainer/src/quantize.rs trainer/src/export.rs`. This plan includes uncommitted local artifact-gate work; compare the excerpts below before editing.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW; reruns under a used name will now require a new name
- **Depends on**: Phase 004 artifact gate already present
- **Category**: correctness / artifact integrity
- **Planned at**: commit `2361304`, 2026-09-26

## Why this matters

`initialize_run` removes the old quantization gate and overwrites the effective config, but leaves `best.mpk`. If loading data or training fails before the first new checkpoint, a later `trainer quantize --run` can quantize that previous checkpoint under the new config if shapes match. The newly bound gate would then attest to current files while the weights were trained with a different config. The simplest safe rule is to refuse a used run directory before any mutation; a new experiment must use a new run name.

## Current state

- `trainer/src/train.rs:112-137`: `train` derives `runs/<cfg.name>` and calls `initialize_run` before loading training data. `initialize_run` creates `checkpoints`, invalidates `quantize.json`, then writes `config.toml`; it does not reject an existing `best.mpk`.
- `trainer/src/train.rs:370-443`: `fit` writes `best.mpk` only after an epoch validates. Failure before that point leaves a previous file untouched.
- `trainer/src/quantize.rs:484-534`: quantization loads the run's current `config.toml` and `best.mpk`, then can write a passing bound gate. The gate hashes files; it cannot infer which config generated an old checkpoint.
- `trainer/src/train.rs:458-478`: existing temp-dir test checks gate removal on reuse; its expectation must change with the new invariant.
- Repo conventions: use `anyhow::ensure!` or `bail!` with a clear path and new-name instruction; unit tests in the same module; `cargo fmt`, Clippy, and `cargo test -p trainer` are gates. `CLAUDE.md` says no commit unless asked.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer reused_run` | Both reused-run cases pass |
| Trainer suite | `cargo test -p trainer` | All pass |
| Format | `cargo fmt --all --check` | Exit 0 |
| Lint | `cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Diff | `git diff --check` | Exit 0 |

## Scope

**In scope:** `trainer/src/train.rs` and its inline tests; `plans/011-reject-run-reuse.md`; `plans/README.md` status.

**Out of scope:** current or historical `runs/` contents; trainer config, checkpoint format, quantizer, exporter, data, shipped `models/`, site. Do not delete files as a workaround.

## Steps

### 1. Reject existing checkpoint before mutation

At the start of `initialize_run`, check for `best.mpk` and existing checkpoint files under `checkpoints/`. If either exists, return an error naming the run directory and instructing a new run name. Check before `create_dir_all`, gate invalidation, or config write. Keep allowing a directory with no checkpoint artifacts, such as one that failed before training. Do not remove old files automatically.

**Verify:** `cargo test -p trainer reused_run` passes the new and adjusted tests.

### 2. Test nonmutation and new-run behavior

Replace the existing reuse test with two cases: (a) directory with `best.mpk`, old config, quantized weights, and passing gate is refused with every byte unchanged; (b) directory with config and gate but no checkpoint is allowed and invalidates the gate before writing the new config. Also cover a checkpoint file under `checkpoints/` without `best.mpk`.

**Verify:** `cargo test -p trainer reused_run` passes and reports at least three focused cases.

### 3. Review

Run the suite, formatting, lint, and diff commands. Inspect the diff for exact scope and record `plans/011-review.md`. Mark the README row DONE.

**Verify:** all commands pass; no current V18 or shipped artifacts changed.

## Done criteria

- [ ] A run directory containing a checkpoint cannot be reused by `trainer train`.
- [ ] Rejection occurs before any config/gate/filesystem mutation.
- [ ] A directory without a checkpoint can be resumed/restarted as before.
- [ ] Focused and full trainer tests, format, lint, and diff checks pass.

## STOP conditions

- Checkpoint naming or directory structure differs from the current `best.mpk` and `checkpoints/epoch-*` convention.
- The change would delete a user's run or alter the current V18 process.
- A required workflow intentionally resumes training from an old checkpoint using this same command; design an explicit resume contract first.

## Maintenance note

Keep this guard if checkpoint naming changes. An explicit resume feature, if later added, needs checkpoint/config provenance stored at checkpoint creation and a separate command path.
