# Plan 009: Reuse encoded silver examples across repeated training draws

> **Executor instructions**: Follow each step and its verification before continuing. Keep the currently running V18 process and its artifacts untouched. This source change applies only to future training binaries. If a STOP condition occurs, report it instead of changing model semantics.
>
> **Drift check (run first)**: `git diff --stat 2361304..HEAD -- trainer/src/train.rs trainer/src/dataset.rs` and `git diff -- trainer/src/train.rs trainer/src/dataset.rs`. This plan was written against uncommitted local work as well as commit `2361304`; inspect both diffs before changing code. Stop if the loops quoted below have changed.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED; training order or weighting may change if the index sequence is wrong
- **Depends on**: none; execute after the running V18 diagnostic finishes
- **Category**: performance / scale
- **Planned at**: commit `2361304`, 2026-09-26

## Why this matters

The detector eagerly encodes every synthetic and silver document, then clones every silver `Encoded` example ten times for `silver_repeat = 10`. `Encoded` owns token offsets, nested n-gram ID vectors, script, shape, flags, and labels. V18 has 14,045 silver pieces, so its configured repeat creates about 140,450 stored encoded silver entries before batch copies. The live process has about 8.6 GiB RSS, though that includes the model and GPU/runtime memory and cannot be attributed solely to repeats. At 20 or 100 countries, repeat materialization compounds the data scale. Reusing the encoded silver entries can remove this cost without changing sampling weight or network output.

## Current state

- `trainer/src/train.rs:240-263`: `train_detector` loads synthetic `Encoded` items, then calls `train_items.extend(silver_docs.iter().map(|d| d.enc.clone()))` inside a `for _ in 0..silver_repeat` loop.
- `trainer/src/train.rs:353-387`: `fit` accepts `items: &[Encoded]`, sets `total_steps` from `items.len()`, creates `order = (0..items.len()).collect()`, shuffles it with `ChaCha8Rng::seed_from_u64(cfg.seed ^ epoch as u64)`, and clones selected entries into each batch.
- `trainer/src/dataset.rs:19-39`: `Encoded` owns several vectors, including `Vec<Vec<u32>>` n-gram IDs. `ParserBatcher` currently takes owned `Vec<Encoded>`; keep that interface in this plan.
- `internal/plans/HANDOFF.md:20-26`: person and address accuracy targets gate release. Training data weights and class weights must retain their meaning. `CLAUDE.md` asks for small modules, useful comments, `anyhow` in the trainer, and no commit unless asked.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer repeat` | All matching tests pass |
| Trainer suite | `cargo test -p trainer` | All tests pass |
| Format | `cargo fmt --all --check` | Exit 0 |
| Lint | `cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Diff | `git diff --check` | Exit 0 |

## Scope

**In scope:** `trainer/src/train.rs` and its inline tests, `plans/009-avoid-silver-repeat-copies.md`, `plans/README.md` status.

**Out of scope:** `trainer/src/dataset.rs` batching interface; data files and configs; network architecture; class weights; optimizer, learning-rate schedule, seed, checkpoint selection; the current V18 run; shipped `models/` and site.

## Steps

### 1. Add a tested draw-index helper

Add a private helper in `trainer/src/train.rs` that takes the synthetic count, silver count, and repeat count and returns the draw-index sequence. Its initial order must exactly match the old materialized sequence: all synthetic indices once, then all silver indices repeated in full, in file order, `silver_repeat` times. Use checked arithmetic for lengths or return an `anyhow` error on overflow. A zero repeat must produce no silver draws, matching current behavior. Add focused tests for repeat 0, 1, and 3 with two synthetic and two silver identities, including the sequence after epoch-seeded shuffling. The expected sequence should be built independently from a small old-style expanded identity vector.

**Verify:** `cargo test -p trainer repeat` passes the new helper tests.

### 2. Reuse encoded silver entries and preserve epoch order

In `train_detector`, store each silver `Encoded` only once after the synthetic entries. Pass the unique entries and Step 1's draws to `fit`. Keep the parser path's draw sequence as `0..items.len()`. Change `fit` to shuffle the draw index vector with the same epoch seed and `ChaCha8Rng`, not a fresh range over unique entries. Set `total_steps` from the draw count. Continue cloning only selected entries into `ParserBatcher` for each batch. Keep the `fit` API private and do not change the optimizer or validation path.

**Verify:** `cargo test -p trainer repeat` passes; `rg -n 'for _ in 0..detector_cfg.silver_repeat|silver_docs.iter\(\).map\(\|d\| d.enc.clone\(\)\)' trainer/src/train.rs` finds no repeat-clone loop.

### 3. Check semantic regression tests

Extend Step 1's tests to assert draw counts and `div_ceil(batch_size)` steps match the old expanded-vector implementation. Assert the per-epoch shuffled **example identity** sequence for at least two epochs. A test that checks only counts is insufficient.

**Verify:** `cargo test -p trainer repeat` passes and the new test names appear.

### 4. Review and finish

Run the suite, formatting, lint, and diff commands above. Inspect `git diff` to confirm no training hyperparameter, data path, or release artifact changed. Record a short `plans/009-review.md` with the checks and limitations, then mark the README row DONE.

**Verify:** all commands in the table exit 0; `git status --short` shows only the scoped source and plan files from this phase in addition to preexisting work.

## Done criteria

- [ ] Each unique silver encoded entry is stored once, regardless of repeat count.
- [ ] The pre-shuffle draw sequence and seeded epoch item sequence match the old implementation in tests.
- [ ] `total_steps`, batch count, and class/sample weighting remain equivalent.
- [ ] Trainer suite, format, lint, and diff checks pass.
- [ ] No model, data, config, site, or current V18 run artifact changes.

## STOP conditions

- The live loop differs from the excerpts above, or the prior implementation cannot be reconstructed in a test.
- The draw index approach requires changes to `ParserBatcher`, checkpoint format, or data files.
- Seeded example order, effective silver weight, or total step count differs.

## Maintenance note

This removes repeat copies but still holds each unique encoded example and clones each selected batch. Streaming or a borrowed batcher is a separate, larger experiment. Future source-balanced sampling must preserve the explicit draw semantics and record any changed weighting.
