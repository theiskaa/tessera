# Phase 011 review — refuse checkpoint reuse

Completed 2026-09-26. The live V18 process uses a release binary compiled before this source change and a fresh run directory; no model, data, config, site, or run artifact was changed by this phase.

## Correctness review

- `initialize_run` checks `best.mpk` and `.mpk` entries in `checkpoints/` before creating directories, invalidating a gate, or rewriting config. It returns a new-run-name error if either exists. It permits a directory with no checkpoint, including an incidental noncheckpoint file.
- The three focused tests cover both refusal paths with byte-for-byte preservation of existing config/gate/checkpoint/weights and the allowed no-checkpoint path with gate invalidation. All passed.
- Full trainer suite: 166 passed. `cargo fmt --all --check`, `cargo clippy -p trainer --all-targets -- -D warnings`, and `git diff --check` passed. The final diff was reviewed for scope.

## Limit

This guard prevents accidental reuse through `trainer train` as implemented today. It is not an explicit resume facility or a cryptographic provenance check for externally copied checkpoints. The user can start another run with a new name; current V18 training and evaluation remain pending.
