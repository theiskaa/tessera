# Phase 009 review — reuse encoded silver pieces

Completed 2026-09-26 while the separate V18 release binary kept training. This source change affects future training runs only. No data, config, model, current run, or site file was changed.

## Correctness review

- Detector training now stores each silver `Encoded` piece once and builds repeated draw indices. The parser still draws its examples once. `fit` shuffles draw positions with the same epoch seed, uses the draw count for its learning-rate step schedule, and clones only the selected batch entries.
- Tests compare the new and old expanded example-identity sequence before shuffling and after two consecutive seeded epoch shuffles at repeats 0, 1, and 3. They also compare batch step counts at sizes 1, 2, and 3 and reject count overflow. All passed.
- For V18's observed 153,998 synthetic entries, 14,045 silver pieces, and repeat 10, the old materialization stored 294,448 encoded entries; the new representation stores 168,043 encoded entries plus 294,448 `usize` draw indices. This is a count calculation, not a measured RSS reduction. The live V18 run still uses the old binary and memory layout.
- Final `cargo test -p trainer --quiet`: 171 passed. `cargo fmt --all --check`, Clippy with warnings denied, and `git diff --check` passed. The final diff was inspected for sampling and step-count changes.

## Limit

Each unique example still lives in memory and each selected batch still clones its entries for `ParserBatcher`. The change does not make a 100-country corpus fit in memory by itself; source-balanced sampling and streaming remain separate decisions. A future diagnostic rerun should measure RSS on the same input snapshot.
