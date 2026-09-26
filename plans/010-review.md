# Phase 010 review — detector input contract

Completed 2026-09-26 while the separate V18 release binary kept training. The trainer source now validates each declared label before any model-kind filtering or paragraph cutting. The shipped bundle, site, input files, and live run were not changed.

## Correctness review

- Every declared kind must be known. Every byte range must be nonempty, in bounds, and on UTF-8 boundaries. Email and phone remain recognized rule-layer kinds excluded from BIO targets. Model-target spans cannot overlap or duplicate. Valid but unreachable silver spans still follow the existing unreachable accounting.
- Both synthetic Parquet and silver JSONL loaders add file and row context to failures. Three new tests cover invalid kinds/ranges/UTF-8/overlap, known rule kinds, and contextual errors from both file formats.
- A temporary local Rust test loaded the actual frozen V18 inputs through the revised loaders and passed: synthetic train 153,998, valid 9,042; silver 12,120 documents, 14,045 pieces, 147,536 reachable spans, 135 unreachable. These exactly match the active run's initial log. The temporary private-data test was removed from tracked source after the check. An independent streaming scan also found zero unknown kinds, invalid ranges/UTF-8 offsets, or overlapping model labels across the 153,998 train, 9,042 valid, and 12,120 silver documents.
- Final `cargo test -p trainer --quiet`: 169 passed. `cargo fmt --all --check`, Clippy with warnings denied, and `git diff --check` passed.

## Limit

The boundary check confirms structure, not whether each silver label names the right entity. Plan 003's human label-quality review remains pending. The running V18 binary was built before this change; future training and quantization binaries need a rebuild to include it.
