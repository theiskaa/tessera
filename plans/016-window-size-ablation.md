# Phase 016: Detector window-size ablation

Status: implementation retained after the quiet browser ablation and final-output review. No shipped model or site changes were made in this phase.

## Hypothesis

The original runtime used `WINDOW_TOKENS = 1024` and `OVERLAP_TOKENS = 256 + 2×64 = 384` retained tokens (`tessera/src/chunk.rs`). The network's six 192-wide convolution blocks dominate browser time. On a long document, each overlapping token repeats the expensive forward pass. Increasing only the maximum window to 2048 may reduce duplicate inference without relaxing the existing overlap, context margin, or 256-token entity coverage guarantee.

## Frozen comparison

1. Build the current source's baseline wasm package, copy it to an ignored isolated directory, record package hashes, then change only `WINDOW_TOKENS` to 2048. Build the variant into a second ignored directory. Restore the source constant immediately after packaging. The tracked final diff must contain no window-size edit until measured and reviewed.
2. After V19c training is idle, run paired Chrome SIMD-worker benchmarks with the **same V17 bundle and 10,000-character fixture**, alternating package order, at least three rounds of 20 calls. Record browser, package/bundle hashes, median/p95, memory, entity count, and full first-result hashes. A faster result under training load is not accepted as a quiet result.
3. If output differs, evaluate original and variant runtime on the same 877-document development gold before proposing a change. Check all 15 primary country metrics and the existing entity coverage tests. If output is identical on the fixture, still run native/wasm golden vectors and targeted long-document tests before keeping it.

The 150 ms target still fails by a wide margin. The observed ~83 ms median improvement was repeatable, full output was identical on the 10,000-character fixture and 100 selected long real documents, and the coverage test passes. Browser additional memory rose by ~2.36 MB and remained within the current 60 MB budget; the browser API does not expose instantaneous peak. This is a runtime ablation, not an accuracy model retrain or a release selection.
