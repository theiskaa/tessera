# Phase 014 review — zero-activation kernel shortcut

`tessera/src/model/kernels.rs` skips a weight row when its input activation is exactly `0.0`, in scalar and wasm SIMD paths. Kernel tests compare bit patterns with the full accumulation path, and prior native and browser golden-vector checks passed. The paired Chrome script also checked the complete first detection result was identical for both wasm packages in every round.

After the V18 evaluation ended and before V19 training began, three quiet-machine paired 10,000-character browser rounds produced baseline medians 392.8, 394.4, 394.4 ms and optimized medians 391.6, 391.8, 391.0 ms. Median of round medians: 394.4→391.6 ms, about 2.8 ms or 0.7% faster. The run order alternated to limit drift. The raw times, browser version, package hashes, bundle hash, and output hash are in ignored `internal/reports/m7/zero-skip-browser.json`.

Decision: keep the shortcut as a small output-preserving improvement. It does not solve the 150 ms browser target. The source also contains the separately reviewed term-index change from phase 008; the compared wasm packages predate that rebuild, so this result isolates the shortcut but is not a benchmark of the latest full source tree. The earlier 475.3 ms baseline ran under a different machine load and cannot be used as the paired comparison.
