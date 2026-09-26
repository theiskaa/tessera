# Phase 008 review — indexed term flags

Completed 2026-09-26. The V18 training binary was built before this source change; its run is unaffected. The shipped bundle and site were not changed.

## Correctness

- `term_flags` now initializes one `OnceLock<HashMap<&'static str, u32>>`. Duplicate terms combine bits with OR, the same operation as the previous eight-array scan. Each phrase still receives exactly one lookup after the old normalization, and matched bits cover the same token range.
- A test runs the original linear scan against the new index on the 10k profile fixture, every dictionary entry, and multilingual punctuation and invisible-character cases. It passed. Native parser and detector golden vectors passed; the Chrome SIMD wasm golden vectors passed. Formatting and Clippy with warnings denied passed.
- The index owns only references to static dictionary strings, is initialized once with Rust's thread-safe `OnceLock`, and does not depend on hash iteration order to form flags.

## Measurement and limit

- On Chrome 153 and the same V17 bundle and 9,975-character fixture, the feature-stage median was 4.67 ms before indexing and 2.84 ms after (30 samples after 10 warmups). Reports: ignored `internal/reports/m7/v17-stage-profile-pre-term-index.json` and `v17-stage-profile.json`.
- The whole-call medians (475.24 and 415.31 ms) are not comparable speed evidence: V18 training and other compilation/bench work ran concurrently. The detector stage dominates either profile (461.50 and 404.42 ms), so indexing cannot resolve the 150 ms release target.
- The timed full native suite had an unrelated two-second assertion fail under concurrent training before this change. A full `cargo test -p tessera` rerun after indexing passed while training remained active. The timing assertion is still load-sensitive; no end-to-end 100-country dataset or memory measurement exists yet.
