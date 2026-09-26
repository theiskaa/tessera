# Phase 007 review — parser country dispatch

Completed 2026-09-26 after Phase 006. No `trainer prepare`, training, model export, or site change ran as part of this phase. The separate V18 diagnostic training process was already running from a binary built before this source change.

## Correctness review

- The old lookup used `eq_ignore_ascii_case` over configured strings. The new table uses the same ASCII case folding for two-letter codes and stores the first configured duplicate. Inputs outside `A`–`Z` two-letter form take the old search path.
- A focused test enumerates all 676 two-letter ASCII codes in mixed case and compares each result to the original list search, plus duplicate case variants and nonstandard strings. It passed.
- Full trainer suite: 164 passed. `cargo fmt --all --check`, Clippy with warnings denied, and `git diff --check` passed.

## Limit

The direct lookup removes dependence on configured-country count for normal ISO codes; it does not reduce the 320-million-line source read or prove an end-to-end speedup. A future `trainer prepare` run should record wall time, accepted row counts, and shard hashes against the same input snapshot. No claim is made for a 100-country preparation time.
