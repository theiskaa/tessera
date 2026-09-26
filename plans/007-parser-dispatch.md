# Phase 007: Index parser country dispatch

The libpostal preparation scan reads 320,115,455 lines and previously checked every configured country until it found a match. With 100 countries, the list scan could make up to 32.01 billion string comparisons. This follow-up implements the output-preserving index identified in Phase 006; it does not prepare data, train a model, or alter the runtime library.

`trainer/src/data.rs` now builds a 26×26 direct table once for two-letter ASCII country codes. Lookup uppercases two bytes without allocating. Unsupported/nonstandard strings retain the original case-insensitive list search. Duplicate case variants preserve the first matching configured country.

Review and verification: [007-review.md](007-review.md). This is a throughput preparation change, not an accuracy fix; full-source wall-time benefit has not yet been measured.
