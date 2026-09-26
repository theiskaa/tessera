# Phase 008 — bound dictionary lookup as countries grow

The country scale review found that each candidate phrase checked eight term arrays linearly. Each added country can enlarge those arrays. Index the static terms once per process, preserving the bitwise union for entries shared by multiple dictionaries. Keep the token phrase construction, six-token cap, and coverage flags unchanged.

Acceptance: compare every token's term flags to the original linear scan on the 10k fixture, all dictionary entries, and multilingual punctuation/invisible-character cases. Run native and browser golden vectors, formatting, and Clippy. Measure the feature stage before and after, but make no full-call speed claim while training loads the machine. Review the diff for duplicate terms, ownership, and initialization safety.
