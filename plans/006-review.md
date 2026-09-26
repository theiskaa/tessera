# Phase 006 review — country scale readiness

Completed 2026-09-26. The result is a design and onboarding contract, not support for another country.

## Deliverables

- `docs/adding-a-country.md` is a draft nine-section checklist for lawful sources, script readiness, distinct synthetic and real examples, parser coverage, phone fixtures, source-separated evaluation, browser performance, support status, and command order. It uses current person/address targets and treats organizations as experimental.
- Ignored `internal/reports/m7/country-scale-readiness.md` maps runtime and trainer surfaces to source locations, names all twelve recognized script groups plus Other, distinguishes measured five-country latency and silver distribution from 20/100-country arithmetic, and gives a seven-option decision table with retraining consequences.

## Review findings

- Recalculated the scenario table from 170,000 synthetic train documents, 14,318 silver pieces repeated tenfold, 76 birth years, and 320,115,455 parser source lines. At 100 countries, fixed-total allocation is 1,700 synthetic train documents per country; preserving 34,000 per country requires 3.4 million. The 32.01 billion country comparisons are an upper-bound arithmetic model, not a measured preparation time.
- Confirmed `tessera/src/token.rs` classifies uncovered alphabetic ranges as Other and `tessera/src/features.rs` caps n-grams at 64 per token. Armenian and Bengali quality remains unmeasured.
- Confirmed Plan 001's measured V17 SIMD browser median/p95 are 475.3/499.7 ms on the 10k fixture; the report does not project that latency by country count.
- Confirmed Plan 003's actual silver totals and GE address scarcity. The audit's unresolved human label-quality review remains visible.
- Reconciled the old Milestone 7.9 organization threshold with the newer `HANDOFF.md` decision in documentation. A future computed support matrix must implement that decision explicitly.
- Document/arithmetic checks and `git diff --check` passed. `git status --short` showed no new runtime, model, or site changes from this phase.

No training command or sealed-set evaluation ran.
