# Phase 003 checkpoint review — 2026-09-26

Phase 003 is **not complete**. Input gates and a structural audit are in place.
Existing silver labels need adjudication before a strict re-export, and the
planned human label-quality review has not happened.

## Verified changes

- Silver export now rejects missing documents, malformed label entries,
  strings not found in text, repeated short CJK labels without context,
  overlapping occurrences, and unreachable labels. An exclusion requires a
  private ID and reason. A passing export writes a temporary file and renames
  it only after validation. Seven Python tests pass, including preservation of
  an existing export after failure.
- A real round-25 strict-export check failed with 1,836 unresolved document
  issue records. Its prior `train.jsonl` SHA-256 was unchanged before and
  after: `b19e62e1e74ae621476e0c2b6d6434983e2506048a8ebf320c12f35da66ee35e`.
  No silver export was replaced.
- Parser shard loading now rejects unequal or null span arrays before `zip`.
  A malformed Parquet row test checks the path, row number, and array lengths.
- Parser coverage now requires full requested quotas except the recorded JP
  minimums of 27,136 train, 2,693 valid, and 2,624 test. Zero-row new
  countries, a count below the JP minimum, a shortfall in a non-exception
  country, and a zero exception all fail tests. The training path rejects old
  manifests without check version 2; no parser training was started.
- Labelled-component leakage is checked across train/valid, train/test, and
  valid/test. An independent scan of the existing Parquet shards found zero
  cross-split component matches and zero unequal span arrays, with country
  counts matching the existing manifest. This Python scan is corroborating
  evidence; the Rust `check::verify` gate is authoritative on new preparation.
- `cargo test -p trainer` passed 156 tests. Python `unittest` passed seven;
  `cargo fmt --all --check`, Clippy with warnings denied, and
  `git diff --check` passed.

## Audit findings

- V17 used 12,373 silver documents repeated tenfold. Raw labelled spans are
  150,613, exactly the trainer's 150,477 reachable plus 136 unreachable spans;
  the trainer encoded 14,318 pieces.
- GE has 202 address labels across 116 silver documents, versus 931 in GB
  and 991 in DE. The audit found 161 repeated document texts beyond their
  first occurrence; 133 repeated-document groups span rounds 3 and 25.
- The new strict check flags 10,457 overlapping label occurrences in 4,276
  documents, plus four missing strings and 108 declared strings with no
  accepted occurrence. These are structural conflicts requiring review; they
  are **not** a measured label-error rate. Exact gold document hash overlap is
  zero. Exact normalized entity-surface overlaps are listed by source and
  country in the private audit and need interpretation, not automatic removal.
- The ignored review queue contains every GE silver document with an address
  label plus 200 source/country/kind-stratified other documents (316 total).
  Reviewed denominator and measured error rate are still zero/pending.
- A follow-up exact-text comparison found 27 duplicated GE document groups with
  conflicting detector labels (37 organization and four person span disagreements).
  Five r25 GE documents contain village/municipality fields headed "address"
  without street or house numbers and have no address label. The existing
  annotation guide excludes a city or region alone, so these locality-only
  fields are consistent with policy and are not confirmed label errors. The original
  202 GE silver address spans are
  only 4.8% of the 4,251 silver address spans; V19c's 600 synthetic GE additions
  raised that share to 16.5%. See the private root-cause audit. The error rate
  remains unmeasured.
- A private deterministic deduplication candidate removes identical copies and
  quarantines all 54 rows in the 27 conflicting groups. It retains 12,185 unique
  texts from 12,373 input rows, with no duplicate text hashes. Georgian silver
  address spans fall from 202 counted with copies to 185 distinct retained spans.
  It is not approved for training until strict-export issues and semantic label
  quality are adjudicated; the candidate manifest records all source hashes and
  quarantined IDs.

Private aggregate detail: `internal/reports/m7/silver-audit.md`,
`silver-audit-summary.json`, and `parser-split-audit.json`. The current parser
manifest still has the old check version. Before any later parser training,
rerun preparation to produce a version-2 manifest; that is not a model run.

No model training, shipped-model change, or site change occurred. At this
checkpoint, the nested-label flags still needed classification; the follow-up
review below records that classification and the remaining cases.

## Follow-up overlap-gate review

The 10,457 flags above were classified: 10,310 are shorter same-kind surfaces wholly covered by an already accepted longer label, which the existing longest-first export rule resolves without changing output spans. The strict gate now accepts only this case. It still rejects 146 cross-kind contained overlaps and one repeated full span; 147 overlap flags remain across 75 documents. Forty wholly covered labels also cease to count as unreachable, leaving 68 unreachable labels and four missing strings. Across the 13 rounds, 125 distinct round/document pairs retain a structural issue. R25 strict export now refuses 51 issue records in 51 documents; its existing export hash is unchanged. An independent old-versus-new span replay on 12,403 sampled documents found zero changed outputs; eight Python tests pass. The current private audit was regenerated, and `internal/reports/m7/silver-overlap-gate-review.md` records the taxonomy and checks.

This is a correction to the gate's interpretation of the already documented longest-first policy, not a semantic label review. The 125 flagged documents, 27 conflicting duplicate-text groups, and 316-document quality queue remain open; no silver export is approved for new training.
