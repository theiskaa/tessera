# Phase 002 checkpoint review — 2026-09-26

Phase 002 is **not complete**. The metric and annotation gates are implemented;
the real development annotations and fresh sealed publisher set remain to be
reviewed. Do not use the fixture result as a product accuracy claim.

## Implemented and checked

- Contact cards now match gold cards one-to-one by the exact set of
  `(kind, start, end)` members. The report shows precision, recall, F1,
  unmatched predicted and gold cards, false-card documents, false assigned
  fields by kind, wrong ownership, and high-confidence exact-card counts.
  The old anchor-based rate remains explicitly labelled `Legacy exact recall`.
- Fixture cases accept optional country and source metadata. The evaluator
  reports those slices when present and includes the bundle hash, manifest
  hash when supplied, metric version, and default no-hint query.
- `gold_case` checks UTF-8 offsets, roles, duplicate spans and anchors, and
  exactly one owner or unassigned position for every gold entity.
- Eight focused Rust tests pass, including extra/duplicate/missing cards,
  an extra phone, wrong ownership, zero gold cards, and malformed ownership.
  `cargo fmt --all --check`, `cargo build --release -p trainer`, Clippy with
  warnings denied, Python compile, and `git diff --check` pass. The existing
  18 fixtures evaluate successfully.
- An ignored queue at `data/interim/review/contacts-dev/candidates.jsonl`
  contains 250 distinct gold documents, 50 in each current country. Every
  document's text matches its stored raw source record, with URL and license
  field present. URL hosts are sampling keys; publisher organizations require
  verification. The private annotation validator checks exact gold spans,
  document hashes, ownership, provenance, independent review, and coverage.

## Current fixture result and limit

On the 18 synthetic grouping fixtures with the v17 bundle, end-to-end exact
cards are 22 matches from 24 predicted and 27 gold: 91.7% precision, 81.5%
recall, and 86.3% F1. Two documents have an extra card. These fixtures are
small and are not a substitute for real contact output.

## Still required

- Independently assign and review ownership for the 250 development
  candidates. Resolve every ambiguous case and independently review at least
  10% of unambiguous cases; only then write scored fixtures and a passing
  manifest. The candidate queue itself is not labelled data.
- Collect and label at least 20 documents per country from fresh publishers
  not present in silver or prior gold, with source terms checked; keep that
  final set sealed until candidate and metric code are frozen.
- Run v16 and v17 on the same validated real development set and report
  country and publisher denominators. Until then, Phase 005 cannot use a
  real contact-card checkpoint objective.

No training, shipped-model update, or site change occurred in this phase.

## Follow-up queue and validator review — 2026-09-26

The initial host-round-robin queue included documents with up to 239 gold entities; several countries' 50 cases would have required thousands of ownership decisions, and GB had only 25 cases with a contact detail. Before any ownership annotation began, the queue was regenerated with a 20-entity cap and explicit detail/negative sampling. It remains 250 distinct documents, 50 per country, with a maximum of 20 entities per document. Detail-bearing counts are US 50, GB 42, DE 49, GE 25, JP 50. Host counts are US 2, GB 28, DE 15, GE 19, JP 23. US publisher diversity is therefore a material limit, even after the workload fix.

The new queue SHA-256 is `b370d3743320e9a0ce4cf286be4d0bdb1e67e87ca0156ae7dc596554501f915d` (the prior unannotated queue was `c7730c51416542c2ad8661300146cc6fcf54af7b194b9b1a66de4629ef2b4271`). An independent check verified source text, document hash, country, URL, license basis, distinctness, entity cap, and detail flags against raw records and gold. The validator now accepts `--candidates` to enforce exact frozen membership and provenance and requires a publisher-verification note. Three focused tests and Python compilation pass. This is still an annotation queue, not real contact gold; independent ownership and publisher review remain pending.
