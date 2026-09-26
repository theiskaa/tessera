# Adding a country — draft onboarding contract

This is a draft until the country registry and computed support matrix exist. A country code in a config or phone table does not imply product support. Choose candidate countries from product needs and available lawful data, then complete the evidence below. Missing mandatory evidence means `experimental`.

## 1. Legal data source and provenance

- [ ] Record source URL, publisher, document type, collection date, **page- or dataset-specific** license or terms, allowed uses, and retention decision for every names, organizations, addresses, and real-document source. A government domain alone is not licence evidence.
- [ ] Keep publisher IDs and document hashes so training, development, and final sets can be separated. Remove any overlapping synthetic names or pages from evaluation before using the set.
- [ ] Audit silver labels by source, country, and kind. Independently review a label-blind, source-stratified sample, compare passes, and adjudicate disagreements. Quarantine unresolved spans and conflicting repeated documents before any training use; report the reviewed denominator and measured error types.

## 2. Script and tokenizer readiness

- [ ] Inspect `tessera/src/token.rs` for every script in target documents, including mixed-script names and addresses. Run tokenizer fixtures with UTF-8 byte offsets, punctuation, and combining marks.
- [ ] Check the 64 n-gram cap, script embedding rows, and feature flags for representative long tokens. Unsupported scripts such as Armenian or Bengali need a measured representation plan and new model/golden vectors before support is claimed.
- [ ] Record a per-script development and final breakdown. Script coverage alone is not an accuracy result.

## 3. Distinct synthetic and real examples by kind

- [ ] Add licensed country-appropriate person and address pools, local templates, greetings, closings, titles, filler, and hard negatives. Keep train/valid/test families and entity surface splits distinct.
- [ ] Collect reviewed real person and address examples from independent publishers and document types. Track **distinct detector-reachable** spans, negatives, and source denominators, rather than repeated training draws. Organization remains experimental and is reported separately.
- [ ] Include contact ownership annotations to measure complete cards, false cards, and wrongly assigned fields on real documents.

## 4. Parser sample coverage

- [ ] Inspect available tagged address source lines, country normalization, UTF-8 span validity, and train/valid/test component separation.
- [ ] Set explicit per-country train, valid, and test minima in the parser config. A scarce-country exception requires counts and a reason; no zero-country exception is allowed.
- [ ] Run parser preparation and its manifest checks before any future parser training. Record source and shard hashes. Recheck full-address exact parse on real addresses.

## 5. Phone metadata and rules fixtures

- [ ] Check the country in phone metadata, international and national formats, short service numbers, and false positives. Add fixtures for valid, invalid, and ambiguous national numbers.
- [ ] Test both `Query::default()` region inference and an explicit country hint. Record where the country cannot be inferred from visible text.
- [ ] Keep country-specific exceptions narrow and backed by official numbering evidence; do not add hand-written person or address detection rules.

## 6. Source-separated development and sealed final evaluation

- [ ] Freeze development labels, query mode, evaluator version, and corpus hashes before choosing checkpoints. Report person exact F1, address exact F1, address lenient precision/recall/F1, parser exact parse, and real contact-card precision/recall/F1 per country and publisher.
- [ ] Reserve a fresh publisher-level final set untouched by training, prompt iteration, error review, and checkpoint selection. Verify source overlap and distinct-document hashes.
- [ ] Use uncertainty estimates clustered by publisher. Do not replace a failed country slice with an aggregate average.

## 7. Browser performance

- [ ] Measure warm `detect` on the 10,000-character browser fixture using the candidate bundle and the same device/browser protocol as the current baseline. Record median and p95, bundle and package hashes, and sample count.
- [ ] Profile feature extraction, rules, parser, and detector stages if the median exceeds 150 ms. Recheck output identity for optimizations intended to preserve behavior.

## 8. Support-matrix result

- [ ] Require person exact F1 ≥90%, address exact F1 ≥80%, address lenient recall ≥95%, parser full-address exact ≥95%, and browser 10k-character median ≤150 ms on the declared real evaluation protocol. Report phone and contact-card measures separately; define their release gate before assigning supported status.
- [ ] Mark `experimental` if any mandatory evidence is missing or a threshold fails. Only the future computed support matrix may write `supported_regions` in a bundle manifest.
- [ ] Reconcile the older Milestone 7.9 organization threshold with the current decision: organizations are experimental and are not a current person/address exit gate.

## 9. Order of commands and decisions

1. Inventory sources and legal terms; record manifests and hashes.
2. Run tokenizer, phone, generator, parser, and fixture checks on representative examples.
3. Prepare and validate parser and detector data; independently review real development and sealed final labels.
4. Freeze the experiment brief, country targets, checkpoint rule, and publisher split before any training proposal.
5. After the user authorizes a concrete run, execute one controlled training change and evaluate every epoch against the fixed development set. Requantize both artifact-bound runs before export.
6. Run the sealed final evaluation once for the selected candidate, measure browser performance, and compute the support matrix. Shipping follows its own decision.

The registry, support-matrix command, and some automated coverage gates described here are planned work; this document does not claim they already exist.
