# Plan 003: Audit labels and reject incomplete training data

> **Executor instructions:** This plan changes data validation and audits existing
> inputs; it does not train weights. Keep real text and labels private. Verify each
> step, then update `plans/README.md`. Stop instead of inventing a label correction.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- bench/silver/agent_label.py trainer/src/config.rs trainer/src/data.rs trainer/src/check.rs trainer/src/train.rs configs/parser-small.toml plans/README.md`.
> Reconcile changed code with the excerpts below before editing.

## Status

- **Priority:** P1
- **Effort:** M for code, additional human review for silver labels
- **Risk:** MED; stricter gates may expose existing data defects
- **Depends on:** none
- **Category:** correctness, tests, quality
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

V17 repeated each silver piece ten times. Silver export currently omits documents
without labels and writes despite missing or risky strings, so errors can be
amplified. The parser's preparation warns when a country misses its quota but
still writes passing shards; malformed Parquet span arrays can silently truncate
labels. We need measured label quality and fail-closed input checks before any
new training run. A small dataset with many repeated examples is not equivalent
to many distinct reviewed target examples.

## Current state

- `bench/silver/agent_label.py:224-248` maps listed strings to every free occurrence;
  it returns `missing` and `risky` cases. `:326-340` ignores those results and
  writes `train.jsonl` even with problems from `read_pass`.
- `trainer/src/data.rs:1664-1684` logs parser quota shortfalls and continues.
  `:1705-1713` requires only structural `check::verify`; `trainer/src/check.rs:13-43`
  has no country minimums. `trainer/src/train.rs:136-159` checks manifest country
  names and `checks.passed`, not coverage.
- `trainer/src/data.rs:1922-1943` combines `span_label`, `span_start`, and
  `span_end` arrays with `zip`; a shorter array silently drops the tail.
- `trainer/src/check.rs:108-125` compares normalized component sets from train
  against valid/test, but not valid against test.
- `configs/parser-small.toml` requests 30,000 train and 3,000 valid/test rows per
  country. Current `data/manifests/parser-sample.json` reports JP at 27,136 train,
  2,693 valid, 2,624 test; all other current countries meet full quotas. JP needs
  an explicit, reviewed exception rather than a silent global relaxation.
- `configs/detector-wide.toml` points at silver rounds r1..r25 and repeats each
  encoded piece ten times. Source metadata is in silver JSONL but is discarded
  before sampling (`trainer/src/detector.rs:190-195`). The audit report should
  retain source/country/kind counts; changing the sampler is a later experiment.
- `internal/plans/HANDOFF.md:27-41` requires learned detection, official/open
  training sources, and private labels outside git. `CLAUDE.md` requires small
  Rust modules, `anyhow` for trainer errors, and no commits unless asked.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Python tests | `python3 -m unittest discover -s bench/silver -p 'test_*.py'` | Exit 0 |
| Rust tests | `cargo test -p trainer` | Exit 0 |
| Format and lint | `cargo fmt --all --check && cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Private-data boundary | `git status --short` | No private silver or review text tracked |

## Scope

**In scope:** `bench/silver/agent_label.py`, a new
`bench/silver/test_agent_label.py`, `trainer/src/config.rs`, `trainer/src/data.rs`,
`trainer/src/check.rs`, `trainer/src/train.rs`, `configs/parser-small.toml`,
private audit outputs in `internal/reports/m7/` and reviewed/quarantined private
labels under `data/interim/silver/`, and the status row in `plans/README.md`.

**Out of scope:** `tessera/src` detection rules, model architecture, model
weights, `models/`, site, corpus generation, sampler rebalancing, and all
training commands. Do not rewrite gold evaluation labels as part of silver QA.

## Git workflow

Do not commit or push unless asked. Keep audit rows containing text, URLs, or
labels in ignored private locations. A later commit, if requested, is a single
lowercase line without a trailer.

## Steps

### 1. Make silver export fail before writing on unresolved labels

Collect `read_pass` problems plus each document's `missing` and `risky` results.
Reject missing document labels, unknown kinds, missing strings, ambiguous short
CJK strings without `within`, and overlapping/conflicting spans with a nonzero
exit before replacing `train.jsonl`. Permit an intentionally excluded document
only through an explicit private exclusion record with ID and reason; show the
excluded count in the export summary. Write successful output to a temporary
file and rename only after all checks pass. Preserve the existing every-occurrence
semantics for reviewed labels; do not add runtime word rules.

**Verify:** New Python tests cover a clean export, missing document, missing
listed string, risky repeated short CJK label, bad kind, and preservation of an
existing output file after failure. `python3 -m unittest discover -s bench/silver
-p 'test_*.py'` exits 0.

### 2. Report silver composition and review real label quality

Add a private read-only audit script or mode that reports documents, distinct
source pages, encoded pieces, person/org/address spans, unreachable spans,
duplicate text hashes, and exact gold entity overlap by source and country.
Include the count of target address labels in GE, GB, and DE. Review every GE
silver document with an address label plus at least 200 other documents sampled
across country, source, and kind. Record missing/incorrect/ambiguous labels as
counts and document IDs in `internal/reports/m7/silver-audit.md`; keep actual
texts private. Correct labels only after review, or quarantine them with a
reason. Never copy gold labels into training.

**Verify:** The audit exits 0 on the current v17 inputs; its totals reconcile
with `runs/detector-wide-v17/summary.json` (`12,373` silver documents,
`14,318` pieces, `150,477` reachable spans) or explain exactly why a count
uses a different unit. The report has a source×country×kind table and a
reviewed-error denominator. `git status --short` contains no private corpus.

### 3. Reject mismatched parser span arrays

In `data.rs::read_shard`, compare list lengths for labels, starts, and ends
before zipping. Reject a mismatch with shard path and row index. Continue
using `check_spans` after constructing equal-length spans. Add a small malformed
Parquet fixture in a test to prove that the loader fails instead of dropping a
gold label.

**Verify:** `cargo test -p trainer` passes; the malformed fixture test reports
the exact row number and array lengths.

### 4. Gate parser country coverage and all split pairs

Add config-backed per-country coverage exceptions, empty by default. Full
requested quotas are required otherwise. In `configs/parser-small.toml`, record
JP's current minimums (27,136 / 2,693 / 2,624) with a source-shortage reason;
do not relax the other countries. Validate coverage both before `prepare`
writes shards and when `train_parser` reads a manifest. Add a manifest check
version so old `checks.passed=true` cannot silently stand in for the new gate.
Extend `check::verify` to compare equivalent labelled component sets across
train/valid, train/test, and valid/test. Keep the existing group-key check.

**Verify:** Unit tests show a zero-row new country fails, all five current
counts pass only with the explicit JP exception, a count below JP's stated
minimum fails, and matching components across valid/test fail. `cargo test
-p trainer` exits 0. Before any later parser training, run `trainer prepare`
to create a new-version manifest; that preparation is not a model training run.

## Test plan

Use standard-library Python `unittest` in `bench/silver/test_agent_label.py` and
the existing Rust test modules in `data.rs`/`check.rs`. Tests must exercise
failure preservation of the output file and bad input rows, not mirror the
implementation's own helper calls. Do not use the 7.5 GB source corpus as a
unit-test fixture.

## Done criteria

- [x] Silver export rejects unresolved labels without replacing the prior export.
- [ ] The v17 silver mix and human-reviewed error rate are reported by source,
      country, and kind, with GE addresses visible.
- [x] Parser shard loading rejects unequal span arrays.
- [x] A new country with no data fails coverage checks; JP's exception is explicit.
- [x] All three split pairs are checked for equivalent labelled components.
- [x] Python tests, Rust tests, format, and lint commands above exit 0.
- [x] No training ran and `models/` and the site are unchanged.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- Existing silver labels show nontrivial unresolved problems; do not silently
  discard them or lower a gate. Report IDs/counts for human relabelling.
- A proposed exception would allow zero valid or test rows for a country.
- The old parser sample cannot pass the new split check; report the offending
  private row IDs and decide whether to resample, without training.
- The audit would expose real text or labels in a tracked file.

## Maintenance notes

For each future silver round, run the export gate and source/country audit
before it enters a training config. A passing structural manifest is not a
proof of adequate per-country data or of clean labels. Keep evaluation sources
out of training and keep the exclusion list reviewable.
