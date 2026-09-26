# Plan 002: Measure the contact cards users actually receive

> **Executor instructions:** Follow each step and its verification. Do not train a
> model or replace the shipped bundle. Keep real documents, labels, and examples
> of errors in ignored private paths. Stop on the conditions below; update the
> status in `plans/README.md` only after all done criteria pass.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- trainer/src/group_eval.rs trainer/src/fixtures.rs trainer/src/main.rs trainer/src/eval.rs plans/README.md`.
> Inspect the live code before editing if any of these paths changed.

## Status

- **Priority:** P1
- **Effort:** L, mostly real-document annotation
- **Risk:** MED for label consistency, LOW for metric code
- **Depends on:** Plan 001's metric definitions; annotation can start in parallel
- **Category:** quality, tests
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

V17 is judged on entity spans and a separate address parser test. Users receive
`Tessera::extract_contacts`, which groups people, organizations, addresses, emails,
and phones into cards. The only present end-to-end grouping report has 18 synthetic
fixtures and 44.4% exact contacts for the shipped bundle
(`internal/reports/m5-grouping.md:5-24`). The release plan calls for real external
contact accuracy (`internal/plans/08-milestone-7.md:1762,1852-1853`). Also, the
current “exact contacts” percentage is recall-like and can remain 100% when extra
false cards appear. This plan measures both sides on real documents.

## Current state

- `trainer/src/group_eval.rs:129-169` scores assignments only over gold entities.
  `exact_contacts` counts matching gold cards, while `false_contacts` counts only
  unexpected anchors. `:202-220` prints `exact_contacts / gold_contacts` as a
  single exact-contact percentage.
- `trainer/src/group_eval.rs:172-190` loads only `GrouperFixture` JSON files. Its
  end-to-end branch calls `extract_contacts(&text, &Query::default())`.
- `trainer/src/fixtures.rs:62-89` defines a case with `name`, `input`, `entities`,
  `contacts`, and `unassigned`. `fixtures/grouper/blocks.json` is the schema
  example: contact members reference entity indices. Some JSON cases have a
  `country` property that the current Rust struct ignores.
- Existing real gold is private at `data/interim/review/gold*.jsonl`; it holds
  entity spans and text, but no contact ownership. These documents have already
  been inspected during model development, so treat them as development data.
- `internal/bench/review/GUIDELINES.md` defines entity kinds and their boundaries.
  Reuse those spans exactly. Keep org as a learned experimental kind; the main
  product analysis is people and addresses, with cards scored as whole objects.
- Match `CLAUDE.md`: small Rust helpers, `anyhow` for trainer errors, no gratuitous
  comments, no commits unless asked, and never push.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer group_eval` | Exit 0 |
| Format | `cargo fmt --all --check` | Exit 0 |
| Trainer binary | `cargo build --release -p trainer` | Exit 0; fixture and real-data runs use new metrics |
| Existing fixture baseline | `./target/release/trainer eval --grouper fixtures/grouper --grouper-hard fixtures/grouper-hard --bundle runs/detector-wide-v17/bundle/tessera-v1.safetensors --report internal/reports/m7/v17-group-fixtures.md` | Exit 0; report identifies v17 |
| Private-data boundary | `git status --short` | No `data/interim/` or `internal/reports/` files staged or tracked |

## Scope

**In scope:** `trainer/src/group_eval.rs`, `trainer/src/fixtures.rs`,
`trainer/src/main.rs` and `trainer/src/eval.rs` only if a CLI option is needed;
private annotations in `data/interim/review/contacts-dev/` and a separate
`data/interim/review/contacts-sealed/`; private aggregate reports in
`internal/reports/m7/`; the status row in `plans/README.md`.

**Out of scope:** `tessera/src/group.rs` behavior, model weights, `models/`, the
site, training data, and any training run. No model-dependent change to labels.

## Git workflow

Do not commit or push unless asked. If later asked, use a lowercase one-line
message without a trailer. Keep private text and labels outside git.

## Steps

### 1. Make contact scoring symmetric

In `group_eval.rs`, match predicted and gold cards one-to-one by the exact set of
`(kind,start,end)` members. Count exact-card true positives, unmatched predicted
cards, and unmatched gold cards; report precision, recall, and F1 with numerator
and denominator. Keep the current recall-like rate under an explicit `exact recall`
name for historical comparison. Count an extra field attached to a valid card as
a false assigned field; count wrong ownership of a gold field separately. Include
per-kind extra field counts and a false-card rate per document. Add `country` and
`source` to `GrouperCase` with optional/default values so old fixtures still load, then
aggregate per country and source when metadata exists.

**Verify:** `cargo test -p trainer group_eval` passes with new cases for (a) one
correct card plus one extra card, (b) one correct anchor with an extra phone, (c)
one missing card, and (d) duplicate predicted cards. In (a), exact-card recall is
100% but precision is below 100%; in (b), false assigned phone count is one.

### 2. Annotate a real development set without changing gold spans

Select existing reviewed real documents containing contact anchors, stratified
by country and publisher. Add contact ownership, including an explicit unassigned
list for every entity. Use the `fixtures/grouper/blocks.json` entity-index schema
in ignored `contacts-dev/` files and a manifest with source/publisher, country,
document hash, and reviewer status. Do not copy an entity or change its byte
offsets; every entity must map to exactly one card or `unassigned`. Independently
review every ambiguous assignment and at least 10% of unambiguous cases. Record
the number of documents and gold cards per country and publisher. Target at least
50 anchor-bearing documents per current country; if the existing gold cannot
supply that, record the deficit and collect additional eligible official/open
documents for **evaluation only**. Do not fill the gap with synthetic cards.

**Verify:** An annotation validator must exit 0 after checking document hashes,
UTF-8 byte offsets, every entity index exactly once, contact anchors, and no
duplicate document IDs or text hashes. `git status --short` does not show the
private corpus as a tracked change. The manifest reports counts for US, GB, DE,
GE, and JP or explicitly reports a deficit.

### 3. Reserve a fresh publisher-level final set

Choose official/open-licensed publishers not used by silver collection or the
existing reviewed gold. Record their license/source terms and content hashes.
Annotate them with the same validator into `contacts-sealed/`, with at least one
publisher and 20 anchor-bearing documents per current country as an initial
sealed minimum. Keep the seal manifest separate from development metrics. Do not
run candidate models against the sealed set during error analysis; its first
aggregate score is for the frozen release candidate and fixed metric code.

**Verify:** A manifest-only check reports zero document hashes shared with
silver, `gold-all`, or `contacts-dev`, and zero publisher IDs shared with silver.
The sealed file paths do not appear in the development report. If suitable
licensed publishers cannot be found for a country, STOP and record that country
as lacking a sealed final test; do not substitute the old gold set.

### 4. Compare current bundles on contact output

Run `extract_contacts` on the development set for v16 and v17 using the same
default no-hint query. Report exact-card precision/recall/F1, false-card count,
false assigned fields, wrong ownership, missed entities, and high-confidence
exact-card rate by country and publisher. Also run the existing fixtures to
retain historical context. Mark any small sample clearly; never average five
countries into a claim that each passed. Do not run the sealed set yet.

**Verify:** Both aggregate reports name bundle SHA-256, corpus manifest hash,
query options, per-country denominators, and metric version. No raw private
document text appears in a tracked file. `cargo fmt --all --check` exits 0.

## Test plan

Follow the tests in `trainer/src/group_eval.rs:277+`. Add the four synthetic
scoring cases in Step 1 and a malformed annotation case for duplicate entity
ownership. A fixture with zero cards must produce finite aggregate counts and
show false cards if the model invents any.

## Done criteria

- [x] Exact-card precision, recall, and F1 are reported with one-to-one matching.
- [x] Spurious cards and fields are separately counted and tested.
- [ ] Private development contact annotations validate and have country/source counts.
- [ ] A fresh publisher-level sealed set exists, or each missing country is an
      explicit blocker; it has not been used for model selection.
- [ ] V16 and v17 are compared end to end on the same development contact set.
- [x] No model training or shipped-model/site update occurred.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- Existing gold span offsets do not match source text; do not silently fix them.
- Contact ownership cannot be made unambiguous under written annotation rules.
- Publisher/source provenance is missing or license terms do not allow reuse.
- A step requires looking at sealed candidate predictions to choose labels.
- The evaluator change would alter the public library grouping behavior.

## Maintenance notes

Preserve metric version and corpus hashes when adding countries. The current
fixture `exact contacts` figure is recall-like and must not become a product
“accuracy” headline. A future release claim needs both precision and recall on
new publishers and a separately recorded high-confidence acceptance rate.
