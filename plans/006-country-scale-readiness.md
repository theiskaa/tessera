# Plan 006: Define the evidence and architecture needed for 100 countries

> **Executor instructions:** This is a design and capacity plan, not a country
> implementation. Read the current code and completed Plans 001 and 003. Write
> the two specified documents, run the checks, and update `plans/README.md`.
> Do not add a country, change tokenization, or train a model. Stop on the
> conditions below rather than claiming support from a country code alone.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- trainer/src/generate.rs trainer/src/names.rs trainer/src/data.rs tessera/src/token.rs tessera/src/features.rs internal/plans/08-milestone-7.md plans/README.md`.
> Re-read changed paths and completed Plans 001/003 before drawing conclusions.

## Status

- **Priority:** P2
- **Effort:** M
- **Risk:** LOW for design; later tokenizer/model migrations are high risk
- **Depends on:** measured runtime from Plan 001 and data audit from Plan 003
- **Category:** direction, performance, architecture
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

The shared model does not run once per country; inference is roughly one pass
per document. But current data generation, preprocessing, feature vocabulary,
script handling, and evidence of country quality do not scale automatically.
At 100 countries, the unchanged 170,000-document synthetic train budget averages
only 1,700 documents per country; keeping today's roughly 34,000 allocation
would need about 3.4 million documents. We need a measurable onboarding
contract and capacity model before implementing a registry or promising broad
support. A bigger model or country packs are not justified by current evidence.

## Current state

- `trainer/src/generate.rs:1705-1716` hardcodes six generator country codes;
  `:1793-1805` samples country uniformly inside fixed split totals.
- `trainer/src/names.rs:43-87` has a separate country list. `:429-435`
  requests every configured birth year per country and language, and
  `:886-910` sleeps one second after each cold successful query. At 76 years,
  100 one-language countries mean at least 7,600 requests and 2h 6m deliberate
  delay before network time or retries.
- `trainer/src/data.rs:1558-1589` searches configured parser countries linearly
  per input line; `data/manifests/parser-sample.json` reports about 320 million
  source lines. `tessera/src/features.rs:619-620,690-741` scans eight static
  dictionaries linearly for up to six candidate phrase lengths per token.
- `tessera/src/token.rs:106-169,474-482` recognizes 12 script groups. Other
  letters become merged `Other` tokens; `tessera/src/features.rs:111-139`
  caps n-grams at 64 per token. Armenian and Bengali are examples of scripts
  with no dedicated class today; quality impact requires measurement.
- `internal/plans/08-milestone-7.md:1334-1358` already proposes a country
  registry, coverage test, and onboarding guide. Its later support rule at
  `:1380-1384` still requires organization F1, while the newer user decision
  in `internal/plans/HANDOFF.md:23-26` makes organizations experimental and
  removes them from current exit targets. Use the newer decision; flag the
  older rule for explicit reconciliation before implementing support status.
- `CLAUDE.md` requires small Rust modules and no commits unless asked. No
  hand-written detection rules may be added for a new country; training data
  and model features are the intended path (`HANDOFF.md:27-30`).

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Inventory | `rg -n 'US|GB|DE|GE|JP|CA|NL' trainer/src tessera/src site/src configs/` | Locations are recorded by category, not copied wholesale |
| Script check | `rg -n 'enum Script|fn script_of|MAX_NGRAMS_PER_TOKEN' tessera/src/token.rs tessera/src/features.rs` | Source locations match the report |
| Document check | `test -f docs/adding-a-country.md && test -f internal/reports/m7/country-scale-readiness.md` | Exit 0 |
| Working-tree check | `git status --short` | No new changes outside planned docs/status compared with the initial snapshot; the two detector configs were already modified for v17 |

## Scope

**In scope:** create `docs/adding-a-country.md` as a clearly marked draft,
create a private aggregate `internal/reports/m7/country-scale-readiness.md`,
and update the status row in `plans/README.md`.

**Out of scope:** country registry implementation, new country data collection,
new phone tables, tokenizer changes, model architecture, runtime word rules,
`models/`, site, and any training run. This plan makes the implementation
decision concrete; it does not claim a sixth country is supported.

## Git workflow

Do not commit or push unless asked. `docs/` is tracked documentation; the
aggregate report under `internal/` is ignored. A later commit uses one
lowercase line without a trailer.

## Steps

### 1. Inventory country dependencies and script coverage

In the private readiness report, map each country-specific surface to its
source file and owner: generator pools/templates, name collection, parser
sampling, feature terms/postcodes, phone tables/region inference, fixtures,
real evaluation, demo sample, bundle manifest, and support status. Distinguish
runtime bytes from trainer-only data. Add a script-coverage table for the 12
recognized groups plus `Other`, showing what happens to an unsupported
alphabetic run and its 64-gram cap. Do not infer accuracy from script names.

**Verify:** The report has a file:line for every category above and identifies
at least Armenian and Bengali as tokenizer-coverage cases, labelled “quality
unmeasured.” `rg` commands in the table still locate the cited code.

### 2. Build a cost model for 5, 20, and 100 countries

Use Plan 001's measured v17 browser latency and Plan 003's actual silver
distribution, not the old 337 ms arithmetic estimate, as the present baseline.
Show the fixed-total and fixed-per-country synthetic data cases; parser source
scan complexity over 320 million lines; cold names request lower bound by
country and language; training memory from tenfold silver materialization;
and dictionary lookup cost with a larger vocabulary. Mark arithmetic models
as estimates and list the measurements needed to validate them. Identify
whether an optimization can preserve identical feature flags/labels (such as
indexed country lookup) or would require retraining (such as tokenizer changes).

**Verify:** The report includes numerical rows for 5, 20, and 100 countries,
with the 100-country fixed-total row showing about 1,700 synthetic train
documents per country and the preserved-allocation row about 3.4 million
total. Browser latency is cited from Plan 001's JSON, not extrapolated.

### 3. Draft a country onboarding and support contract

Write `docs/adding-a-country.md` with these sections: legal data source and
provenance; script/tokenizer readiness; distinct synthetic and real examples
by person/address kind; parser sample coverage; phone metadata and rules
fixtures; source-separated development and sealed final evaluation; browser
performance; support-matrix result; and the order of commands. Use the
current per-country targets in `HANDOFF.md:23-26` and explicitly mark org
experimental. A country missing any mandatory evidence must be labelled
`experimental`, not `supported`. State that the guide is a draft until the
registry and support-matrix implementation land. Do not choose a sixth
country without the user's product priorities.

**Verify:** The document contains all nine named sections or subsections,
and a checklist item for each. It does not tell an operator to edit
`models/`, publish the site, or start training as part of this plan.

### 4. Record the architecture decision boundary

The private report should recommend what can be done before retraining:
source-indexed parser dispatch, dictionary indexing with identical flags,
coverage gates, and a data-driven registry. It should reserve decisions on
country packs, larger embeddings, and tokenizer versioning for experiments
with new-source real evaluation. Include a decision table with expected
benefit, implementation risk, whether a retrain is required, and the metric
that would prove it useful.

**Verify:** The table has at least the seven options named above, each with a
`retrain required` value. No runtime or training source file changed.

## Test plan

This is a document/design spike, so no new code test is required. Use the
file-existence check and source-location checks above. A reviewer should
verify the cost model's arithmetic and the reconciliation with the current
organization decision before the draft is treated as an implementation spec.

## Done criteria

- [x] Country-specific dependencies are inventoried with file locations.
- [x] 5/20/100-country cost cases distinguish measured values from estimates.
- [x] Script coverage and retraining implications are explicit.
- [x] A draft onboarding checklist requires real per-country evidence and
      uses the newer organization policy.
- [x] No country was added, no model trained, and no runtime behavior changed.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- The target list of countries is needed to claim readiness for particular
  scripts or legal data sources; report that dependency rather than guessing.
- The old milestone support rule conflicts with the current user decision in
  a way that cannot be reconciled in documentation alone.
- Plan 001 has no browser measurement; leave runtime capacity as unknown and
  do not substitute an arithmetic estimate.

## Maintenance notes

Revisit the capacity model when per-country budgets, silver repeat, or
tokenizer representation change. A one-model runtime may remain viable at
100 countries, but the data and evidence burden grows even when model
inference does not multiply by country count.
