# Plan 005: Select future checkpoints by real country targets

> **Executor instructions:** This plan evaluates already saved checkpoints and
> freezes a selection rule. It does not run training or ship a model. Use only
> the development sets established by Plans 001 and 002; leave the sealed
> final set untouched. Stop on the conditions below and update the status row
> in `plans/README.md` when the protocol and retrospective pass verification.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- trainer/src/train.rs trainer/src/quantize.rs trainer/src/export.rs trainer/src/detect_eval.rs plans/README.md`.
> Re-read the functions below if code changed; use Plan 004's gate format.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** MED; a small real development set can be overfit
- **Depends on:** Plans 001, 002, and 004
- **Category:** quality, reproducibility
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

The detector currently chooses `best` by synthetic validation macro exact F1
over person, organization, and address (`trainer/src/train.rs:267-280`). The
current exit targets are **per country** on real people and addresses;
organizations are experimental (`internal/plans/HANDOFF.md:23-26`). V17's
synthetic validation reached 95.3%, yet GB/GE/JP people and several address
slices remain far below target. A future run must not be judged by a selection
metric that rewards an experimental kind and can hide country regressions.
The existing v17 epoch checkpoints allow a retrospective test without any
new training.

## Current state

- `trainer/src/train.rs:267-280` computes synthetic `SpanScores` each epoch and
  returns `macro_exact_f1` as the selection score. `:419-426` saves every epoch
  under `checkpoints/epoch-N.mpk` and overwrites `best.mpk` when that score rises.
- `runs/detector-wide-v17/checkpoints/epoch-1.mpk` through `epoch-4.mpk`, its
  `config.toml`, and `runs/parser-small-v7/` exist. `trainer quantize --run`
  consumes `best.mpk`; `trainer export --parser-run --detector-run --out`
  packages the two networks. Plan 004 binds a passing quantization gate to
  the exact staged checkpoint/config/weights.
- `trainer eval --gold ... --bundle ...` uses the shipped runtime path and
  produces per-country precision/recall/F1 in JSON (`trainer/src/detect_eval.rs:
  604-625`). Plan 001 makes the hint mode and metric labels explicit. Plan 002
  adds real contact-card development metrics. The same parser v7 must be used
  in every retrospective candidate.
- The existing gold pages have been examined repeatedly. They are development
  data, not a sealed final test. Do not select on the new publisher-level
  final set from Plan 002. Keep real data and prediction errors ignored by git.
- Match `CLAUDE.md`: small scripts, no commit unless asked, never push. Keep
  `models/` and the live site unchanged without separate approval.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Confirm checkpoints | `ls runs/detector-wide-v17/checkpoints/epoch-{1,2,3,4}.mpk` | Four files listed |
| Trainer binary | `cargo build --release -p trainer` | Exit 0; Plan 004 gate is in this binary |
| Quantize staged run | `./target/release/trainer quantize --run runs/selection-v17/epoch-1` | Exit 0 or a recorded gate failure; no training |
| Export staged run | `./target/release/trainer export --parser-run runs/parser-small-v7 --detector-run runs/selection-v17/epoch-1 --out runs/selection-v17/epoch-1/bundle --date 2026-09-26` | Exit 0 when gate passes |
| Evaluate staged bundle | `./target/release/trainer eval --gold data/interim/review/gold-all.jsonl --bundle runs/selection-v17/epoch-1/bundle/tessera-v1.safetensors --out runs/selection-v17/epoch-1/eval --report internal/reports/m7/v17-epoch-1.md` | Exit 0; per-country report exists |
| Working-tree boundary | `git status --short` | No change to model, site, gold labels, or training configs |

## Scope

**In scope:** an isolated private script under `internal/bench/selection/`,
staged ignored runs under `runs/selection-v17/`, private aggregate reports
under `internal/reports/m7/`, and the status row in `plans/README.md`.

**Out of scope:** `trainer train`, modifications to `trainer/src/train.rs`,
private gold labels, sealed final data, `models/`, site, training configs,
and choosing a new architecture. If CLI changes prove necessary, STOP and
write a narrower follow-up plan rather than extending this one.

## Git workflow

Do not commit or push. Staged runs and error reports are ignored; do not copy
them into tracked directories. A later source-code follow-up needs a separate
review and user direction.

## Steps

### 1. Freeze a country-aware decision rule before scoring epoch checkpoints

Write `internal/reports/m7/selection-protocol.md` using Plan 001's exact
metric names and Plan 002's development contact metric. Primary slices are
person exact F1 (target 90%), address exact F1 (80%), and address lenient
recall (95%), each per current country. Parser exact parse is a separate
fixed-v7 gate, not a detector checkpoint ranking term. Organizations are
reported but excluded from selection. For each detector candidate, calculate
positive percentage-point shortfall to every primary target, the maximum
shortfall, and the sum of shortfalls. Also record any regression beyond two
percentage points from the current best on a previously passing slice.
Rank first by fewer previously passing slices lost, then by lower maximum
shortfall, then lower sum; treat differences smaller than a source-cluster
bootstrap confidence interval as a tie. In a tie, keep the earlier epoch.
Use only development sources; do not tune this rule after seeing epoch scores.

**Verify:** The protocol names all five countries and all three primary
detector metrics, includes the tie and regression rules, cites a version/hash
of the development corpus, and explicitly excludes the sealed final set.

### 2. Evaluate all existing v17 epochs in isolated run directories

Write a deterministic script that creates one fresh ignored staging directory
per epoch, copies `config.toml` and the corresponding `epoch-N.mpk` as
`best.mpk`, records source checkpoint/config hashes, then invokes quantize,
export with the unchanged parser v7, and real development evaluation. Never
modify `runs/detector-wide-v17/` or `runs/parser-small-v7/`. If an epoch fails
the quantization gate, mark it rejected; do not bypass the gate. Run the
script for all four v17 epochs. If Plan 002's development contact annotations
are ready, evaluate their cards too with the same bundle and query mode.

**Verify:** Four staging manifests list distinct detector checkpoint hashes,
the same parser hash, the same development corpus hash, and the same evaluation
mode. Every passing staged bundle passes Plan 004's artifact-bound export
gate. No `trainer train` process or command appears in the script.

### 3. Compare selection rules without touching the final test

Write `internal/reports/m7/v17-selection-retrospective.md` showing each epoch's
synthetic macro F1, every primary real country slice, shortfall calculation,
contact-card metric where available, and any precision/recall trade-off.
Bootstrap uncertainty by publisher/source cluster, not by individual entity,
so documents from one body do not masquerade as independent evidence. Name
the epoch selected by the frozen rule and whether it differs from epoch 4.
If the rule is unstable within uncertainty, say “no reliable checkpoint
difference” and retain the current candidate; do not manufacture a gain.

**Verify:** Re-running the script produces the same bundle hashes and ranking.
`git status --short` shows no model/site/gold change. The report contains no
sealed-set score and no raw private text.

### 4. State the condition for a future training proposal

Add to the protocol the required experiment brief: one change (for example,
source-balanced sampling or targeted GE addresses), expected direction for
specific weak slices, maximum acceptable regression on passing slices, fixed
development/selection metrics, and one sealed final confirmation. The next
training run is proposed to the user only after Plans 001–004 have passing
outcomes and this brief is reviewable. Do not launch it in this plan.

**Verify:** The protocol has those six fields and an explicit `training_status:
paused` line. `plans/README.md` status row is updated.

## Test plan

The selection script should have a tiny fixture table for two mock candidates:
one improves experimental organization but worsens GE address recall, and one
improves the worst person/address gap. Assert the second ranks first under
the frozen rule. A second fixture tests a confidence-interval tie, which
keeps the earlier epoch. Real retrospective evaluation is an integration
check; no model training or sealed-set evaluation is part of tests.

## Done criteria

- [x] The selection protocol was written before epoch scores were inspected.
- [ ] All four existing v17 checkpoints were staged and scored on the same
      development set, or a gate failure is recorded without bypass.
- [x] The ranking uses per-country people and addresses; org is experimental.
- [ ] Source-cluster uncertainty and passing-slice regressions are visible.
- [x] The sealed set has not been used and no new training ran.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- Any v17 epoch checkpoint or parser v7 artifact is missing or changed.
- Plan 004's artifact gate is not complete.
- Development membership labels from Plan 002 cannot be validated.
- An evaluator would use the sealed final set or a different gold corpus for
  different epochs.
- Scores are too unstable to choose an epoch; report uncertainty and stop
  short of a model change.

## Maintenance notes

Future training may still log synthetic validation for overfitting diagnosis,
but release selection is a separate, predeclared decision. Preserve all epoch
checkpoints, corpus hashes, and source-level denominators. A new country
requires updating the target matrix before its data is used for selection.
