# Tessera improvement plans after v17

Written 2026-09-26 against commit `2361304`. Read the [review](review-2026-09-25.md)
for evidence and the complete v16/v17 comparison. The plans below are for execution
after this planning session; creating them does not change the library, model, or site.
At planning time, `configs/detector-silver.toml` and `configs/detector-wide.toml` already
have uncommitted v17 input changes. Executors must record initial `git status --short`
and compare against it; those existing modifications are not caused by these plans.
The focused [training-path review](review-2026-09-26-training-path.md) adds Plans 009 and 010.

## Operating decision

At plan time, **no new model training run** was authorized. The user later authorized a
controlled retrain after two reviews when the existing results remained poor. V18's
silver-hygiene diagnostic was reviewed and completed on 2026-09-26; see the private
`internal/reports/m7/pretrain-review-{1-data,2-evaluation}.md` and
`v18-clean-silver-experiment.md`. Its real evaluation failed the frozen regression
guard; see `post-v18-review-{1-data,2-evaluation}.md` and `v18-v17-comparison.md`.
V19's first Georgian contact-list input was stopped during epoch 1 after its sample
audit found unlabelled business names. V19b was also stopped before epoch 1 ended
after an address-format audit found its plain-number/comma bias. V19c completed and
improved Georgian address found recall by 10.6 points on development data, but lost
2.4 points in passing US address exact F1 and failed the frozen guard; see private
`v19c-v17-comparison.md` and `post-v19c-review-{1-data,2-evaluation}.md`.
The user then requested diagnosis and correction of the recurring data/selection
problems before any further training. Phase 021 supersedes dose-only retraining.
V20 had already started when this correction arrived. The user interrupted its
trainer and watcher during epoch 3; both PIDs are gone, and no final bundle was
exported. V20 is aborted and unscored. Do not start another run or treat V20 as
a selectable model.
None of these runs is a release-selection run: the real contact
ownership set and fresh sealed publishers are still pending. Keep
`models/tessera-v1.safetensors` and the live site unchanged. Further runs require a
new hypothesis and review of V18's results. Shipping still requires the user's
separate approval and the per-country targets in `internal/plans/HANDOFF.md:20-26`.

## Execution order and status

| Plan | Result | Priority | Effort | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| [001](001-measure-v17-honestly.md) | Exact metric labels, a v17 decision scorecard, and measured browser latency | P1 | M | — | DONE 2026-09-26; [review](001-review.md) |
| [002](002-measure-real-contacts.md) | Real end-to-end contact evaluation with precision, recall, and false-card counts | P1 | L | 001 metric definitions | PARTIAL; [checkpoint review](002-review.md) |
| [003](003-audit-and-gate-data.md) | Silver-label audit and fail-closed parser sample checks | P1 | M | — | PARTIAL; [checkpoint review](003-review.md) |
| [004](004-bind-export-to-artifacts.md) | Export rejects stale or unvalidated quantized weights | P1 | S | — | DONE 2026-09-26; [review](004-review.md) |
| [005](005-freeze-selection-protocol.md) | Choose future checkpoints against real per-country targets, without retraining now | P1 | M | 001, 002, and 004 | PARTIAL; [checkpoint review](005-review.md) |
| [006](006-country-scale-readiness.md) | Evidence-based design for adding countries without silent quality loss | P2 | M | 001 and 003 results | DONE 2026-09-26; [review](006-review.md) |
| [007](007-parser-dispatch.md) | Constant-time parser country dispatch for ISO codes | P2 | S | 006 inventory | DONE 2026-09-26; [review](007-review.md) |
| [008](008-index-term-flags.md) | Indexed term flags independent of dictionary scan length | P2 | S | 006 inventory | DONE 2026-09-26; [review](008-review.md) |
| [009](009-avoid-silver-repeat-copies.md) | Reuse encoded silver examples across repeated training draws | P1 | M | 006 scale inventory | DONE 2026-09-26; [review](009-review.md) |
| [010](010-fail-closed-detector-input.md) | Reject malformed detector labels at the trainer boundary | P1 | M | 003 rules | DONE 2026-09-26; [review](010-review.md) |
| [011](011-reject-run-reuse.md) | Refuse reused training directories with checkpoints before mutation | P1 | S | 004 artifact gate | DONE 2026-09-26; [review](011-review.md) |
| 012 | V18 cleaned-silver diagnostic against frozen real gold | P1 | L | 001, 003, 004 | DONE 2026-09-26; failed quality guard; [review](012-review.md) |
| 013 | V19 Georgian contact-list diagnostic | P1 | L | 012 reviews | DONE 2026-09-26; V19c failed US regression guard; [review](013-review.md) |
| 014 | Verify zero-activation kernel shortcut in a quiet paired browser run | P2 | S | 001 runtime measure | DONE 2026-09-26; [review](014-review.md) |
| [015](015-audit-gold-hosts.md) | Verify source-host overlap and add host slices to detector development reports | P1 | M | 001 metric definitions | DONE 2026-09-26; [review](015-audit-gold-hosts.md) |
| [016](016-window-size-ablation.md) | Test larger detector windows against duplicate inference work | P2 | M | 001 browser baseline | DONE 2026-09-26; 20.9% lower quiet browser median; [review](016-review.md) |
| [017](017-audit-lenient-matching.md) | Count one-to-one overlaps alongside the existing lenient detector metric | P1 | S | 001 metric definitions | DONE 2026-09-26; V17/V19c paired rescore checked; [review](017-review.md) |
| [018](018-count-fields-on-false-cards.md) | Count field errors inside false contact cards | P1 | S | 002 metric version 2 | DONE 2026-09-26; [review](018-review.md) |
| [019](019-preserve-best-duplicate-confidence.md) | Preserve higher confidence for duplicate window predictions | P2 | S | 013 paired evaluation | DONE 2026-09-26; output unchanged on 877 gold docs; [review](019-review.md) |
| [020](020-half-dose-ge-contact.md) | Test half-dose Georgian contact-list data after V19c guard failure | P1 | L | 013 reviews, 019, 016 | RUNNING from 14:18 Asia/Tbilisi; [launch review](020-review.md) |
| [021](021-diagnose-before-retrain.md) | Repair silver supervision and checkpoint selection before another run | P1 | L | 003 audit, V19c error review | IN PROGRESS; duplicate guard and QA candidate checked; [review](021-review.md) |

Plans 001, 003, and 004 can proceed independently. Plan 002 uses the definitions
fixed by 001; its private annotation work can begin while 001 runs. Plan 005 uses
the real development metrics and artifact gate to freeze checkpoint selection.
Plan 006 uses the measured runtime and data audit; it does not add a country or
change tokenization.

## What the v17 result already tells us

V17 improved person exact F1 in GB, GE, and JP but remains below the 90% target in
four of five countries. Georgian address exact F1 rose from 30.9% to 35.7%, while
lenient **recall** fell from 59.8% to 57.2%; German address recall is 93.7%, below
95%. The short `v17-measure.txt` prints lenient **F1**, so use the full per-country
report for recall. The same parser v7 was used in v16 and v17; its exact full-address
rate is 82.3% overall on the v17 evaluation. Browser latency and real contact-card
accuracy have not been measured for v17. These gaps rule out shipping now.

## Gate before proposing another training run

All of the following must be recorded before treating a candidate as selectable for
release. A failed gate stays visible; a diagnostic training run does not satisfy it.

1. V16 and v17 are compared on the same named real-document set, with exact F1 and
   lenient precision, recall, and F1 shown separately per country. The default
   no-hint phone mode and the hinted mode are distinguished.
2. V17 browser latency on the 10,000-character fixture is measured, with the 150 ms
   target and the measured environment shown. No arithmetic estimate substitutes.
   Parser error families are counted on the existing development half; test
   examples are not used to tune an explanation or decoder change. If the wide
   bundle misses 150 ms, the next work item is a profiled, output-preserving
   runtime optimization; a still wider model is not the default answer.
3. Real `extract_contacts` results and false contact cards are measured separately
   from detector and parser scores; any fresh final test is kept sealed until the
   evaluation definition and candidate are frozen.
4. Silver labels have a source-by-country-by-kind audit and unresolved labels are
   quarantined or corrected by review. Parser shards reject malformed span arrays
   and missing country coverage. Export approval is bound to current artifacts.
5. A proposed experiment changes one major factor at a time and identifies the
   specific weak slices it is meant to improve. It includes an explicit rollback
   condition for countries that regress. Its checkpoint-selection rule is frozen
   on development data before the run, with the fresh final set reserved for
   confirmation.

## Deliberately deferred

- A larger or separate model, country packs, and parser Viterbi are experiments,
  not fixes established by the current evidence. No training starts under these
  plans.
- Markdown masking, the stale parse UI result, tagged-email grouping, and the site
  CI build gate are confirmed follow-up issues in the review. They are not blockers
  for the next **training decision**; site and public API work gets its own scoped
  implementation plan before release.
- Organization quality remains experimental and outside the current person/address
  exit targets, as recorded in `internal/plans/HANDOFF.md:23-26`.

## Findings considered and rejected

- Restoring the parser veto on detected addresses: `internal/plans/HANDOFF.md:27-30`
  records its deliberate removal. The stale API documentation was corrected;
  the veto is not a proposed accuracy fix.
- Multiplying model inference by country count: the runtime uses one shared model.
  Country growth affects data, representation, verification, and some linear
  preprocessing/feature lookups; it does not run one network per country.
- Changing Latin-only email-to-person matching in `tessera/src/group.rs:380-469`
  without reviewed real GE/JP ownership labels: the current refusal is an
  intentional precision choice; Phase 002 must measure its actual cost first.
- Reindexing the grouper's repeated block/entity scans at
  `tessera/src/group.rs:315-331,476-501,747-772` before a group-stage profile:
  the complexity grows with blocks and entities, but the present 10k-character
  browser bottleneck is model inference. Measure representative long contact
  documents before changing this output-sensitive code.
