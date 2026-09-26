# Phase 020: Test a smaller Georgian contact-list supplement

Status: aborted during epoch 3 after step 20,200/37,276 at the user's request to diagnose quality problems before more training. Both trainer PID 94834 and watcher PID 94948 exited; no quantized artifact or exported bundle exists. The private frozen brief is `internal/reports/m7/v20-ge-contact-dose-experiment.md`.

## Reason

V19c raised Georgian address found recall from 57.2% to 67.8% on the existing real development set, but US address exact F1 fell 92.4% to 90.0%, beyond the frozen 2-point guard. Two post-result reviews found a real list-format signal, concentrated on numbered branch pages, and showed that V19c epoch 3 regressed further in US and JP. A smaller supplement is a single controlled dose change that may preserve some recall without the same cross-country cost. It may also fail; this is a diagnostic, not a release run.

## Boundary and checks

The V20 input is the first 100 already audited V19c contact-list documents, with unchanged text and spans: 300 unique addresses, 75 in each plain/`#`/`№`/`N` number form, and 34/33/33 documents across the three layouts. The generated file and config are ignored private inputs with SHA-256 values in the frozen brief. V17's other inputs, seed, architecture, parser table, and schedule stay fixed. The gold set supplies only an exact-overlap exclusion check and is never trained on.

After the quiet runtime benchmark and Phase 019 review, start a fresh four-epoch run. Quantize/export only if the artifact gate passes. Evaluate V17 and V20 with the same release trainer on all 877 frozen real development documents in auto primary and known diagnostic modes, plus paired URL-host intervals. Reject V20 for any >2-point loss on a previously passing primary slice, GE found-recall gain under 5 points, GE address exact F1 gain under 1 point, or a worse maximum target shortfall than V17. Review the result and update this plan; do not ship or use a sealed publisher set.
