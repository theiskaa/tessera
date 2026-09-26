# Phase 017: Audit one-to-one lenient matching

## Finding

The existing reviewed-document evaluator counts a gold span as leniently found if **any** same-kind prediction overlaps it, and counts a prediction as leniently correct if **any** same-kind gold span overlaps it. One broad prediction can therefore credit two nearby gold spans; two overlapping predictions can both receive credit for one gold span. The current score is retained as `lenient` so the frozen V17/V19c comparison remains comparable.

## Change and verification

Add `lenient_one_to_one` to the JSON score for each kind and slice. It uses a maximum-cardinality matching within each document, with each span assigned at most once. Existing `lenient` and report columns keep their meanings. Exact matches also count each gold span at most once, so duplicate external predictions cannot push exact recall above 100%; this has no effect on nonduplicate runtime output. A focused unit test covers one-prediction-to-two-gold, two-predictions-to-one-gold, an augmenting path, and duplicate exact predictions. Re-evaluate the same frozen development set with the current binary for V17 and V19c; report any difference between legacy and one-to-one recall by country. Treat the one-to-one score as a diagnostic until the target protocol is explicitly versioned.

## Review status

Implementation, focused tests, and release-trainer V17/V19c real-document rescores passed on 2026-09-26. The Georgian found-recall gain is identical under legacy and one-to-one overlap counting. See [review](017-review.md).
