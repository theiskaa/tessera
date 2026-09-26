# Plan 018: Count fields assigned to false contact cards

> **Executor instructions:** Follow the steps and gates below. Work only on the contact evaluator and its focused tests; do not alter grouping behavior, model weights, training data, or release assets. The working tree already has unrelated uncommitted changes, so inspect the live diff before editing. Do not commit or push unless the user asks.
>
> **Drift check first:** `git diff -- trainer/src/group_eval.rs plans/README.md`; compare the `score` loop below with the live code before changing it. Planned against commit `2361304` plus the uncommitted Phase 002 evaluator changes.

## Status

- **Priority:** P1
- **Effort:** S
- **Risk:** LOW; aggregate diagnostic counts change, exact-card matching stays fixed
- **Depends on:** Phase 002's metric version 2
- **Category:** correctness, evaluation
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

The contact report claims to count extra assigned fields and wrong ownership. `trainer/src/group_eval.rs:293-299` skips any predicted card whose anchor does not match a gold card. Its attached phone, address, or email therefore appears in the false-card count but disappears from the field-error counts. This understates the diagnostic error total exactly when the model invents a card. The primary exact-card precision/recall/F1 already penalize the false card and must not change.

## Current state

- `trainer/src/group_eval.rs:225-315` scores one `GoldCase` and one `Extraction`. At lines 293-299 it finds a predicted anchor and immediately `continue`s if the anchor is absent from `case.contacts`; the loop at lines 300-313 classifies only members of cards with gold anchors.
- `gold_owner` maps a gold member span to its gold anchor. Members absent from it are either unassigned gold entities or invented predictions; both currently count as false assigned fields when attached to a valid anchor.
- `GrouperCounts` carries `false_assigned_fields`, `false_assigned_by_kind`, and `wrong_ownership_fields`; `render_markdown` prints them. Existing test patterns are at `trainer/src/group_eval.rs:564-694`.
- `tessera/src/group.rs` owns production grouping. This plan changes **only** evaluation accounting. Real contact ownership annotations are still pending in Phase 002; synthetic fixture scores are not product accuracy claims.
- Follow `CLAUDE.md`: small Rust helpers, `anyhow` for trainer errors, no decorative comments, and no commits unless requested.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer group_eval` | Exit 0 |
| Format | `cargo fmt --all --check` | Exit 0 |
| Lint | `cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Diff check | `git diff --check` | Exit 0 |

## Scope

**In scope:** `trainer/src/group_eval.rs`, `plans/018-review.md`, and the Phase 018 row in `plans/README.md`.

**Out of scope:** `tessera/src/group.rs`, fixture source files, contact annotations, metric matching of exact cards, model/training paths, `models/`, and the site.

## Steps

### 1. Characterize the missing counts

Add focused tests in `trainer/src/group_eval.rs` using the existing `gold`, `entity`, and `contact` test helpers. Cover a predicted false-anchor card with an attached extra phone and no gold card: exact-card false positives remain one, and both the anchor and phone are counted by kind as false assigned fields. Cover a false-anchor card that takes a member of an existing gold card: that member counts as wrong ownership. Assert the existing valid-anchor extra-phone case remains unchanged.

**Verify:** `cargo test -p trainer group_eval` → the new characterization tests fail before the fix while existing tests pass.

### 2. Classify members of every predicted card

In `score`, keep a predicted card's gold-member set optional instead of skipping the card when its anchor is absent from gold. For each member, skip it only when the corresponding gold card contains that exact member. Otherwise, use the existing `gold_owner` split: a member of another gold card increments wrong ownership; all other attached members increment false assigned fields and its kind count. Leave `exact_card_matches`, `unmatched_predicted_contacts`, and `false_card_documents` untouched. Bump the Markdown contact metric version from 2 to 3 and state that extra assigned fields include members of false-anchor cards; do not rename the already published columns. Old version-2 reports remain historical and must not be silently compared as identical field-error definitions.

**Verify:** `cargo test -p trainer group_eval` → all tests, including the new false-anchor cases, pass. Compare the 18-fixture V17 report if available: exact-card TP/predicted/gold counts must be identical before and after, while field-error counts may rise.

### 3. Review the result

Run format, lint, and diff checks; inspect `git diff -- trainer/src/group_eval.rs` to ensure no production grouping code changed. Record the new metric count meaning and focused test result in a private or tracked review note, and update the Phase 018 row in `plans/README.md`.

**Verify:** all four commands above exit 0; `git status --short` shows no model, site, or data changes caused by this phase.

## Done criteria

- [x] False-anchor cards contribute assigned members to field-error counts by kind.
- [x] A stolen gold member on a false card contributes to wrong ownership.
- [x] Exact-card precision, recall, F1, and false-card counts are unchanged on the existing fixture set.
- [x] New reports identify contact metric version 3 and the broader field-error count.
- [x] Focused tests, formatting, lint, and diff checks pass.
- [x] Phase 018 status and review are recorded.

## STOP conditions

- The live evaluator no longer has the early `continue` at the cited location.
- The proposed change alters `tessera/src/group.rs` or exact-card matching.
- A production output can contain the same member in multiple cards and the desired counting unit is unclear; report the ambiguity before changing totals.

## Maintenance note

The field-error counters are diagnostics per assigned member, not a substitute for exact-card precision/recall. When Phase 002's real contact set is reviewed, compare both false-card and field-error rates per country and publisher; do not present fixture counts as real accuracy.
