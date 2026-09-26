# Phase 019: Preserve the stronger duplicate window prediction

Status: implemented and reviewed after the V19c paired evaluation. The shipped model and site are unchanged.

## Finding

`tessera/src/chunk.rs:333-343` sorts candidate spans by start and descending end, then calls `dedup_by` on `(start, end, kind)`. The duplicate key excludes confidence. The later ranking includes confidence, but the duplicate with lower confidence may already have survived while the stronger one was removed. The current duplicate test at `chunk.rs:517-527` uses identical confidence, so it cannot detect this.

Two detector windows can decode the same span with different context and confidence. The confidence can affect attachment and contact confidence (`tessera/src/group.rs:522-527,809-826`), so this is more than a display-value issue. It is not evidence that span precision or recall has changed. The impact needs measurement on long real documents. Do not change the frozen V17/V19c runtime before paired scoring.

## Scoped change

1. Add a test with the same `(start, end, kind, source)` twice at confidences `0.7` and `0.9`, in both input orders. The merged entity must retain `0.9`.
2. Rank exact duplicates by the existing priority before deduplication, or group them and explicitly select the highest-ranked entity. Keep the existing rules-before-model and overlap-resolution behavior. Avoid changing spans that have no duplicates.
3. Run focused `chunk` tests, native and wasm golden vectors, and V17 fixture output comparison. On long documents, compare full extraction and contact output before and after; record whether only confidence changes. If spans or grouping change, score the same frozen real development set before retaining the fix.
4. Review the diff and measured outputs, then update this plan's status. Do not ship a new bundle or site as part of this phase.

## Acceptance

- Duplicate winner is deterministic and independent of insertion order.
- Nonduplicate entity spans and rule priority stay identical in regression tests.
- Any changed confidence or contact ranking is explicitly reported; no accuracy gain is claimed without real evaluation.

## Result

The old code failed the new two-order confidence test; the corrected code passes. The full 877-document V17/V19c before/after entity comparison found zero output differences, including confidence. Complete contact fixture reports are byte-for-byte identical for both bundles. Release library tests, Clippy, format, diff, and wasm target checks pass. See [review](019-review.md). This fixes the edge case but does not explain V19c's country-score trade-off.
