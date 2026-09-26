# Plan 010: Reject malformed detector labels before encoding

> **Executor instructions**: Validate label contracts at the trainer boundary, with source path and row context. Keep existing handling of rule-layer email and phone spans and the documented unreachable-span accounting. Run each verification command before proceeding. If a STOP condition occurs, report it rather than silently dropping data.
>
> **Drift check (run first)**: `git diff --stat 2361304..HEAD -- trainer/src/detector.rs` and `git diff -- trainer/src/detector.rs`. The plan was written against uncommitted local work as well as commit `2361304`; compare the code excerpts below with the live file and stop if they differ.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED; old invalid training files may now fail and need quarantine
- **Depends on**: Phase 003's strict export rules for deciding how to repair source data; execute after V18 finishes
- **Category**: correctness / data quality
- **Planned at**: commit `2361304`, 2026-09-26

## Why this matters

`load_silver` currently maps `Kind::from_str_label` through `filter_map`, so an unrecognized kind disappears without an error. It also builds pieces before validating every original offset, and `reachable_spans` slices `text[s..e]` for accepted spans. A malformed input can lose labels or panic instead of producing a useful data error. Training on a silently altered target set can look like an accuracy plateau. Phase 003 added strict validation to the labelling exporter, but trainer inputs may be older exports or produced by another route; the trainer must check its own boundary.

## Current state

- `trainer/src/detector.rs:158-171`: synthetic Parquet `entities_json` uses `filter_map` and drops kinds outside detector `KINDS`; email and phone intentionally belong to the rule layer.
- `trainer/src/detector.rs:220-261`: `load_silver` parses each `SilverJson`, uses the same `filter_map`, then calls `pieces` and `reachable_spans` before encoding each piece. The counts report unreachable spans but do not report invalid labels.
- `trainer/src/detector.rs:302-320`: `reachable_spans` tests token boundaries and then evaluates `has_blank_line(&text[s..e])`; it assumes `s..e` is an in-bounds UTF-8 range.
- `trainer/src/detector.rs:69-116`: `encode_document` assigns BIO labels per span and later spans overwrite the same token positions if spans overlap.
- `plans/003-review.md`: the labelling exporter has a strict check; historical exported files were left untouched when the new check found unresolved issues. The active V18 input was separately validated and frozen. Do not mutate it during this plan.
- Conventions: `anyhow` with path/row context in the trainer; inline unit tests in `trainer/src/detector.rs`; `cargo fmt`, Clippy, and `cargo test -p trainer` are verification gates.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused tests | `cargo test -p trainer detector::tests` | New and existing detector tests pass |
| Trainer suite | `cargo test -p trainer` | All tests pass |
| Format | `cargo fmt --all --check` | Exit 0 |
| Lint | `cargo clippy -p trainer --all-targets -- -D warnings` | Exit 0 |
| Diff | `git diff --check` | Exit 0 |

## Scope

**In scope:** `trainer/src/detector.rs` and its inline tests, `plans/010-fail-closed-detector-input.md`, `plans/README.md` status.

**Out of scope:** `bench/silver/agent_label.py`; any silver/synthetic data file, especially V18; parser preparation; model architecture; email/phone detection behavior; shipped `models/` and site.

## Steps

### 1. Validate every label before filtering to detector targets

Add a private parser/validator for a `GoldJson` and the document text. Reject unknown kind strings, empty spans, out-of-bounds ranges, and non-UTF-8 boundaries. Accept `person`, `org`, and `address` as model targets. Keep `email` and `phone` recognized but excluded from detector BIO targets, because the rules layer supplies them. Reject overlapping model-target spans before `encode_document`; duplicate spans are overlaps. Do this before `pieces`, so an out-of-range label cannot disappear at a piece boundary.

**Verify:** `cargo test -p trainer detector::tests` passes focused tests for all named cases.

### 2. Apply the contract to both loaders

Use the same validation for synthetic `load_split` and JSONL `load_silver`, with `path:row` plus the label index in errors. Preserve the existing intentional skip accounting for reachable-but-unrepresentable silver labels (partial tokens, rule overlap, blank-line spans); do not redefine it as invalid. Make malformed data fail the load before training starts.

**Verify:** `cargo test -p trainer detector::tests` passes tests with a malformed JSONL row and a malformed Parquet row, including path/row evidence. A clean row with email/phone and one model span still encodes the model span.

### 3. Check the frozen input and review

Run a read-only validator against the V18 silver JSONL inputs and current synthetic Parquet train/valid shards, or add a CLI check mode if existing APIs cannot do so without model training. Record document and span counts before/after; they must match the frozen V18 manifest and current synthetic summary. Do not regenerate or edit those files. Run the suite, formatting, lint, and diff commands, then add `plans/010-review.md` and mark the README row DONE.

**Verify:** all commands pass; the read-only counts match; `git status --short` shows no model/data/config/site changes caused by this plan.

## Test plan

Use existing detector unit tests in `trainer/src/detector.rs` as the pattern. Cover unknown kind, known rule kind, empty and out-of-bounds span, a start or end inside a UTF-8 code point, duplicate/overlapping model spans, and valid adjacent spans. Assert contextual errors and unchanged valid counts.

## Done criteria

- [ ] No unknown/invalid label can be silently dropped or sliced unsafely in either loader.
- [ ] Email/phone remain recognized rule-layer kinds and are excluded from model BIO targets.
- [ ] Nonrepresentable but otherwise valid silver spans still follow the documented reachable accounting.
- [ ] Frozen V18 data is unchanged and loads to the same counts.
- [ ] Focused and full trainer tests, format, lint, and diff checks pass.

## STOP conditions

- A V18 input fails the new validation; record exact path/row and report before any repair.
- The loader currently relies on overlapping spans as a documented target convention.
- Fixing the issue requires changing data, the labelling exporter, model labels, or rules behavior.

## Maintenance note

Keep the trainer's input contract aligned with the exporter, but do not rely on the exporter as the only gate. Future country data sources and import routes should surface actionable path/row errors before a long training run begins.
