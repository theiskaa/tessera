# Plan 001: Produce an unambiguous v17 accuracy and latency scorecard

> **Executor instructions:** Work through the steps in order and verify each. Do not run
> `trainer train`, `trainer generate`, `trainer prepare`, or change `models/` or the site.
> Stop on a condition below instead of improvising. Update this plan's status in
> `plans/README.md` when done.
>
> **Drift check first:** `git diff --stat 2361304..HEAD -- internal/bench/silver/measure.sh bench/web/measure.py trainer/src/main.rs trainer/src/detect_eval.rs plans/README.md`.
> Compare the current-state excerpts below with live code if any in-scope path changed.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** LOW for reporting, MED for benchmark reproducibility
- **Depends on:** none
- **Category:** tests, performance, quality
- **Planned at:** commit `2361304`, 2026-09-26

## Why this matters

V17 changed both silver training data and the synthetic corpus. It gained person F1 in
GB/GE/JP but Georgian address recall fell. The short report currently prints lenient
**F1** under a vague “found” interpretation, while the release target is lenient
**recall**. The browser latency of this wider model is also an estimate, not a
measurement. A decision scorecard must separate the actual metrics and measure the
current bundle before another training experiment is chosen.

## Current state

- `internal/bench/silver/measure.sh:8-10` loops over five countries and prints `$8`
  (exact F1) and `$11` (lenient F1) from the evaluator's table. The raw table at
  `runs/eval-v4/all-v17-strict.md` has columns `exact P/R/F1`, `lenient P/R/F1`;
  `$10` is lenient recall in the shell split.
- `trainer/src/detect_eval.rs:166-191` always passes the known gold country as a
  `Query.country_hint`. `tessera/src/lib.rs:468-475` also supports no-hint inference.
- `bench/web/measure.py:35-39,365-413` serves the repo root and hardcodes
  `models/tessera-v1.safetensors`. Its browser page already accepts `modelUrl` and
  `integrity` from the Python harness. The two unshipped bundles exist at
  `runs/detector-wide-v16/bundle/tessera-v1.safetensors` and
  `runs/detector-wide-v17/bundle/tessera-v1.safetensors`.
- Current v17 real scores are in `runs/eval-v4/all-v17-strict.md`; v16 is in
  `runs/eval-v4/all-wide16-all.md`. The parser is v7 in both bundles. The current
  five-country exit thresholds are in `internal/plans/HANDOFF.md:20-26` (people and
  addresses have priority; org is experimental).
- Match the Rust style in `CLAUDE.md`: `anyhow` in trainer, small private helpers,
  useful comments only. Keep existing CLI defaults so historical reports remain
  comparable. Do not put private document text in a report or git.

## Commands you will need

| Purpose | Command | Expected result |
| --- | --- | --- |
| Shell check | `sh -n internal/bench/silver/measure.sh` | Exit 0 |
| Python check | `python3 -m py_compile bench/web/measure.py` | Exit 0 |
| Rust tests | `cargo test -p trainer detect_eval` | Exit 0 |
| Rust format | `cargo fmt --all --check` | Exit 0 |
| Trainer binary | `cargo build --release -p trainer` | Exit 0; subsequent CLI runs use new flag |
| Build browser package | `just wasm` | Exit 0; `tessera/pkg/index.js` exists |
| Existing-bundle eval | `./target/release/trainer eval --gold data/interim/review/gold-all.jsonl --bundle runs/detector-wide-v17/bundle/tessera-v1.safetensors --country-hint-mode auto --out runs/eval-v17-auto --report internal/reports/m7/v17-auto.md` | Exit 0 after the new flag is implemented; auto mode is named in output |

## Scope

**In scope:** `internal/bench/silver/measure.sh`, `bench/web/measure.py`,
`trainer/src/main.rs`, `trainer/src/detect_eval.rs`, private aggregate outputs under
`internal/reports/m7/`, temporary eval outputs under `runs/`, and the status row in
`plans/README.md`.

**Out of scope:** `models/`, `site/`, tokenizer, model weights, training configs,
private gold labels, and any training or new data generation.

## Git workflow

Do not commit or push unless the user asks. Keep private gold and aggregate working
reports in ignored `internal/` or `runs/`; do not stage them. If a commit is later
requested, use one lowercase single-line message with no trailer, per `CLAUDE.md`.

## Steps

### 1. Correct the short report's columns and names

Change `measure.sh` to print exact F1, lenient precision, lenient recall, and lenient
F1 explicitly, with no “found” alias. Preserve the existing five-country report
order for the v16/v17 comparison. Add a small check in the script or a companion
read-only assertion that the v17 GB address row parses as exact F1 `64.2`, lenient
recall `97.9`, and lenient F1 `90.9`; the shell's pipe fields must not shift.

**Verify:** `sh -n internal/bench/silver/measure.sh` exits 0; rerunning the script
on the v17 bundle prints those three GB values under distinct labels.

### 2. Add an explicit no-hint evaluation mode

Add a trainer `eval --gold` option for country-hint mode `known` (current default)
or `auto` (pass `Query::default()` with no country hint). Apply it only to the
library prediction path in `detect_eval.rs`; keep the gold case's country for
stratification. Record the chosen mode in the evaluator JSON/Markdown so reports
cannot be mixed. Add one test using a national phone without country evidence:
known GB can read it and auto does not, matching `tessera/src/lib.rs:815-846`.

**Verify:** `cargo test -p trainer detect_eval`, `cargo fmt --all --check`, and
`cargo build --release -p trainer` exit 0.
Run the same v17 gold file in `known` and `auto` mode into distinct `runs/` output
directories; both reports name their mode and show per-country phone precision and
recall. Do not replace the existing v17 report.

### 3. Let the browser harness select an existing bundle

Add `--bundle REPO_RELATIVE_PATH` to `bench/web/measure.py`, defaulting to its
current shipped path. Reject a path outside the repo root, a missing file, or a
path that cannot be served by its local HTTP server. Use the selected path in both
`sizes()` and `base_cfg.modelUrl`, and derive integrity from those exact bytes.
Record bundle path and SHA-256 in the JSON output. Do not copy a candidate over
`models/tessera-v1.safetensors`.

**Verify:** `python3 -m py_compile bench/web/measure.py` exits 0. `just wasm` exits
0. Run Chrome on v16 and v17 with the same `tessera/pkg`, machine, fixture, and
browser settings, writing two JSON files under `internal/reports/m7/`; each JSON
identifies a different bundle hash. If Chrome/WebDriver is unavailable, STOP and
report the missing dependency rather than replacing browser time with native time.

### 4. Write the decision scorecard

Write `internal/reports/m7/v17-decision.md` with: bundle hashes; per-country
person exact F1, address exact F1, address lenient recall, and parser exact parse;
v16→v17 deltas; known/auto phone metrics; Chrome 10,000-character warm median and
p95 for each bundle against 150 ms; and the baseline's environment. Show
organization metrics in a separate experimental section. Say that v17 changed two
training inputs, so the comparison cannot attribute a gain to either one. Mark
real end-to-end contact accuracy “pending Plan 002” instead of estimating it.
For parser diagnosis, inspect only the existing real-address **development** half:
count error families by country and component label, including district, suburb,
unit, and PO box. Keep test-half results aggregate only. Mark proposed greedy
versus Viterbi decoding as an unproven ablation, not a fix.

**Verify:** A small script confirms the report includes all
five country codes and separate `lenient recall` and `lenient F1` labels. Every
number traces to a named JSON/Markdown source or browser JSON. Compare
`git status --short` with its initial snapshot: no new model, site, or config
changes. The two detector configs were already modified for v17 at planning time.

## Test plan

- The new CLI mode has a national-number test for `known` versus `auto` and
  preserves `known` as the default. Follow the nearby `detect_eval.rs` test style.
- The short report's column check uses the existing v17 GB address row, which
  distinguishes 97.9 recall from 90.9 F1.
- Browser command is an actual measurement gate, not a unit test. Record raw JSON.

## Done criteria

- [x] The short report names and prints both lenient recall and lenient F1 correctly.
- [x] Both phone-hint modes are measured and labelled on the same gold set.
- [x] V16 and v17 have browser measurements with the same harness and environment.
- [x] The decision report lists every current target by country and its numeric gap.
- [x] Parser development errors are counted by country and component label;
      test-half examples were not used for diagnosis.
- [x] No training command ran; `models/`, `site/`, and training configs are unchanged.
- [x] `plans/README.md` status row is updated.

## STOP conditions

- Current evaluator columns or bundle paths differ from the excerpts above.
- The browser or WebDriver is unavailable; record the blocker, do not invent latency.
- The v16 and v17 comparison uses different gold documents or runtime package builds.
- Any step would require changing private gold labels or the shipped bundle.

## Maintenance notes

Future evaluation scripts must use metric names from the raw evaluator, not “found”
for both recall and F1. Browser results depend on the package build and host; keep
both hashes and environment with each report.
