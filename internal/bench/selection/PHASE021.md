# Phase 021 epoch selection input

This is a private preparation note. The script ranks real-document reports; it does not train, quantize, export, evaluate, or approve release.

Before use, rescore the frozen V17 bundle and every completed epoch bundle with the **same** evaluator executable, automatic country hints, and `data/interim/review/gold-all.jsonl`. Each epoch bundle must be exported from a separately staged, quantization-gated copy of that epoch checkpoint. Keep the original run's `metrics.jsonl`, `summary.json`, `checkpoints/`, and input snapshot intact. The script rejects an aborted run or a missing epoch report.

The epoch manifest is JSON:

```json
{
  "epochs": [
    {
      "epoch": 1,
      "checkpoint_sha256": "sha256 of original run checkpoints/epoch-1.mpk",
      "config": "path to staged run config.toml",
      "input_snapshot": "path to staged run input_snapshot.json",
      "quantized": "path to staged run quantized.safetensors",
      "gate": "path to staged run quantize.json",
      "bundle": "path to staged bundle tessera-v1.safetensors",
      "report": "path to that bundle's eval/review.json"
    }
  ]
}
```

Use absolute paths or run the selector from the repository root. The selector checks the frozen V17 and gold hashes; report gold, evaluator, and bundle hashes; each original checkpoint hash; each passing quantization gate; the detector hashes embedded by the exporter; every completed epoch; and the frozen per-country score guard. It reports a selected epoch only when one passes. Its `release_eligible` flag stays false because real contact ownership, publisher robustness, sealed final evaluation, parser quality, and browser latency have separate gates.

The selector proves the **detector** checkpoint-to-bundle chain under the current export format. It checks report case counts, gold support by kind and country, exact-score arithmetic, and lenient metric ranges against the selected gold. Eligibility also checks that exact F1 does not fall by more than two absolute points from the baseline for every supported-country/kind slice, each supported-country overall slice, all five global kinds, and global overall. Address one-to-one found recall has the same baseline-relative guard in each supported country. This regression guard applies even when a baseline is below an aspirational `TARGETS` value. Any guard slice with fewer than 50 gold labels requires manual review and cannot select an epoch automatically; the pinned dev-v2 gold has at least 113 in every supported-country/kind slice. The separate GE address gain requirements and target-based ranking remain in force; these comparisons use the verified integer counts rather than rounded report rates. It does not establish that the parser or frozen n-gram table matches V17. Check both parser hashes and feature settings separately before attributing a detector gain to changed training data. Existing V17/parser-v7 runs predate the new input-snapshot gate and cannot simply be re-exported through it; design and review a provenance-preserving migration before staging future epoch bundles.

Migration review note: `export::load_run` now requires both a version-2 quantization gate and a run-bound input snapshot for **each** network; parser v7 has neither. Re-running quantization would also require that missing snapshot. Do not create one from today's mutable parser shards. A defensible detector-only diagnostic route is a narrowly allowlisted bridge tied to the frozen V17 bundle hash plus the exact parser-v7 config, checkpoint, and quantized hashes, with golden-vector verification. Record that legacy exception in bundle metadata and keep it release-ineligible. A replacement parser trained later from fully snapshotted inputs is the other route. Neither route is implemented or approved yet.

Current parser-v7 artifact hashes, checked 2026-09-26: config `972cf030b9b90888169762906b4f19dbe0da2612bd1b334f3887633f1ed0c309`; checkpoint `0a4941dbfcfb8bc047245401d28ff97bf4e0fc432f9f9491ddb872d1b96d736b`; quantized `294ed5247b57f98883f2d6c61e53a265989fb003ce4dabc2c400fc54a1b68fcf`; version-1 gate `12c68c6349e241cd2bdf0dcc5e5d8c14ddd81ad69dee069ee174e912fc579a9c`. These are identity checks, not proof of its original parser training inputs.

I compared every one of the 24 `parser.*` tensor names, dtypes, shapes, and raw data bytes in `parser-small-v7/quantized.safetensors` against the frozen V17 bundle (SHA-256 `cb056415282f79504c5b14f9766f126fc5379cf3941503bb030a00fec530b4c6`). All 24 match exactly, with none missing. This proves the legacy parser *weights* in that quantized file are the same as the V17 parser weights; it still does not recreate parser input provenance. Any bridge should recheck this equality and hashes at export time, not rely on this note alone.

The private `verify_legacy_parser.py` now repeats those checks and writes `internal/reports/m7/parser-v7-identity.json` (SHA-256 `bb2091c05266ef17816592fc411674b983fe1563ba38a85cb8bde46b0c1f006e`). It binds the four parser files and baseline bundle to the fixed hashes, confirms the version-1 gate's recorded hashes, and compares all 24 parser tensors byte for byte. A focused test verifies valid identity, changed tensor bytes, and changed checkpoint rejection; all 12 selection tests pass. This report is evidence for a future narrowly scoped export bridge, not an export gate or release approval.

Export integration detail: the existing exporter builds parser golden vectors and a parser test report from the parser run's current `test.parquet`, then writes that report under the parser run's `eval/` directory. A legacy detector-only bridge must label those diagnostics as current-file checks, because parser v7 has no frozen test-shard hash, and should place its new report in the staged export directory so it does not modify the historical parser run. This is another reason not to simply disable the strict snapshot check.

```bash
python3 internal/bench/selection/select_phase021.py \
  --baseline-report PATH_TO_FRESH_V17_REVIEW_JSON \
  --baseline-bundle runs/detector-wide-v17/bundle/tessera-v1.safetensors \
  --manifest PATH_TO_EPOCH_MANIFEST_JSON \
  --run PATH_TO_COMPLETED_DETECTOR_RUN \
  --gold data/interim/review/gold-all.jsonl \
  --evaluator PATH_TO_THE_EXACT_EVALUATOR_EXECUTABLE
```

The current V17 historical JSON reports predate report-provenance fields. Rescore V17 with the evaluator executable chosen for the future matched experiment before using this script. No new training is allowed until the Phase 021 data and source gates are resolved.

## Corrected development v2

The selector defaults to the original 877-row `dev-v1` set. To use the reviewed 874-row correction, pass `--development-set dev-v2`, `--gold data/interim/review/dev-v2-ge-shell-exclusion-20260927-v1/gold-all.jsonl`, and `--baseline-report runs/eval-v17-dev-v2-auto-20260927/eval/review.json` with the same frozen V17 bundle and evaluator binary that produced that report. The native baseline report SHA-256 is `6930f8409940a952848c9a9ed5202a7cbe8346a87f7aa8da0d8e5eb0add3d185`; its evaluator SHA-256 is `d181440589158d00c4184f9e53af28125148e0b598c067cb652bdcbe9e28e068`. The selector checks these through the report and executable. It also checks the correction manifest, original inputs, all 874 raw source path/hash/URL/text links, retained-line identity, paired V17 predictions, and native scores before considering any candidate epoch.

The three excluded GE Economy records were empty news shells. The original v1 files remain unchanged and available for historical comparisons. Dev-v2 is development scoring only; it does not certify publisher disjointness or create a final holdout. Candidate epochs still need their own native reports on exactly the 874-row gold with the same evaluator binary and automatic country hints. No candidate epoch has been selected or trained by this lock.
