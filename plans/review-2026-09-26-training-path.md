# Training and evaluation path review — 2026-09-26

Scope: detector training inputs, repeat sampling, evaluator input contracts, and the runtime stage profile. This was a read-only review of source and data while the V18 binary trained. Earlier reviews cover browser/API/site and the full country-scale inventory. No trained result existed at review time.

| Priority | Finding | Evidence | Impact | Effort / fix risk | Confidence | Action |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Repeated silver entries own duplicate feature vectors | `trainer/src/train.rs:249-261,371-384`; `trainer/src/dataset.rs:19-39` | At repeat 10, V18's 14,045 unique silver pieces become about 140,450 stored encoded entries before each batch clone. This raises memory pressure as countries/data grow; the live process is about 8.6 GiB RSS, including other memory. | M / medium; sampling order must stay fixed | High | [Plan 009](009-avoid-silver-repeat-copies.md) |
| P1 | Detector loaders can silently discard unknown labels and trust offsets | `trainer/src/detector.rs:158-171,220-261,302-320` | An unrecognized kind vanishes through `filter_map`; a malformed span can be dropped at piece cutting or make `reachable_spans` slice outside valid UTF-8. Bad training targets can proceed without a clear error. | M / medium; old data may fail | High | [Plan 010](010-fail-closed-detector-input.md) |
| P1 | A reused run can bind a stale checkpoint to a new config after early failure | `trainer/src/train.rs:112-137,370-443`; `trainer/src/quantize.rs:484-534` | A new `config.toml` can coexist with the previous `best.mpk`; if training fails before its first save, later quantization may approve old weights for the new config. | S / low; used names must change | High | [Plan 011](011-reject-run-reuse.md) |
| P2 | The development gold set contains exact duplicate text | `data/interim/review/gold-all.jsonl` read-only SHA-256 scan; `trainer/src/detect_eval.rs:80-112` | 877 named cases contain 872 unique text hashes. Four duplicate-text groups add five extra documents, all with identical labels within each group. The score gives those texts extra weight. This does not explain the large quality gap; keep the frozen corpus unchanged for V18 comparison and deduplicate a future fresh set before sealing it. | S / low if done on a new set | High | Fold into future sealed-publisher evaluation, not the frozen diagnostic |

## Vetting and limits

- The two input failures are not claimed to have corrupted V18. Its filtered silver export had independent offset, kind, and hash checks, and an exact-text scan found no overlap with the frozen gold set. The trainer itself still needs a fail-closed boundary for later sources.
- `trainer/src/detect_eval.rs:368-405` intentionally counts same-kind overlap separately on the predicted and gold sides. That is the established “found” metric; a many-to-one overlap is not reported as a newly discovered bug here. Exact F1 remains the boundary-sensitive target.
- The V17 Chrome stage profile on the 9,975-character fixture recorded 461.50 ms in detector inference of 475.24 ms total before term indexing. On a later busy-machine pass it recorded 404.42 of 415.31 ms. This confirms the inference stage dominates but does not prove an overall speed gain from unrelated source edits. The pending zero-activation skip needs a quiet paired benchmark after training.
- No current source edits were made as part of this review. The review created Plans 009 and 010 for later execution. The current V18 experiment, shipped bundle, and site remain untouched.

Not audited in this pass: dependency advisories, public JS API, site UI, and real contact ownership. Those areas have prior reviews or pending real annotations; this review makes no claim about their current state.
