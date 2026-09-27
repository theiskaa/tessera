import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dev_v2_lock
import select_phase021 as selection


GOLD = dev_v2_lock.ROOT / dev_v2_lock.DIRECTORY / "gold-all.jsonl"
NATIVE_REPORT = dev_v2_lock.ROOT / "runs/eval-v17-dev-v2-auto-20260927/eval/review.json"
BASELINE_BUNDLE = dev_v2_lock.ROOT / "runs/detector-wide-v17/bundle/tessera-v1.safetensors"
EVALUATOR = dev_v2_lock.ROOT / "target/release/trainer"


class DevV2LockTests(unittest.TestCase):
    @unittest.skipUnless(GOLD.is_file(), "private dev-v2 proposal is unavailable")
    def test_frozen_proposal_is_exact_v1_subset(self):
        scores = dev_v2_lock.verify(GOLD)
        self.assertEqual(scores["v2"]["rows"], 874)
        self.assertEqual(scores["v2"]["all"]["micro"]["correct"], 13762)

    @unittest.skipUnless(GOLD.is_file(), "private dev-v2 proposal is unavailable")
    def test_changed_artifact_pin_fails_closed(self):
        changed = {**dev_v2_lock.EXPECTED,
                   str(dev_v2_lock.DIRECTORY / "v17-auto-predictions.jsonl"): "0" * 64}
        with patch.object(dev_v2_lock, "EXPECTED", changed):
            with self.assertRaisesRegex(ValueError, "pinned file changed"):
                dev_v2_lock.verify(GOLD)

    @unittest.skipUnless(GOLD.is_file(), "private dev-v2 proposal is unavailable")
    def test_pinned_files_without_raw_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in dev_v2_lock.EXPECTED:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dev_v2_lock.ROOT / name, destination)
            with self.assertRaisesRegex(ValueError, "raw source unavailable"):
                dev_v2_lock.verify(root / dev_v2_lock.DIRECTORY / "gold-all.jsonl", root)

    @unittest.skipUnless(GOLD.is_file() and NATIVE_REPORT.is_file(),
                         "private dev-v2 native V17 report is unavailable")
    def test_native_v17_report_matches_and_changed_score_fails(self):
        scores = dev_v2_lock.verify(GOLD)
        report = json.loads(NATIVE_REPORT.read_text())
        dev_v2_lock.verify_native_v17(report, scores)
        changed = copy.deepcopy(report)
        changed["systems"][0]["by_slice"]["country"]["GE"]["address"]["exact_tp"] += 1
        with self.assertRaisesRegex(ValueError, "GE/address exact_tp differs"):
            dev_v2_lock.verify_native_v17(changed, scores)

    @unittest.skipUnless(all(path.is_file() for path in (GOLD, NATIVE_REPORT, BASELINE_BUNDLE, EVALUATOR)),
                         "private dev-v2 native V17 report is unavailable")
    def test_selector_accepts_paired_baseline_before_epoch_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "epochs.json"
            manifest.write_text('{"epochs": []}')
            with self.assertRaisesRegex(ValueError, "epoch manifest is empty"):
                selection.select(
                    NATIVE_REPORT,
                    BASELINE_BUNDLE, manifest, directory, GOLD, EVALUATOR, "dev-v2",
                )


if __name__ == "__main__":
    unittest.main()
