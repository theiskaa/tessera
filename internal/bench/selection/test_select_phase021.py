import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import select_phase021 as selection


def metric(gold, predicted, correct, found):
    exact = {"precision": correct / predicted if predicted else 0.0,
             "recall": correct / gold if gold else 0.0,
             "f1": 2 * correct / (gold + predicted) if gold + predicted else 0.0}
    lenient = {"precision": found / predicted if predicted else 0.0,
               "recall": found / gold if gold else 0.0,
               "f1": 2 * found / (gold + predicted) if gold + predicted else 0.0}
    return {"gold": gold, "predicted": predicted, "exact_tp": correct,
            "exact": exact, "lenient": lenient, "lenient_one_to_one": lenient,
            "boundary_accuracy": 0.0,
            "lenient_predicted_hits": found, "lenient_gold_hits": found,
            "one_to_one_predicted_hits": found, "one_to_one_gold_hits": found}


def report(gold, evaluator, bundle, *, ge_found=.5, ge_exact=.3, us_exact=.9):
    countries = {}
    for country in selection.COUNTRIES:
        address_correct = int(round(10 * (ge_exact if country == "GE" else
                                          us_exact if country == "US" else .8)))
        address_found = int(round(10 * (ge_found if country == "GE" else .9)))
        countries[country] = {
            "person": metric(10, 10, 9, 9),
            "address": metric(10, 10, address_correct, address_found),
        }
    kinds = {kind: metric(0, 0, 0, 0) for kind in selection.KINDS}
    for kind in ("person", "address"):
        kinds[kind] = metric(50, 50,
                             sum(countries[country][kind]["exact_tp"] for country in countries),
                             sum(countries[country][kind]["one_to_one_gold_hits"] for country in countries))
    overall = metric(100, 100, sum(row["exact_tp"] for row in kinds.values()),
                     sum(row["one_to_one_gold_hits"] for row in kinds.values()))
    return {"country_hint_mode": "auto", "provenance": {
        "gold_sha256": selection.digest(gold),
        "evaluator_sha256": selection.digest(evaluator),
        "bundle_sha256": selection.digest(bundle),
    }, "systems": [{"system": "tessera", "cases": 50, "overall": overall,
                    "by_kind": kinds, "by_slice": {"country": countries}}]}


class EpochSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        (self.run / "checkpoints").mkdir(parents=True)
        (self.run / "config.toml").write_text("trained config")
        (self.run / "input_snapshot.json").write_text("trained inputs")
        self.gold = self.write("gold.jsonl", "".join(
            json.dumps({"name": f"{country}-{number}", "country": country,
                        "input": "Alice Smith, 12 Main St",
                        "expected": [{"kind": "person"}, {"kind": "address"}]}) + "\n"
            for country in selection.COUNTRIES for number in range(10)))
        self.evaluator = self.write("trainer", "evaluator binary")
        self.base_bundle = self.write("base.safetensors", "base bundle")
        self.base_report = self.write_json(
            "base.json", report(self.gold, self.evaluator, self.base_bundle))
        self.manifest = self.root / "epochs.json"

    def write(self, name, value):
        path = self.root / name
        path.write_text(value)
        return path

    def write_json(self, name, value):
        return self.write(name, json.dumps(value))

    def add_epoch(self, epoch, **scores):
        checkpoint = self.run / "checkpoints" / f"epoch-{epoch}.mpk"
        checkpoint.write_text(f"checkpoint {epoch}")
        quantized = self.write(f"quantized-{epoch}.safetensors", f"weights {epoch}")
        snapshot = self.write(f"snapshot-{epoch}.json", "trained inputs")
        config = self.write(f"config-{epoch}.toml", "trained config")
        hashes = {"best_sha256": selection.digest(checkpoint),
                  "quantized_sha256": selection.digest(quantized),
                  "input_snapshot_sha256": selection.digest(snapshot),
                  "config_sha256": selection.digest(config)}
        gate = self.write_json(f"gate-{epoch}.json", {
            "gate_version": 2, "passed": True, **hashes,
        })
        header = json.dumps({"__metadata__": {
            f"detector_{key}": hashes[key] for key in
            ("best_sha256", "quantized_sha256", "input_snapshot_sha256")
        }}).encode()
        bundle = self.root / f"epoch-{epoch}.safetensors"
        bundle.write_bytes(len(header).to_bytes(8, "little") + header + f"bundle {epoch}".encode())
        evaluated = report(self.gold, self.evaluator, bundle, **scores)
        path = self.write_json(f"epoch-{epoch}.json", evaluated)
        return {"epoch": epoch, "checkpoint_sha256": selection.digest(checkpoint),
                "bundle": str(bundle), "report": str(path), "quantized": str(quantized),
                "input_snapshot": str(snapshot), "config": str(config), "gate": str(gate)}

    def decide(self, entries, metrics=None):
        if metrics is None:
            metrics = [entry["epoch"] for entry in entries]
        (self.run / "metrics.jsonl").write_text("".join(
            json.dumps({"epoch": epoch}) + "\n" for epoch in metrics))
        (self.run / "summary.json").write_text(json.dumps({
            "steps": 100, "best_epoch": metrics[0],
        }))
        self.manifest.write_text(json.dumps({"epochs": entries}))
        with patch.object(selection, "EXPECTED_GOLD", selection.digest(self.gold)), \
                patch.object(selection, "EXPECTED_BASELINE", selection.digest(self.base_bundle)):
            return selection.select(self.base_report, self.base_bundle, self.manifest,
                                    self.run, self.gold, self.evaluator)

    def test_selects_eligible_epoch_by_worst_shortfall_then_total(self):
        first = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        second = self.add_epoch(2, ge_found=.7, ge_exact=.5)
        decision = self.decide([first, second])
        self.assertEqual(decision["selected_epoch"], 2)
        self.assertTrue(decision["detector_lineage_verified"])
        self.assertFalse(decision["release_eligible"])
        self.assertNotIn("development_set", decision)
        self.assertTrue(all(row["eligible"] for row in decision["epochs"]))

    def test_rejects_us_regression_despite_ge_gain(self):
        entry = self.add_epoch(1, ge_found=.7, ge_exact=.4, us_exact=.8)
        decision = self.decide([entry])
        self.assertIsNone(decision["selected_epoch"])
        self.assertIn("US.address_exact_f1", decision["epochs"][0]["reasons"][0])

    def test_missing_completed_epoch_is_an_error(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        with self.assertRaisesRegex(ValueError, "every completed epoch"):
            self.decide([entry], metrics=[1, 2])

    def test_aborted_run_without_summary_is_an_error(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        self.decide([entry])
        (self.run / "summary.json").unlink()
        with patch.object(selection, "EXPECTED_GOLD", selection.digest(self.gold)), \
                patch.object(selection, "EXPECTED_BASELINE", selection.digest(self.base_bundle)):
            with self.assertRaisesRegex(ValueError, "no completed training summary"):
                selection.select(self.base_report, self.base_bundle, self.manifest,
                                 self.run, self.gold, self.evaluator)

    def test_changed_bundle_or_checkpoint_is_an_error(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        original = Path(entry["bundle"]).read_bytes()
        Path(entry["bundle"]).write_bytes(original + b"changed after evaluation")
        with self.assertRaisesRegex(ValueError, "bundle hash differs"):
            self.decide([entry])
        Path(entry["bundle"]).write_bytes(original)
        (self.run / "checkpoints/epoch-1.mpk").write_text("changed checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            self.decide([entry])

    def test_gate_or_export_metadata_cannot_point_to_other_weights(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        Path(entry["quantized"]).write_text("other weights")
        with self.assertRaisesRegex(ValueError, "quantization gate does not bind"):
            self.decide([entry])
        Path(entry["quantized"]).write_text("weights 1")
        header = json.dumps({"__metadata__": {"detector_best_sha256": "wrong"}}).encode()
        Path(entry["bundle"]).write_bytes(len(header).to_bytes(8, "little") + header)
        with self.assertRaisesRegex(ValueError, "exported bundle does not bind"):
            self.decide([entry])

    def test_staged_config_cannot_change_the_trained_model_contract(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        Path(entry["config"]).write_text("other config")
        gate = json.loads(Path(entry["gate"]).read_text())
        gate["config_sha256"] = selection.digest(entry["config"])
        Path(entry["gate"]).write_text(json.dumps(gate))
        with self.assertRaisesRegex(ValueError, "staged config or inputs differ"):
            self.decide([entry])

    def test_baseline_bundle_cannot_be_substituted(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        self.decide([entry])
        self.base_bundle.write_text("another baseline")
        with patch.object(selection, "EXPECTED_GOLD", selection.digest(self.gold)):
            with self.assertRaisesRegex(ValueError, "frozen V17 reference"):
                selection.select(self.base_report, self.base_bundle, self.manifest,
                                 self.run, self.gold, self.evaluator)

    def test_dev_v2_requires_explicit_pinned_gold_path(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.32)
        self.decide([entry])
        with self.assertRaisesRegex(ValueError, "pinned corrected gold path"):
            selection.select(self.base_report, self.base_bundle, self.manifest,
                             self.run, self.gold, self.evaluator, "dev-v2")

    def test_candidate_report_cannot_shrink_case_or_gold_support(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        candidate = json.loads(Path(entry["report"]).read_text())
        candidate["systems"][0]["cases"] = 1
        Path(entry["report"]).write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, "case count differs"):
            self.decide([entry])
        candidate["systems"][0]["cases"] = 50
        candidate["systems"][0]["by_slice"]["country"]["GE"]["address"]["gold"] = 1
        Path(entry["report"]).write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, "gold support"):
            self.decide([entry])

    def test_candidate_report_cannot_invent_exact_or_lenient_scores(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        candidate = json.loads(Path(entry["report"]).read_text())
        address = candidate["systems"][0]["by_slice"]["country"]["GE"]["address"]
        address["exact"]["f1"] = 1.0
        Path(entry["report"]).write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, "exact f1 arithmetic differs"):
            self.decide([entry])
        address["exact"]["f1"] = .4
        address["lenient_one_to_one"]["recall"] = float("nan")
        Path(entry["report"]).write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError, "invalid metric"):
            self.decide([entry])


if __name__ == "__main__":
    unittest.main()
