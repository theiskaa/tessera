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
        address_correct = int(round(50 * (ge_exact if country == "GE" else
                                          us_exact if country == "US" else .8)))
        address_found = int(round(50 * (ge_found if country == "GE" else .9)))
        countries[country] = {
            "person": metric(50, 50, 35 if country == "GB" else 45, 45),
            "org": metric(50, 50, 30, 35),
            "address": metric(50, 50, address_correct, address_found),
            "email": metric(50, 50, 49, 49),
            "phone": metric(50, 50, 48, 48),
        }
    kinds = {
        kind: metric(250, 250,
                     sum(countries[country][kind]["exact_tp"] for country in countries),
                     sum(countries[country][kind]["one_to_one_gold_hits"]
                         for country in countries))
        for kind in selection.KINDS
    }
    overall = metric(1250, 1250, sum(row["exact_tp"] for row in kinds.values()),
                     sum(row["one_to_one_gold_hits"] for row in kinds.values()))
    return {"country_hint_mode": "auto", "provenance": {
        "gold_sha256": selection.digest(gold),
        "evaluator_sha256": selection.digest(evaluator),
        "bundle_sha256": selection.digest(bundle),
    }, "systems": [{"system": "tessera", "cases": 250, "overall": overall,
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
                        "expected": [{"kind": kind} for kind in selection.KINDS]}) + "\n"
            for country in selection.COUNTRIES for number in range(50)))
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

    @staticmethod
    def change_exact(report_data, country, kind, exact_tp):
        system = report_data["systems"][0]
        current = system["by_slice"]["country"][country][kind]["exact_tp"]
        delta = exact_tp - current
        for row in (system["by_slice"]["country"][country][kind],
                    system["by_kind"][kind], system["overall"]):
            row.update(metric(row["gold"], row["predicted"], row["exact_tp"] + delta,
                              row["one_to_one_gold_hits"]))

    @staticmethod
    def change_found(report_data, country, kind, found):
        system = report_data["systems"][0]
        current = system["by_slice"]["country"][country][kind]["one_to_one_gold_hits"]
        delta = found - current
        for row in (system["by_slice"]["country"][country][kind],
                    system["by_kind"][kind], system["overall"]):
            row.update(metric(row["gold"], row["predicted"], row["exact_tp"],
                              row["one_to_one_gold_hits"] + delta))

    @staticmethod
    def remove_one_gold(report_data, country, kind):
        system = report_data["systems"][0]
        for row in (system["by_slice"]["country"][country][kind],
                    system["by_kind"][kind], system["overall"]):
            row.update(metric(row["gold"] - 1, row["predicted"] - 1,
                              row["exact_tp"] - 1, row["one_to_one_gold_hits"] - 1))

    def rewrite_epoch_exact(self, entry, country, kind, exact_tp):
        path = Path(entry["report"])
        candidate = json.loads(path.read_text())
        self.change_exact(candidate, country, kind, exact_tp)
        path.write_text(json.dumps(candidate))

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
        self.assertTrue(any("US.address_exact_f1" in reason
                            for reason in decision["epochs"][0]["reasons"]))

    def test_rejects_ge_organization_collapse_with_address_gain(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        self.rewrite_epoch_exact(entry, "GE", "org", 0)
        decision = self.decide([entry])
        self.assertIsNone(decision["selected_epoch"])
        self.assertFalse(decision["epochs"][0]["manual_review_required"])
        self.assertTrue(any("GE.org_exact_f1" in reason
                            for reason in decision["epochs"][0]["reasons"]))

    def test_rejects_gb_person_collapse_below_target(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        self.rewrite_epoch_exact(entry, "GB", "person", 0)
        decision = self.decide([entry])
        self.assertIsNone(decision["selected_epoch"])
        self.assertTrue(any("GB.person_exact_f1" in reason
                            for reason in decision["epochs"][0]["reasons"]))

    def test_rejects_email_and_phone_regressions(self):
        email = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        phone = self.add_epoch(2, ge_found=.6, ge_exact=.4)
        self.rewrite_epoch_exact(email, "US", "email", 0)
        self.rewrite_epoch_exact(phone, "JP", "phone", 0)
        decision = self.decide([email, phone])
        self.assertIsNone(decision["selected_epoch"])
        self.assertTrue(any("US.email_exact_f1" in reason
                            for reason in decision["epochs"][0]["reasons"]))
        self.assertTrue(any("JP.phone_exact_f1" in reason
                            for reason in decision["epochs"][1]["reasons"]))

    def test_rejects_address_found_regression_below_target(self):
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        path = Path(entry["report"])
        candidate = json.loads(path.read_text())
        self.change_found(candidate, "DE", "address", 40)
        path.write_text(json.dumps(candidate))
        decision = self.decide([entry])
        self.assertIsNone(decision["selected_epoch"])
        self.assertTrue(any("DE.address_found_recall" in reason
                            for reason in decision["epochs"][0]["reasons"]))

    def test_low_support_requires_manual_review(self):
        rows = [json.loads(line) for line in self.gold.read_text().splitlines()]
        phone = rows[-1]["expected"]
        rows[-1]["expected"] = [span for span in phone if span["kind"] != "phone"]
        self.gold.write_text("".join(json.dumps(row) + "\n" for row in rows))
        baseline = json.loads(self.base_report.read_text())
        baseline["provenance"]["gold_sha256"] = selection.digest(self.gold)
        self.remove_one_gold(baseline, "JP", "phone")
        self.base_report.write_text(json.dumps(baseline))
        entry = self.add_epoch(1, ge_found=.6, ge_exact=.4)
        path = Path(entry["report"])
        candidate = json.loads(path.read_text())
        self.remove_one_gold(candidate, "JP", "phone")
        path.write_text(json.dumps(candidate))
        decision = self.decide([entry])
        self.assertIsNone(decision["selected_epoch"])
        self.assertTrue(decision["epochs"][0]["manual_review_required"])
        self.assertTrue(any("JP.phone_exact_f1 has 49 gold spans" in reason
                            for reason in decision["epochs"][0]["reasons"]))

    def test_target_scores_use_checked_counts_despite_report_rounding(self):
        baseline = json.loads(self.base_report.read_text())
        address = baseline["systems"][0]["by_slice"]["country"]["GE"]["address"]
        address["exact"]["f1"] = .30004
        address["lenient_one_to_one"]["recall"] = .50004
        support = selection.gold_support(self.gold)
        selection.verify_support(baseline, support, "baseline")
        scores = selection.report_scores(baseline, "baseline")
        self.assertEqual(scores["GE.address_exact_f1"], .3)
        self.assertEqual(scores["GE.address_found_recall"], .5)

    def test_regression_scores_cover_country_and_global_overall(self):
        baseline = json.loads(self.base_report.read_text())
        scores = selection.regression_scores(baseline)
        self.assertEqual(len(scores), 41)
        self.assertEqual(scores["GE.overall_exact_f1"][1], 250)
        self.assertEqual(scores["all.overall_exact_f1"][1], 1250)
        self.assertEqual(scores["GB.person_exact_f1"][0], .7)
        self.assertEqual(scores["US.email_exact_f1"][0], .98)

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
        candidate["systems"][0]["cases"] = 250
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
