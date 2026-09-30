"""Check configured input integrity and evaluation separation without recipe imports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_data


class DataChecks(unittest.TestCase):
    def test_config_preserves_source_order_and_rejects_duplicate_paths(self):
        text = ('[detector]\nsilver = ["second.jsonl", "first.jsonl"]\n'
                '[generate]\nexclude_gold = "gold.jsonl"\n'
                '[data]\nprocessed = "processed"\nmanifests = "manifests"\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text)
            self.assertEqual(["second.jsonl", "first.jsonl"],
                             check_data.read_config(path)["silver"])
            path.write_text(text.replace('"first.jsonl"', '"./second.jsonl"'))
            with self.assertRaisesRegex(ValueError, "duplicate file paths"):
                check_data.read_config(path)

    def test_missing_or_repeated_config_fields_fail(self):
        for text in ('[data]\n', '[data]\nprocessed = "a"\nprocessed = "b"\n'):
            with self.assertRaisesRegex(ValueError, "expected one"):
                check_data.config_field(text, "data", "processed")

    def test_manifest_checks_local_recipe_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "ignored_recipe.py"
            recipe.write_text("original")
            path = root / "rows.jsonl"
            path.write_text(json.dumps({"country": "US"}) + "\n")
            manifest = {"sha256": check_data.digest(path), "documents": 1,
                        "input_sha256": {recipe.name: check_data.digest(recipe)}}
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            with patch.object(check_data, "ROOT", root):
                self.assertEqual(1, len(check_data.verified_rows(path)))
                recipe.write_text("changed")
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    check_data.verified_rows(path)

    def test_dataset_hash_and_explicit_ineligibility_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"country":"US"}\n')
            manifest_path = path.with_suffix(".manifest.json")
            manifest_path.write_text(json.dumps({"sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                check_data.verified_rows(path)
            manifest_path.write_text(json.dumps({"sha256": check_data.digest(path),
                                                 "training_eligible": False}))
            with self.assertRaisesRegex(ValueError, "eligibility"):
                check_data.verified_rows(path)
            self.assertEqual(1, len(check_data.verified_rows(path, evaluation=True)))

    def test_prose_check_includes_historical_rows(self):
        text = "One two three four five six seven eight nine ten eleven twelve"
        gold = [{"name": "reserved", "input": text.upper()}]
        real = [("historical:1", text.replace(" ", "\n") + " extra", [])]
        self.assertEqual({"historical:1": ["reserved"]},
                         check_data.prose_overlaps(real, gold))
        self.assertEqual({}, check_data.prose_overlaps([("short", "one two", [])], gold))

    def test_only_real_relay_contacts_are_exempt(self):
        for phone in ("711", "202-555-0142"):
            with self.subTest(phone=phone):
                span = {"kind": "phone", "text": phone, "start": 0, "end": len(phone)}
                gold = [{"name": "reserved", "input": phone, "expected": [span]}]
                sources = lambda: iter([("real", phone, [span])])
                self.assertTrue(check_data.overlap_checks(gold, sources)["contacts"])
                result = check_data.overlap_checks(gold, sources, allow_relay=True)
                self.assertEqual(phone != "711", bool(result["contacts"]))

    def test_url_overlap_cannot_hide_behind_different_source_groups(self):
        gold = {"name": "reserved", "input": "gold", "expected": [],
                "source_group": "evaluation", "source_url": "https://www.example.gov/doc/"}
        real = {"id": "train", "text": "different", "entities": [],
                "source_group": "training", "source_url": "http://example.gov/doc"}
        config = {"exclude_gold": "gold.jsonl", "silver": ["silver.jsonl"],
                  "processed": "."}
        with patch.object(check_data, "verified_rows", side_effect=[[gold], [real]]), \
                patch.object(check_data, "verify_synthetic"), \
                patch.object(check_data, "synthetic_sources", side_effect=lambda _: iter([])):
            result = check_data.audit(config)
        self.assertFalse(result["passed"])
        self.assertEqual(["silver.jsonl:train"], result["failures"]["source_or_url"])

    def test_all_real_prose_is_reported_and_optional_strict_gate_fails(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        gold = {"name": "reserved", "input": text, "expected": [], "source_group": "gold"}
        real = {"id": "train", "text": text, "entities": [], "source_group": "real"}
        config = {"exclude_gold": "gold.jsonl", "silver": ["silver.jsonl"], "processed": "."}
        for strict in (False, True):
            with patch.object(check_data, "verified_rows", side_effect=[[gold], [real]]), \
                    patch.object(check_data, "verify_synthetic"), \
                    patch.object(check_data, "synthetic_sources", side_effect=lambda _: iter([])):
                result = check_data.audit(config, strict_prose=strict)
            self.assertEqual(not strict, result["passed"])
            self.assertEqual({"silver.jsonl:train": ["reserved"]}, result["prose_check"]["overlaps"])

    def test_synthetic_manifest_requires_exact_membership_and_all_three_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("silver.jsonl", "gold.jsonl", "train.parquet", "valid.parquet", "test.parquet"):
                (root / name).write_text(name)
            config = {"silver": ["silver.jsonl"], "exclude_gold": "gold.jsonl",
                      "processed": ".", "manifests": "."}
            manifest = {"source": "tessera-generator", "exclude_gold": "gold.jsonl",
                        "exclude_gold_sha256": check_data.digest(root / "gold.jsonl"),
                        "real_silver_sha256": {"silver.jsonl": check_data.digest(root / "silver.jsonl")},
                        "synthetic_parquet_sha256": {
                            split: check_data.digest(root / f"{split}.parquet")
                            for split in ("train", "valid", "test")}}
            manifest_path = root / "detector-synthetic.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(check_data, "ROOT", root):
                check_data.verify_synthetic(config)
                (root / "test.parquet").write_text("changed")
                with self.assertRaisesRegex(ValueError, "test shard changed"):
                    check_data.verify_synthetic(config)
                manifest["real_silver_sha256"]["extra.jsonl"] = "extra"
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "configured silver"):
                    check_data.verify_synthetic(config)


if __name__ == "__main__":
    unittest.main()
