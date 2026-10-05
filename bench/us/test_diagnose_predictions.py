import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from diagnose_predictions import classify, load_predictions, validate_gold


class DiagnosisTests(unittest.TestCase):
    def test_duplicate_predictions_are_errors(self):
        gold = [{"kind": "person", "start": 0, "end": 4}]
        self.assertEqual(classify(gold * 2, gold), ["exact", "duplicate"])

    def test_boundary_wrong_kind_and_missing_are_distinct(self):
        gold = [{"kind": "org", "start": 0, "end": 4}]
        self.assertEqual(classify(gold, [{"kind": "org", "start": 0, "end": 3}]), ["boundary"])
        self.assertEqual(classify(gold, [{"kind": "person", "start": 0, "end": 4}]), ["wrong_kind"])
        self.assertEqual(classify(gold, []), ["no_overlap"])

    def test_gold_rejects_duplicates_and_bad_utf8_offsets(self):
        span = {"kind": "person", "start": 0, "end": 4, "text": "Jane"}
        row = {"name": "one", "input": "Jane", "expected": [span]}
        validate_gold([row])
        for invalid in ([row, row], [{**row, "expected": [span, span]}]):
            with self.assertRaises(ValueError):
                validate_gold(invalid)
        with self.assertRaises(UnicodeDecodeError):
            validate_gold([{**row, "input": "😀", "expected": [{**span, "end": 1}]}])

    def test_predictions_bind_the_exact_gold_and_all_case_names(self):
        gold = [{"name": "one", "input": "Jane", "expected": []}]
        data = b"fixed gold"
        record = {"provenance": {"gold_sha256": hashlib.sha256(data).hexdigest()}, "predictions": [{"name": "one", "entities": []}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.json"
            path.write_text(json.dumps(record))
            self.assertEqual(load_predictions(path, data, gold)[1], {"one": []})
            with self.assertRaises(ValueError):
                load_predictions(path, b"changed", gold)
            record["predictions"] *= 2
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_predictions(path, data, gold)


if __name__ == "__main__":
    unittest.main()
