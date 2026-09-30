"""Check source-document exposure and fail-closed trainer-count attribution."""

import copy
import json
import unittest

from check_us_next_sampling import config_values, summarize
from next_data_coverage import TARGETS


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            ("first.jsonl", [
                {"country": "US", "source_url": "https://example.gov/house.pdf",
                 "text": "first excerpt"},
                {"country": "US", "source_url": "https://example.gov/other.pdf",
                 "text": "another excerpt"}]),
            ("second.jsonl", [
                {"country": "US", "source_url": "https://example.gov/house.pdf",
                 "text": "long " * 200}]),
        ]
        self.config = {"silver": [name for name, _ in self.sources],
                       "silver_repeats": [2, 3], "synthetic_per_epoch": 3,
                       "epochs": 3, "batch_size": 4}
        self.counts = {"documents": 3, "pieces": 3, "duplicate_documents": 0,
                       "unreachable_spans": 0, "source_pieces": [2, 1],
                       "source_content_tokens": [11, 200], "content_tokens": 211,
                       "by_country": {"US": {"documents": 3, "pieces": 3}}}
        self.targets = json.loads(TARGETS.read_text())

    def audit(self):
        return summarize(self.config, self.counts, self.sources, self.targets)

    def test_shared_pdf_across_files_is_one_source(self):
        result = self.audit()
        self.assertEqual(2, result["source_documents"])
        largest = result["largest_source_document"]
        self.assertEqual("url:https://example.gov/house.pdf", largest["key"])
        self.assertEqual(5, largest["draws_per_epoch"])
        self.assertEqual(15, largest["draws_all_epochs"])
        self.assertEqual(0.5, result["fractions_of_all_draws"]["maximum_source_document"])
        self.assertFalse(result["sampling_passed"])

    def test_exact_weighted_tokens_and_ceiling_updates(self):
        result = self.audit()
        self.assertEqual({"total": 10, "synthetic": 3, "real": 7, "long_real": 3},
                         result["draws_per_epoch"])
        self.assertEqual(622, result["real_content_token_draws_per_epoch"])
        self.assertEqual(1866, result["real_content_token_draws_all_epochs"])
        self.assertEqual(9, result["optimizer_updates"])
        self.assertEqual(9, result["by_file"]["second.jsonl"]["repeats_all_epochs"])

    def test_bad_counts_do_not_guess_document_attribution(self):
        changes = [
            {"duplicate_documents": 1}, {"unreachable_spans": 1},
            {"source_pieces": [1, 2]}, {"source_pieces": [2]},
            {"documents": 4}, {"pieces": 4}, {"content_tokens": 212},
            {"source_content_tokens": []}, {"source_content_tokens": [11, -1]},
            {"source_pieces": [2, True]}, {"by_country": {}},
        ]
        for change in changes:
            with self.subTest(change=change):
                counts = {**self.counts, **change}
                with self.assertRaises(ValueError):
                    summarize(self.config, counts, self.sources, self.targets)

    def test_exact_text_duplicates_cannot_hide_in_diagnostics(self):
        self.sources[1][1][0]["text"] = self.sources[0][1][0]["text"]
        with self.assertRaisesRegex(ValueError, "duplicate source text"):
            self.audit()

    def test_nonpositive_sampling_is_rejected(self):
        for value in (0, -1, True, 1.5):
            for field in ("epochs", "batch_size", "synthetic_per_epoch", "silver_repeats"):
                with self.subTest(field=field, value=value):
                    config = copy.deepcopy(self.config)
                    config[field] = [value, 3] if field == "silver_repeats" else value
                    with self.assertRaises(ValueError):
                        summarize(config, self.counts, self.sources, self.targets)

    def test_source_order_must_match(self):
        self.config["silver"].reverse()
        with self.assertRaisesRegex(ValueError, "paths differ"):
            self.audit()

    def test_known_toml_fields_and_scalar_repeat_fallback(self):
        text = ('[train]\nepochs = 3\nbatch_size = 32\n'
                '[detector]\nsilver = [\n"a.jsonl",\n"b.jsonl",\n]\n'
                'silver_repeat = 4\nsynthetic_per_epoch = 50\n')
        result = config_values(text)
        self.assertEqual([4, 4], result["silver_repeats"])
        for addition in ("silver_repeat = 2\n", "silver_repeats = [0, 1]\n",
                         "silver_repeats = [1]\n"):
            with self.subTest(addition=addition):
                with self.assertRaises(ValueError):
                    config_values(text + addition)


if __name__ == "__main__":
    unittest.main()
