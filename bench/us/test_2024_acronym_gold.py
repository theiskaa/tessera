"""Keep adjudicated acronym prose separated from US evaluation and contact sources."""

import unittest

from build_2024_acronym_gold import OUT, gold_rows
from build_contact_snippets import lines
from build_silver import family
from check_holdout_overlap import collisions


class AcronymGold2024Tests(unittest.TestCase):
    def test_gold_is_reconciled_and_source_distinct(self):
        rows, adjudications = gold_rows()
        self.assertEqual(rows, lines(OUT))
        self.assertEqual(len(rows), 14)
        self.assertEqual(adjudications, 4)
        self.assertEqual(len({row["source_id"] for row in rows}), len(rows))
        self.assertEqual(len({row["source_group"] for row in rows}), len(rows))
        contacts = lines(OUT.parent / "us-2024-contact-gold-v3.jsonl")
        self.assertFalse({row["source_id"] for row in rows} &
                         {row["source_id"] for row in contacts})
        self.assertFalse({row["source_group"] for row in rows} &
                         {family(row["source_url"]) for row in contacts})

    def test_gold_has_no_labeled_evaluation_name_or_address(self):
        rows = lines(OUT)
        evaluation = []
        for filename in ("us-eval-exclusions-v1.jsonl",
                         "us-2025-contact-holdout-v2.jsonl",
                         "us-full-notice-gold-v1.jsonl"):
            evaluation.extend(lines(OUT.parent / filename))
        sources = [(row["name"], row["input"], row["expected"])
                   for row in evaluation]
        self.assertEqual(collisions(rows, sources), {})


if __name__ == "__main__":
    unittest.main()
