"""Keep the adjudicated contact labels separate from US evaluation."""

import collections
import unittest

from build_2024_contact_gold import OUT, gold_rows
from build_contact_snippets import lines
from check_holdout_overlap import collisions
from freeze_2024_contact_review_packet import OUT as PACKET


class ContactGold2024Tests(unittest.TestCase):
    def test_gold_is_complete_and_reconciled(self):
        rows, disputes = gold_rows()
        self.assertEqual(rows, lines(OUT))
        self.assertEqual(len(rows), len(lines(PACKET)))
        self.assertEqual(len({row["source_id"] for row in rows}), len(rows))
        self.assertEqual(disputes, 14)
        counts = collections.Counter(span["kind"] for row in rows
                                     for span in row["expected"])
        self.assertTrue(all(counts[kind] > 0 for kind in
                            ("person", "org", "address", "email", "phone")))

    def test_gold_identifies_the_known_evaluation_alias_overlap(self):
        rows = lines(OUT)
        evaluation = []
        for filename in ("us-eval-exclusions-v1.jsonl",
                         "us-2025-contact-holdout-v2.jsonl",
                         "us-full-notice-gold-v1.jsonl"):
            evaluation.extend(lines(OUT.parent / filename))
        sources = [(row["name"], row["input"], row["expected"])
                   for row in evaluation]
        hits = collisions(rows, sources)
        self.assertEqual(set(hits), {"us-2024-contact-2024-28058-1"})
        self.assertEqual(set(hits["us-2024-contact-2024-28058-1"]), {"org"})
        safe = [row for row in rows
                if row["name"] != "us-2024-contact-2024-28058-1"]
        self.assertEqual(collisions(safe, sources), {})


if __name__ == "__main__":
    unittest.main()
