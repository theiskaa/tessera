"""Check that new Federal Register examples stay clean and disjoint."""

import json
import unittest

from freeze_2025_contact_packet import (ARTIFACT, HOLDOUT, OUT, RAW, SNAPSHOT,
                                        digest, ordered_candidates, reserved_gold,
                                        selected_cases)
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from freeze_2025_year_contact_packet import OUT as YEAR_OUT


class ContactPacketTests(unittest.TestCase):
    def test_full_year_packet_covers_every_month_without_shared_eval_streets(self):
        rows = lines(YEAR_OUT)
        months = {json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                  for row in rows}
        self.assertEqual(months, {f"2025-{month:02d}" for month in range(1, 13)})
        excluded = {"2025-05441", "2025-09472", "2025-11815"}
        self.assertFalse({row["source_id"] for row in rows} & excluded)

    def test_balanced_selection_interleaves_months_without_changing_legacy_order(self):
        candidates = [
            (9, "a", 1, "url", "text", "2025-01"),
            (8, "b", 1, "url", "text", "2025-01"),
            (7, "c", 1, "url", "text", "2025-02"),
            (6, "d", 1, "url", "text", "2025-02"),
        ]
        self.assertEqual([row[1] for row in ordered_candidates(candidates, None)],
                         ["a", "b", "c", "d"])
        self.assertEqual([row[1] for row in ordered_candidates(candidates, 1)],
                         ["a", "c", "b", "d"])

    def test_reserved_labels_include_reviewed_names_and_addresses(self):
        gold = reserved_gold()
        text = {span["text"] for row in gold for span in row["expected"]}
        self.assertIn("Pacific Fishery Management Council", text)
        self.assertTrue(any("7700 NE Ambassador Place" in value for value in text))

    def test_frozen_cases_match_sources_and_avoid_reserved_labels(self):
        snapshot = json.loads(SNAPSHOT.read_text())
        rows = lines(OUT)
        self.assertEqual(len(rows), len({row["source_id"] for row in rows}))
        self.assertTrue(rows)
        forbidden = excluded_surface_pattern(reserved_gold())
        reserved_ids = {row["source_id"] for row in lines(HOLDOUT)}
        for row in rows:
            with self.subTest(row=row["name"]):
                source_id = row["source_id"]
                raw_bytes = (RAW / f"{source_id}.json").read_bytes()
                source = json.loads(raw_bytes)
                text = row["input"]
                self.assertEqual(digest(raw_bytes), snapshot[source_id])
                self.assertEqual(row["raw_sha256"], snapshot[source_id])
                self.assertIn(text, source["text"])
                self.assertTrue(text.startswith(("ADDRESSES:",
                                                 "FOR FURTHER INFORMATION CONTACT:")))
                self.assertTrue(text.rstrip().endswith((".", "!", "?")))
                self.assertIsNone(ARTIFACT.search(text))
                self.assertIsNone(forbidden.search(normalize_surface(text)))
                self.assertNotIn(source_id, reserved_ids)

    def test_packet_rebuild_is_identical(self):
        selected, _, _ = selected_cases()
        self.assertEqual(selected, lines(OUT))


if __name__ == "__main__":
    unittest.main()
