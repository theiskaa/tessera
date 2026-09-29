"""Keep reviewed 2024 contact candidates tied to unused official sources."""

import json
import unittest

from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from freeze_2024_contact_packet import (NEW_HOLDOUT, OUT, RAW, SNAPSHOT,
                                        digest, excluded_source_ids, source_snapshot)
from freeze_2025_contact_packet import (ARTIFACT, GOLD, reserved_gold,
                                        selected_cases)
from freeze_2024_contact_review_packet import DECISIONS, OUT as REVIEW_PACKET


class ContactPacket2024Tests(unittest.TestCase):
    def test_review_packet_applies_all_triage_decisions(self):
        rows = lines(OUT)
        kept = lines(REVIEW_PACKET)
        dropped = json.loads(DECISIONS.read_text())["drop"]
        self.assertEqual([row for row in rows if row["source_id"] not in dropped], kept)
        self.assertEqual(len(kept), 60)

    def test_frozen_cases_are_source_distinct_and_rebuild(self):
        snapshot = source_snapshot()
        used = excluded_source_ids()
        extra = lines(NEW_HOLDOUT)
        rebuilt, _, _ = selected_cases(snapshot, year=2024,
                                       extra_gold=extra, excluded_sources=used)
        rows = lines(OUT)
        self.assertEqual(rebuilt, [row for row in rows
                                   if row["name"] != "us-2024-contact-2024-28058-1"])
        self.assertEqual(snapshot, json.loads(SNAPSHOT.read_text()))
        self.assertEqual(len(rows), len({row["source_id"] for row in rows}))
        forbidden = excluded_surface_pattern(
            [row for path in GOLD for row in lines(path)] + reserved_gold() + extra)
        for row in rows:
            with self.subTest(row=row["name"]):
                source_id = row["source_id"]
                raw_bytes = (RAW / f"{source_id}.json").read_bytes()
                source = json.loads(raw_bytes)
                text = row["input"]
                self.assertEqual(digest(raw_bytes), row["raw_sha256"])
                self.assertNotIn(source_id, used)
                self.assertTrue(source["date"].startswith("2024-"))
                self.assertIn(text, source["text"])
                self.assertTrue(text.startswith(("ADDRESSES:",
                                                 "FOR FURTHER INFORMATION CONTACT:")))
                self.assertTrue(text.rstrip().endswith((".", "!", "?")))
                self.assertIsNone(ARTIFACT.search(text))
                self.assertIsNone(forbidden.search(normalize_surface(text)))


if __name__ == "__main__":
    unittest.main()
