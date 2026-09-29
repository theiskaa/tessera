"""Check acronym prose boundaries, provenance, and frozen selection."""

import json
import unittest

from build_contact_snippets import lines
from build_silver import family
from freeze_2024_acronym_packet import (CONTACT_GOLDS, CONTACT_PACKETS,
                                        MANIFEST, NUMBERED_HEADING, OUT, RAW,
                                        SNAPSHOT, digest,
                                        expanded_bodies, prose_acronyms,
                                        selected_cases, sentence_slices,
                                        strict_expanded_bodies, used_source_ids)
from freeze_2025_year_contact_packet import PRIOR_2025_PACKETS


class AcronymPacketTests(unittest.TestCase):
    def test_strict_expansions_reject_nonorg_and_page_break_evidence(self):
        rejected = {"2025-02918": "HHA", "2025-05232": "BHC",
                    "2025-11783": "FEA", "2025-16443": "ESP",
                    "2025-07356": "MBS", "2025-16613": "TDA"}
        accepted = {"2025-05038": "HUD", "2025-07123": "TUVRNA",
                    "2025-23573": "ACHDNC"}
        for source_id, acronym in rejected.items():
            text = json.loads((RAW / f"{source_id}.json").read_text())["text"]
            self.assertNotIn(acronym, strict_expanded_bodies(text), source_id)
        for source_id, acronym in accepted.items():
            text = json.loads((RAW / f"{source_id}.json").read_text())["text"]
            self.assertIn(acronym, strict_expanded_bodies(text), source_id)
        self.assertIsNotNone(NUMBERED_HEADING.search(
            "The agency advises firms. 1. Electronic Submission of Comments."))

    def test_2025_use_screen_includes_prior_context_sources(self):
        ids, families, hashes = used_source_ids(2025, PRIOR_2025_PACKETS)
        context = lines(PRIOR_2025_PACKETS[0])
        self.assertTrue(context)
        self.assertIn("data/interim/silver/r24/us-2025-context-blind-v1.jsonl", hashes)
        for row in context:
            self.assertIn(row["source_id"], ids)
            self.assertIn(family(row["source_url"]), families)

    def test_sentence_slices_keep_abbreviated_us_name_intact(self):
        paragraph = ("The U.S. Department issued guidance. "
                     "EDA works with communities. Dr. Ada Smith joined.")
        slices = list(sentence_slices(paragraph))
        self.assertEqual([part for _, _, part in slices], [
            "The U.S. Department issued guidance.",
            "EDA works with communities.",
            "Dr. Ada Smith joined.",
        ])
        for start, end, sentence in slices:
            self.assertEqual(paragraph[start:end], sentence)

    def test_expansion_and_action_screen_avoid_non_body_acronyms(self):
        source = ("Economic Development Administration (EDA) works here. "
                  "Federal Advisory Committee Act (FACA) applies. "
                  "Federal Information Relay Service (FIRS) takes calls.")
        bodies = expanded_bodies(source)
        self.assertIn("EDA", bodies)
        self.assertNotIn("FACA", bodies)
        self.assertNotIn("FIRS", bodies)
        self.assertEqual(prose_acronyms("EDA works with communities.", bodies), ["EDA"])
        self.assertEqual(prose_acronyms("Docket EDA-2024-15 was filed.", bodies), [])

    def test_frozen_cases_are_source_exact_and_rebuild_identically(self):
        snapshot = json.loads(SNAPSHOT.read_text())
        frozen = lines(OUT)
        manifest = json.loads(MANIFEST.read_text())
        rebuilt, _, _, _ = selected_cases()
        self.assertEqual(rebuilt, frozen)
        self.assertEqual(manifest["sha256"], digest(OUT.read_bytes()))
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(len(frozen), len({row["source_id"] for row in frozen}))
        self.assertEqual(len(frozen), len({row["source_group"] for row in frozen}))
        for row in frozen:
            with self.subTest(source_id=row["source_id"]):
                raw_bytes = (RAW / f"{row['source_id']}.json").read_bytes()
                raw = json.loads(raw_bytes)
                text = raw["text"].encode()
                self.assertEqual(digest(raw_bytes), snapshot[row["source_id"]])
                self.assertEqual(row["raw_sha256"], snapshot[row["source_id"]])
                self.assertEqual(
                    text[row["source_byte_start"]:row["source_byte_end"]].decode(),
                    row["input"],
                )
                self.assertEqual(
                    text[row["expansion_byte_start"]:row["expansion_byte_end"]].decode(),
                    row["expansion_evidence"],
                )
                self.assertIn(f"({row['acronym']})", row["expansion_evidence"])
                self.assertTrue(row["input"].rstrip().endswith((".", "!", "?")))

    def test_v2_is_disjoint_and_v1_remains_immutable(self):
        frozen = lines(OUT)
        used_ids, used_families, _ = used_source_ids()
        contact = [row for path in (*CONTACT_PACKETS, *CONTACT_GOLDS)
                   for row in lines(path)]
        contact_ids = {row["source_id"] for row in contact}
        contact_families = {family(row["source_url"]) for row in contact}
        for row in frozen:
            with self.subTest(source_id=row["source_id"]):
                self.assertNotIn(row["source_id"], contact_ids)
                self.assertNotIn(row["source_group"], contact_families)
                self.assertNotIn(row["source_id"], used_ids)
                self.assertNotIn(row["source_group"], used_families)
        previous = OUT.with_name("us-2024-acronym-prose-blind-v1.jsonl")
        previous_manifest = json.loads(previous.with_suffix(".manifest.json").read_text())
        self.assertEqual(digest(previous.read_bytes()), previous_manifest["sha256"])
        self.assertFalse(previous_manifest["training_eligible"])


if __name__ == "__main__":
    unittest.main()
