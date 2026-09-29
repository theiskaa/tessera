"""Check frozen notice lineage and a name alias that exact matching misses."""

import collections
import hashlib
import json
import unittest

from build_silver import PAGE_MARKER
from freeze_full_notice_packet import (FROZEN_SHA256, MANIFEST, RAW,
                                       TEMPLATE_PICKS, digest,
                                       has_person_alias, person_alias_patterns,
                                       recurring_template, selected_cases)


class FullNoticePacketTests(unittest.TestCase):
    def test_frozen_cases_match_pinned_raw_sources_after_marker_cleanup(self):
        cases, _, _ = selected_cases()
        data = ("\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True)
                          for case in cases) + "\n").encode()
        self.assertEqual(len(cases), 23)
        self.assertEqual(digest(data), FROZEN_SHA256)
        self.assertEqual(len({case["source_id"] for case in cases}), len(cases))
        ids = {case["source_id"] for case in cases}
        self.assertTrue(TEMPLATE_PICKS["civil_rights_advisory"] <= ids)
        self.assertTrue(TEMPLATE_PICKS["air_force_record_of_decision"] <= ids)
        cleaned_markers = 0
        templates = collections.Counter()
        for case in cases:
            raw_bytes = (RAW / f"{case['source_id']}.json").read_bytes()
            raw = json.loads(raw_bytes)
            templates[recurring_template(raw)] += 1
            expected = PAGE_MARKER.sub(b"\n\n", raw["text"].encode()).decode()
            self.assertEqual(case["raw_sha256"], hashlib.sha256(raw_bytes).hexdigest())
            self.assertEqual(case["source_url"], raw["url"])
            self.assertEqual(case["input"], expected)
            self.assertNotIn("[[Page ", case["input"])
            cleaned_markers += expected != raw["text"]
        self.assertGreater(cleaned_markers, 0)
        self.assertEqual(templates["civil_rights_advisory"], 2)
        self.assertEqual(templates["air_force_record_of_decision"], 1)

    def test_person_alias_matches_middle_initial_and_surname_first(self):
        gold = [{"expected": [{"kind": "person", "text": "Anthony T. Lee"}]}]
        patterns = person_alias_patterns(gold)
        self.assertTrue(has_person_alias("Anthony Lee", patterns))
        self.assertTrue(has_person_alias("Lee, Anthony", patterns))
        self.assertFalse(has_person_alias("Anthony Reed", patterns))

    def test_packet_is_diagnostic_until_entity_overlap_is_removed(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["intended_use"],
                         "source_distinct_diagnostic_only")
        self.assertFalse(manifest["training_eligible"])
        self.assertFalse(manifest["strict_entity_holdout"])
        self.assertEqual(manifest["source_group_key"], "source_id")


if __name__ == "__main__":
    unittest.main()
