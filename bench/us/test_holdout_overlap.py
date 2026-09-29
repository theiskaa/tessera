"""Check entity aliases and physical addresses in the held-out split gate."""

import unittest

from check_holdout_overlap import collisions


class HoldoutOverlapTests(unittest.TestCase):
    def test_org_alias_is_attributed_only_to_its_matching_case(self):
        rows = [
            {"name": "fda", "expected": [
                {"kind": "org", "text": "Food and Drug Administration"}]},
            {"name": "school", "expected": [
                {"kind": "org", "text": "Example School"}]},
        ]
        hits = collisions(rows, [("train", "FDA", [
            {"kind": "org", "start": 0, "end": 3}])])
        self.assertEqual(set(hits), {"fda"})

    def test_person_alias_and_physical_address_are_detected(self):
        rows = [{"name": "case", "expected": [
            {"kind": "person", "text": "Jane M. Doe"},
            {"kind": "address", "text": "123 Main Street, Denver, CO 80202"},
        ]}]
        text = "Doe, Jane at 123 Main St, Denver, CO 80202"
        end = len(text.encode())
        hits = collisions(rows, [("train", text, [
            {"kind": "person", "start": 0, "end": 9},
            {"kind": "address", "start": 13, "end": end},
        ])])
        self.assertEqual(set(hits["case"]), {"person", "address"})


if __name__ == "__main__":
    unittest.main()
