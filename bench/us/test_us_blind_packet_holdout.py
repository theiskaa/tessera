"""Guard blind packet screening before reviewers spend time on a source."""

import unittest

from check_us_blind_packet_holdout import heldout_reasons


class BlindPacketHoldoutTests(unittest.TestCase):
    def email_case(self, text, expected=None):
        gold = [{
            "name": "reserved", "input": "FWS",
            "source_url": "https://example.gov/reserved",
            "expected": [{"kind": "org", "start": 0, "end": 3, "text": "FWS"}],
        }]
        packet = [{"name": "candidate", "input": text,
                   "source_url": "https://example.gov/candidate"}]
        if expected is not None:
            packet[0]["expected"] = expected
        return packet, gold

    def test_email_domain_is_not_an_organization_mention(self):
        packet, gold = self.email_case("Contact Stephanie_Hebert@fws.gov for help.")
        self.assertEqual({}, heldout_reasons(packet, gold))

    def test_same_page_with_different_group_names_still_blocks(self):
        packet, gold = self.email_case("A different excerpt without shared entities.")
        packet[0]["source_group"] = "new-collector-group"
        gold[0]["source_group"] = "old-collector-group"
        packet[0]["source_url"] = "http://www.example.gov:80/reserved/#contact"
        self.assertIn("source", heldout_reasons(packet, gold)["candidate"])

    def test_standalone_org_still_blocks_with_an_email(self):
        packet, gold = self.email_case("FWS: contact Stephanie_Hebert@fws.gov")
        self.assertIn("org referent", heldout_reasons(packet, gold)["candidate"])

    def test_url_and_invalid_email_surfaces_still_block(self):
        for text in ("https://FWS@example.gov/path", "FWS@", "FWS@bad..gov",
                     "FWS@example.gov2", "..FWS@example.gov", "/FWS@example.gov",
                     "https://example.gov/\u00a0FWS@example.gov", "https://fws.gov"):
            with self.subTest(text=text):
                packet, gold = self.email_case(text)
                self.assertIn("org referent", heldout_reasons(packet, gold)["candidate"])

    def test_identical_email_still_blocks(self):
        text = "Stephanie_Hebert@fws.gov"
        packet, gold = self.email_case(text)
        gold.append({"name": "reserved-email", "input": text,
                     "source_url": "https://example.gov/another-reserved",
                     "expected": [{"kind": "email", "start": 0,
                                   "end": len(text), "text": text}]})
        self.assertIn("email", heldout_reasons(packet, gold)["candidate"])

    def test_labeled_org_inside_email_is_not_exempt(self):
        packet, gold = self.email_case("FWS@example.gov", [
            {"kind": "org", "start": 0, "end": 3, "text": "FWS"}])
        self.assertIn("labeled org", heldout_reasons(packet, gold)["candidate"])

    def test_shared_prose_is_not_exempt_when_it_contains_an_email(self):
        text = "Contact FWS@example.gov for information about the next scheduled public meeting at our office"
        packet, gold = self.email_case(text)
        gold[0]["input"] = text
        gold[0]["expected"] = []
        self.assertIn("twelve-word prose", heldout_reasons(packet, gold)["candidate"])

    def test_unlabeled_alias_and_address_are_screened(self):
        gold = [{
            "name": "reserved", "input": "NTSB at 1301 Clay Street, Oakland, CA 94612",
            "source_url": "https://example.gov/reserved",
            "expected": [
                {"kind": "org", "start": 0, "end": 4, "text": "NTSB"},
                {"kind": "address", "start": 8, "end": 43,
                 "text": "1301 Clay Street, Oakland, CA 94612"},
            ],
        }]
        packet = [{
            "name": "candidate", "source_url": "https://example.gov/candidate",
            "input": ("National Transportation Safety Board, 1301 Clay St., "
                      "Suite 700N, Oakland, CA 94612"),
        }]
        hits = heldout_reasons(packet, gold)
        self.assertIn("org referent", hits["candidate"])
        self.assertIn("physical address", hits["candidate"])

    def test_labeled_hit_is_attached_to_packet_case(self):
        gold = [{
            "name": "reserved", "input": "NTSB",
            "source_url": "https://example.gov/reserved",
            "expected": [{"kind": "org", "start": 0, "end": 4, "text": "NTSB"}],
        }]
        text = "National Transportation Safety Board"
        packet = [{
            "name": "candidate", "source_url": "https://example.gov/candidate",
            "input": text,
            "expected": [{"kind": "org", "start": 0, "end": len(text), "text": text}],
        }]
        hits = heldout_reasons(packet, gold)
        self.assertNotIn("reserved", hits)
        self.assertIn("labeled org", hits["candidate"])

    def test_disjoint_packet_passes(self):
        gold = [{
            "name": "reserved", "input": "NTSB",
            "source_url": "https://example.gov/reserved",
            "expected": [{"kind": "org", "start": 0, "end": 4, "text": "NTSB"}],
        }]
        packet = [{
            "name": "candidate", "source_url": "https://example.gov/candidate",
            "input": "A different agency met at 23 Elm Road.",
        }]
        self.assertEqual({}, heldout_reasons(packet, gold))

    def test_two_letter_org_does_not_match_lowercase_prose(self):
        gold = [{
            "name": "reserved", "input": "IN",
            "source_url": "https://example.gov/reserved",
            "expected": [{"kind": "org", "start": 0, "end": 2, "text": "IN"}],
        }]
        packet = [{
            "name": "candidate", "source_url": "https://example.gov/candidate",
            "input": "A person works in another office.",
        }]
        self.assertEqual({}, heldout_reasons(packet, gold))
        packet[0]["input"] = "The unit IN another office."
        self.assertIn("org referent", heldout_reasons(packet, gold)["candidate"])


if __name__ == "__main__":
    unittest.main()
