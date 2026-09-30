"""Keep strict name splits without equating generic offices across agencies."""

import unittest

from org_aliases import active_aliases


class OrgAliasTests(unittest.TestCase):
    def test_official_aliases_share_an_evaluation_family(self):
        gold = [{"expected": [
            {"kind": "org", "text": "Commerce"},
            {"kind": "org", "text": "Department of Homeland Security"},
            {"kind": "org", "text": "National Marine Fisheries Service"},
        ]}]
        aliases = active_aliases(gold, strict=True)
        self.assertIn("department of commerce", aliases)
        self.assertIn("doc", aliases)
        self.assertIn("dhs", aliases)
        self.assertIn("noaa fisheries", aliases)
        self.assertIn("national oceanic and atmospheric administration fisheries", aliases)

    def test_strict_aliases_cover_heldout_referents_without_changing_legacy(self):
        gold = [{"expected": [{"kind": "org", "text": name} for name in
                              ("FEMA", "HHS", "OPM", "NTSB", "OUII", "DOE")]}]
        strict = active_aliases(gold, strict=True)
        for name in ("federal emergency management agency",
                     "department of health and human services",
                     "u.s. office of personnel management",
                     "national transportation safety board",
                     "office of unfair import investigations",
                     "department of energy"):
            self.assertIn(name, strict)
        self.assertNotIn("department of energy", active_aliases(gold))

    def test_generic_office_names_do_not_share_a_family(self):
        gold = [{"expected": [
            {"kind": "org", "text": "Office of the General Counsel"},
        ]}]
        self.assertNotIn("ogc", active_aliases(gold))

    def test_roster_expands_real_training_acronyms_for_split_check(self):
        silver = [{"expected": [
            {"kind": "org", "text": "NCI"},
            {"kind": "org", "text": "MFH"},
            {"kind": "org", "text": "PHMSA"},
        ]}]
        aliases = active_aliases(silver, include_roster=True)
        for name in ("national cancer institute",
                     "office of multifamily housing programs",
                     "pipeline and hazardous materials safety administration"):
            self.assertIn(name, aliases)

    def test_historical_packet_aliases_do_not_change(self):
        silver = [{"expected": [{"kind": "org", "text": "NCI"}]}]
        self.assertNotIn("national cancer institute", active_aliases(silver))
        self.assertIn("national cancer institute",
                      active_aliases(silver, include_roster=True))

    def test_evaluation_only_bodies_expand_during_readiness(self):
        gold = [{"expected": [{"kind": "org", "text": name} for name in
                              ("SBA", "FEMA", "FAA", "FCC", "BLM",
                               "U.S. Merchant Marine Academy",
                               "U.S. Army Corps of Engineers")]}]
        aliases = active_aliases(gold, include_roster=True)
        for name in ("small business administration",
                     "federal emergency management agency",
                     "federal aviation administration",
                     "federal communications commission",
                     "bureau of land management", "usmma",
                     "us army corps. of engineers"):
            self.assertIn(name, aliases)
        self.assertNotIn("small business administration", active_aliases(gold))

    def test_strict_evaluation_acronyms_and_punctuation_variants(self):
        gold = [{"expected": [
            {"kind": "org", "text": "SSA"},
            {"kind": "org", "text": "SEC"},
            {"kind": "org", "text": "U.S. Department of Justice"},
            {"kind": "org", "text": "State, Private, and Tribal Forestry"},
            {"kind": "org", "text": "Office of Regulatory Affairs and Collaborative Action--Indian Affairs"},
            {"kind": "org", "text": "International Trade Administration"},
            {"kind": "org", "text": "Administrative Office of the U.S. Courts"},
        ]}]
        aliases = active_aliases(gold)
        for value in ("social security administration", "securities and exchange commission",
                      "united states department of justice",
                      "department of justice", "state, private and tribal forestry",
                      "office of regulatory affairs and collaborative action"):
            self.assertIn(value, aliases)
        self.assertIn("ita", aliases)
        self.assertIn("aousc", aliases)


if __name__ == "__main__":
    unittest.main()
