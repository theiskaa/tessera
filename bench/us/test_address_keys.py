"""Regression cases for evaluation building overlap checks."""

import unittest

from address_keys import address_keys, address_keys_in_text


class AddressKeysTest(unittest.TestCase):
    def test_abbreviated_street_and_suite_variants_share_a_building(self):
        training = "1200 New Jersey Ave. SE, Washington, DC 20590"
        evaluation = "1200 New Jersey Avenue SE, West Building, Room W12-140, Washington, DC 20590"
        self.assertTrue(address_keys(training) & address_keys(evaluation))

    def test_room_prefix_does_not_hide_the_street(self):
        training = "5600 Fishers Lane, Rockville, MD 20857"
        evaluation = "Room 13N82, 5600 Fishers Lane, Rockville, Maryland 20857"
        self.assertTrue(address_keys(training) & address_keys(evaluation))

    def test_different_streets_with_same_number_and_zip_stay_distinct(self):
        first = "1900 Good Hope Road, Washington, DC 20020"
        second = "1900 Anacostia Drive, Washington, DC 20020"
        self.assertFalse(address_keys(first) & address_keys(second))

    def test_same_street_and_zip_with_different_suites_matches(self):
        training = "1101 MERCANTILE LN, SUITE 220, LARGO, MD 20774"
        evaluation = "1101 Mercantile Lane\nSuite 210\nLargo, MD 20774"
        self.assertTrue(address_keys(training) & address_keys(evaluation))

    def test_letter_suffix_and_direction_do_not_hide_street(self):
        training = "308B E GOVERNMENT ST, BRANDON, MS 39042"
        evaluation = "308 East Government Street, Brandon, MS 39042"
        self.assertTrue(address_keys(training) & address_keys(evaluation))

    def test_each_street_uses_its_own_postcode_in_a_long_passage(self):
        passage = ("Council address: 7700 NE Ambassador Place, Portland, OR 97220. "
                   "Mail to 101 North Main Street, Rockford, IL 61128.")
        self.assertIn(("97220", "7700", "ambassador"), address_keys(passage))
        self.assertIn(("61128", "101", "main"), address_keys(passage))
        self.assertNotIn(("61128", "7700", "ambassador"), address_keys(passage))

    def test_street_without_postcode_does_not_borrow_the_next_address_postcode(self):
        passage = ("Old office: 7700 NE Ambassador Place, Portland, OR. "
                   "New office: 101 North Main Street, Rockford, IL 61128.")
        self.assertNotIn(("61128", "7700", "ambassador"), address_keys(passage))
        self.assertIn(("61128", "101", "main"), address_keys(passage))

    def test_repeated_address_still_matches_when_later_postcode_differs(self):
        evaluation = "7700 NE Ambassador Place, Suite 101, Portland, OR 97220-1384"
        passage = ("Council address: 7700 NE Ambassador Place, Suite 101, Portland, "
                   "OR 97220-1384. Meeting room: 500 State Street, Rockford, IL 61128.")
        self.assertTrue(address_keys(evaluation) & address_keys(passage))

    def test_numbered_street_keeps_its_number_as_the_first_word(self):
        address = "550 17th Street NW, Washington, DC 20429"
        self.assertIn(("20429", "550", "17"), address_keys(address))

    def test_numbered_street_inside_contact_prose_is_found(self):
        passage = ("Send comments to the 7(a) Loan Origination Division, "
                   "409 3rd Street, Washington, DC 20416.")
        evaluation = "409 3rd Street SW, Suite 6050, Washington, DC 20416"
        self.assertTrue(address_keys_in_text(passage) & address_keys(evaluation))

    def test_hyphenated_house_numbers_do_not_collapse_to_the_last_part(self):
        first = "60-23 Cooper Avenue, Queens, NY 11385"
        second = "68-23 Cooper Avenue, Queens, NY 11385"
        self.assertFalse(address_keys(first) & address_keys(second))
        self.assertIn(("11385", "68-23", "cooper"), address_keys(second))

    def test_hyphenated_house_number_survives_a_suite_variant(self):
        first = "68-23 Cooper Avenue\nApt Number 103\nQueens\nNY 11385"
        second = "Room 504, 68-23 Cooper Avenue, Queens, NY 11385"
        self.assertTrue(address_keys(first) & address_keys(second))

    def test_west_and_street_abbreviations_share_a_building(self):
        first = "211 Wst Oak Str, Louisville, KY 40203"
        second = "211 West Oak Street, Louisville, KY 40203"
        self.assertTrue(address_keys(first) & address_keys(second))


if __name__ == "__main__":
    unittest.main()
