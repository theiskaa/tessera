"""Keep ZIP-free street screening separate from driving instructions."""

import unittest

from screen_us_address_long_blind_v2 import street_keys


class StreetKeyTests(unittest.TestCase):
    def test_omitted_zip_does_not_hide_a_reserved_street(self):
        self.assertEqual(street_keys("26 Wall Street"),
                         street_keys("26 Wall St., New York, NY 10005"))
        self.assertEqual({("15", "pine")}, street_keys("15 Pine Street"))

    def test_direction_and_ordinal_are_part_of_the_street(self):
        self.assertIn(("600", "2"), street_keys(
            "600 North 2nd Street in Richmond, Virginia 23219"))
        self.assertIn(("111", "19"), street_keys("111 19th Street"))

    def test_named_street_keeps_its_number(self):
        self.assertEqual({("423", "john")}, street_keys(
            "423 John Wesley Dobbs Avenue NE Atlanta Georgia 30312"))

    def test_distance_and_exit_numbers_are_not_street_numbers(self):
        for text in ("Go 3 blocks to 2nd Street.", "3 blocks north on 2nd Street",
                     "2 miles north along Pine Street", "Take exit 22 to Callowhill Street"):
            with self.subTest(text=text):
                self.assertEqual(set(), street_keys(text))

    def test_room_phone_and_document_numbers_are_not_streets(self):
        self.assertEqual(set(), street_keys(
            "Room 400. Call 555-123-4567. Document 2026-12345. I-95 exit 22."))


if __name__ == "__main__":
    unittest.main()
