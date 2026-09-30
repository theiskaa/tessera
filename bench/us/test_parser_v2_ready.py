"""Regression cases for parser source split checks."""

import unittest

from check_parser_v2_ready import street_city_key


def row(house, road, city):
    text = f"{house} {road}\n{city}"
    return {
        "text": text,
        "span_label": [0, 1, 5],
        "span_start": [0, len(house) + 1, len(house) + len(road) + 2],
        "span_end": [len(house), len(house) + len(road) + 1, len(text)],
    }


class ParserV2ReadyTest(unittest.TestCase):
    def test_street_type_alias_is_detected_without_a_zip(self):
        self.assertEqual(street_city_key(row("2101", "Commons Drive", "Rogers")),
                         street_city_key(row("2101", "Commons Dve", "Rogers")))

    def test_numbered_streets_stay_distinct(self):
        self.assertNotEqual(street_city_key(row("3715", "Northeast 9th Avenue", "Portland")),
                            street_city_key(row("3715", "Northeast 115th Ave", "Portland")))


if __name__ == "__main__":
    unittest.main()
