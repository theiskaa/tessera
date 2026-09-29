"""Focused checks for contact snippet leakage and byte slicing."""

import unittest

from build_contact_snippets import excluded_surface_pattern, paragraphs
from build_silver import normalize_surface


class ContactSnippetTests(unittest.TestCase):
    def test_dotted_evaluation_acronym_is_excluded_without_substring_matches(self):
        gold = [{"expected": [{"kind": "org", "text": "OMB"}]}]
        forbidden = excluded_surface_pattern(gold)
        self.assertIsNotNone(forbidden.search(normalize_surface("write to O.M.B. today")))
        self.assertIsNone(forbidden.search(normalize_surface("the COMBINE unit")))

    def test_paragraph_offsets_remain_byte_offsets_with_unicode(self):
        source = "ADDRESSES:\nJosé Street.\n\nCONTACT:\nAna Pérez.".encode()
        pieces = [source[start:end].decode() for start, end in paragraphs(source)]
        self.assertEqual(pieces, ["ADDRESSES:\nJosé Street.", "CONTACT:\nAna Pérez."])


if __name__ == "__main__":
    unittest.main()
