"""Check that source identifiers catch shared US documents across split formats."""

import unittest

from source_identity import document_key


class SourceIdentityTests(unittest.TestCase):
    def test_federal_register_url_and_case_name_share_document_key(self):
        source = {"source_url": "https://www.federalregister.gov/documents/2026/07/16/2026-14361/example",
                  "source_group": "unrelated"}
        gold = {"name": "2026-14361"}
        self.assertEqual(document_key(source), document_key(gold))

    def test_mapped_directory_group_catches_missing_gold_url(self):
        source = {"source_url": "https://www.nrcs.usda.gov/state-offices/maine/directory",
                  "source_group": "us-nrcs-maine-directory-v1"}
        gold = {"name": "us-nrcs-maine-office-bangor",
                "source_group": "us-nrcs-maine-directory-v1"}
        self.assertEqual(document_key(source), document_key(gold))


if __name__ == "__main__":
    unittest.main()
