"""Keep source capture from cutting notices in the middle of text."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fetch_federal_register import extract, fetch, main, pages


class FederalRegisterCaptureTests(unittest.TestCase):
    def test_supplementary_information_is_not_cut_at_legacy_limit(self):
        body = "Full sentence. " * 180
        text = "ADDRESSES:\n123 Main Street.\n\nSUPPLEMENTARY INFORMATION:\n" + body
        captured = extract(text)
        self.assertIn(body.strip(), captured)
        self.assertTrue(captured.endswith("Full sentence."))

    def test_month_query_covers_the_entire_leap_month(self):
        with patch("fetch_federal_register.get", return_value=b'{"results": [{"document_number": "one", "publication_date": "2024-02-29"}]}') as get:
            self.assertEqual(list(pages(1, 2024, 2))[0]["document_number"], "one")
        query = parse_qs(urlparse(get.call_args.args[0]).query)
        self.assertEqual(query["conditions[publication_date][gte]"], ["2024-02-01"])
        self.assertEqual(query["conditions[publication_date][lte]"], ["2024-02-29"])
        self.assertNotIn("conditions[publication_date][year]", query)

    def test_capture_does_not_overshoot_requested_count(self):
        docs = [{"document_number": str(number), "publication_date": "2025-01-01"}
                for number in range(30)]
        with TemporaryDirectory() as directory:
            with patch("fetch_federal_register.pages", return_value=iter(docs)):
                with patch("fetch_federal_register.fetch", return_value=True) as fetch:
                    main(3, 2025, Path(directory))
            self.assertEqual(fetch.call_count, 3)

    def test_month_query_rejects_documents_outside_the_month(self):
        wrong = b'{"results": [{"document_number": "wrong", "publication_date": "2024-03-01"}]}'
        with patch("fetch_federal_register.get", return_value=wrong):
            with self.assertRaises(ValueError):
                list(pages(1, 2024, 2))

    def test_interrupted_capture_does_not_leave_a_partial_source(self):
        doc = {"document_number": "2025-00001", "publication_date": "2025-01-02",
               "html_url": "https://www.federalregister.gov/documents/2025/01/02/2025-00001/example"}
        with TemporaryDirectory() as directory:
            out = Path(directory)
            with patch("fetch_federal_register.get", return_value=b"unused"):
                with patch("fetch_federal_register.page_text",
                           return_value="ADDRESSES:\n" + "Full sentence. " * 20):
                    with patch("fetch_federal_register.time.sleep"):
                        with patch("fetch_federal_register.os.replace",
                                   side_effect=OSError("interrupted")):
                            with self.assertRaises(OSError):
                                fetch(doc, out)
            self.assertFalse((out / "2025-00001.json").exists())
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
