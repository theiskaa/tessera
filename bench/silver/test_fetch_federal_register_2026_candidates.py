"""Regression checks for official pagination and interrupted listing capture."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_federal_register_2026_candidates as capture


def document(number):
    return {
        "document_number": number,
        "publication_date": "2026-01-15",
        "html_url": (f"https://www.federalregister.gov/documents/2026/01/15/"
                     f"{number}/example-notice"),
        "agencies": [],
    }


class ListingPaginationTests(unittest.TestCase):
    def test_special_documents_are_valid_but_excluded_from_source_candidates(self):
        correction = document("C1-2025-22604")
        capture.validate_document(correction, 1)
        revision = document("R1-2026-05038")
        capture.validate_document(revision, 1)
        novel = document("S2-2026-05039")
        capture.validate_document(novel, 1)
        with self.assertRaisesRegex(ValueError, "invalid notice"):
            capture.validate_document(document("../2026-05039"), 1)
        ordinary = document("2026-00001")
        ordinary["agencies"] = [{"slug": "test-agency"}]

        def save(row, out):
            (out / f"{row['document_number']}.json").write_text(json.dumps({
                "id": row["document_number"], "date": row["publication_date"],
                "url": row["html_url"], "text": "A" * 201,
            }))
            return True

        with tempfile.TemporaryDirectory() as temporary:
            listing = Path(temporary) / "listing"
            (listing / "2026-01").mkdir(parents=True)
            (listing / "2026-01" / "manifest.json").write_text("{}")
            with patch.object(capture, "LISTING", listing), patch.object(
                    capture, "SOURCES", Path(temporary) / "sources"), patch.object(
                    capture, "MONTHS", (1,)), patch.object(
                    capture, "PER_MONTH", 1), patch.object(
                    capture, "historical_sources", return_value=(set(), set(), {})), patch.object(
                    capture, "read_listing", return_value=(
                        [correction, revision, novel, ordinary], {"month": "2026-01"})), patch.object(
                    capture, "fetch", side_effect=save):
                rows, counts, _, _ = capture.selected_sources(allow_network=False)
            self.assertEqual([row["source_id"] for row in rows], ["2026-00001"])
            self.assertEqual(counts["special_notices"], 3)

    def test_resumes_frozen_page_and_accepts_official_extensionless_next_url(self):
        next_url = ("https://www.federalregister.gov/api/v1/documents?"
                    "conditions%5Bpublication_date%5D%5Bgte%5D=2026-01-01&"
                    "conditions%5Bpublication_date%5D%5Blte%5D=2026-01-31&"
                    "page=2&per_page=100&search_after_cursor=cursor")
        first = {"count": 2, "results": [document("2026-00001")],
                 "next_page_url": next_url}
        second = {"count": 2, "results": [document("2025-24290")],
                  "next_page_url": None}
        with tempfile.TemporaryDirectory() as temporary:
            listing = Path(temporary)
            month = listing / "2026-01"
            month.mkdir()
            (month / "page-001.json").write_text(json.dumps(first))
            with patch.object(capture, "LISTING", listing), patch.object(
                    capture, "get", return_value=json.dumps(second).encode()) as get:
                rows, manifest = capture.read_listing(1, allow_network=True)
                self.assertEqual([row["document_number"] for row in rows],
                                 ["2026-00001", "2025-24290"])
                self.assertEqual(manifest["documents"], 2)
                get.assert_called_once_with(next_url)
                replay, frozen = capture.read_listing(1, allow_network=False)
                self.assertEqual(replay, rows)
                self.assertEqual(frozen, manifest)

    def test_rejects_pagination_outside_official_endpoint(self):
        first = {"count": 2, "results": [document("2026-00001")],
                 "next_page_url": "https://example.com/api/v1/documents?page=2"}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(capture, "LISTING", Path(temporary)), patch.object(
                    capture, "get", return_value=json.dumps(first).encode()):
                with self.assertRaisesRegex(ValueError, "changed endpoint"):
                    capture.read_listing(1, allow_network=True)


if __name__ == "__main__":
    unittest.main()
