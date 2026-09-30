"""Keep readiness counts tied to distinct sources and actual labeled addresses."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_us_next_data_split import verify_source_manifest
from next_data_coverage import TARGETS, coverage_checks, summarize


def sample(source_id, words=200, address_prefix=True):
    """Build a UTF-8 row with the optional suite either inside or outside its label."""
    beginning = "Émile: Suite 4, 22 Elm Road, Boston, MA 02111"
    text = beginning + " content" * (words - len(beginning.split()))
    address = ("Suite 4, " if address_prefix else "") + "22 Elm Road, Boston, MA 02111"
    start = text.encode().index(address.encode())
    return {"source_id": source_id, "text": text, "entities": [
        {"kind": "address", "start": start, "end": start + len(address.encode())}]}


class CoverageTests(unittest.TestCase):
    def test_multiple_windows_do_not_increase_source_coverage(self):
        summary = summarize([sample("2026-12345"), sample("2026-12345", 220)])
        self.assertEqual(2, summary["long_examples"])
        self.assertEqual(1, summary["long_source_documents"])
        self.assertEqual(1, summary["by_kind"]["address"]["long_source_documents"])

    def test_long_threshold_is_not_lowered_to_fill_quota(self):
        summary = summarize([sample("2026-12345", 199), sample("2026-12346", 200)])
        self.assertEqual(1, summary["long_source_documents"])

    def test_extra_address_parts_must_be_inside_utf8_label(self):
        summary = summarize([sample("2026-12345", address_prefix=False)])
        self.assertEqual(0, summary["long_prefixed_address_source_documents"])
        summary = summarize([sample("2026-12345", address_prefix=True)])
        self.assertEqual(1, summary["long_prefixed_address_source_documents"])

    def test_mail_stop_slash_counts_but_state_abbreviation_does_not(self):
        row = sample("2026-12345")
        row["text"] = row["text"].replace("Suite 4", "M/S 321-134")
        row["entities"][0]["end"] += len("M/S 321-134") - len("Suite 4")
        self.assertEqual(1, summarize([row])["long_prefixed_address_source_documents"])
        row = sample("2026-12345", address_prefix=False)
        row["text"] = row["text"].replace("MA 02111", "MS 39211")
        self.assertEqual(0, summarize([row])["long_prefixed_address_source_documents"])

    def test_federal_register_and_unknown_do_not_fill_other_website_quota(self):
        unknown = sample("office")
        unknown["source_group"] = "unknown-office"
        summary = summarize([sample("2026-12345"), unknown])
        checks = coverage_checks(summary, json.loads(TARGETS.read_text()))
        website = next(check for check in checks if "websites" in check["name"])
        self.assertEqual(0, website["current"])
        self.assertEqual(2, website["missing"])

    def test_numbered_stop_inside_address_is_a_mail_stop(self):
        row = sample("2026-12345")
        row["text"] = row["text"].replace("Suite 4", "Stop 9037")
        row["entities"][0]["end"] += len("Stop 9037") - len("Suite 4")
        self.assertEqual(1, summarize([row])["long_prefixed_address_source_documents"])
        row = sample("2026-12345", address_prefix=False)
        row["text"] += " Stop 9037 is on the next page."
        self.assertEqual(0, summarize([row])["long_prefixed_address_source_documents"])

    def test_missing_source_identity_is_not_counted_as_diversity(self):
        row = sample("unidentified")
        with self.assertRaisesRegex(ValueError, "source document identity"):
            summarize([row])

    def test_www_and_port_variants_do_not_inflate_website_coverage(self):
        first, second = sample("first"), sample("second")
        first["source_url"] = "https://www.example.gov/first"
        second["source_url"] = "https://example.gov:443/second"
        summary = summarize([first, second])
        checks = coverage_checks(summary, json.loads(TARGETS.read_text()))
        website = next(check for check in checks if "websites" in check["name"])
        self.assertEqual(1, website["current"])
        self.assertFalse(website["passed"])


class SourceManifestTests(unittest.TestCase):
    def test_historical_files_are_hash_checked_without_new_eligibility_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.jsonl"
            path.write_text("original\n")
            path.with_suffix(".manifest.json").write_text(json.dumps({
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            verify_source_manifest(path, require_eligible=False)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "reviewed silver changed"):
                verify_source_manifest(path, require_eligible=False)

    def test_new_file_requires_explicit_training_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.jsonl"
            path.write_text("original\n")
            path.with_suffix(".manifest.json").write_text(json.dumps({
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError, "not eligible"):
                verify_source_manifest(path, require_eligible=True)


if __name__ == "__main__":
    unittest.main()
