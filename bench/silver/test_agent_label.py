"""Failure-preservation tests for the private silver export gate."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json
import os
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
import agent_label  # noqa: E402


class CutTests(TestCase):
    def test_long_page_without_late_line_break_keeps_full_limit(self):
        limit = agent_label.MAX_CHARS
        self.assertEqual(agent_label.cut("x" * (limit + 1)), "x" * limit)
        self.assertEqual(agent_label.cut("x\n" + "y" * limit),
                         "x\n" + "y" * (limit - 2))


class ExportTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.out = self.directory / "train.jsonl"
        self.doc = {"id": "doc", "source": "official", "country": "GB", "text": "Jane Smith, Acme Ltd"}

    def export(self, labels, problems=None, docs=None):
        with patch.object(agent_label, "root", return_value=self.directory), \
             patch.object(agent_label, "load", return_value=[self.doc] if docs is None else docs), \
             patch.object(agent_label, "read_pass", return_value=(labels, problems or [])), \
             redirect_stdout(StringIO()):
            agent_label.export(1, 2)

    def test_clean_export_writes_reviewed_spans(self):
        self.export({"doc": [{"kind": "person", "text": "Jane Smith"}, {"kind": "org", "text": "Acme Ltd"}]})
        row = json.loads(self.out.read_text())
        self.assertEqual([item["kind"] for item in row["entities"]], ["person", "org"])

    def test_missing_document_preserves_old_output(self):
        self.out.write_text("old output\n")
        with self.assertRaisesRegex(ValueError, "no labels"):
            self.export({})
        self.assertEqual(self.out.read_text(), "old output\n")

    def test_missing_string_preserves_old_output(self):
        self.out.write_text("old output\n")
        with self.assertRaisesRegex(ValueError, "missing string"):
            self.export({"doc": [{"kind": "person", "text": "John Doe"}]})
        self.assertEqual(self.out.read_text(), "old output\n")

    def test_repeated_short_cjk_needs_context(self):
        doc = self.doc | {"text": "東京と東京"}
        with self.assertRaisesRegex(ValueError, "ambiguous short CJK"):
            self.export({"doc": [{"kind": "person", "text": "東京"}]}, docs=[doc])

    def test_bad_kind_refuses_export(self):
        with self.assertRaisesRegex(ValueError, "bad kind"):
            self.export({"doc": []}, problems=["doc: bad kind"])

    def test_cross_kind_overlapping_labels_refuse_export(self):
        doc = self.doc | {"text": "New York"}
        with self.assertRaisesRegex(ValueError, "overlapping labels"):
            self.export({"doc": [{"kind": "org", "text": "New York"}, {"kind": "address", "text": "York"}]}, docs=[doc])

    def test_same_kind_contained_surface_keeps_longer_span(self):
        doc = self.doc | {"text": "New York"}
        self.export({"doc": [{"kind": "org", "text": "New York"}, {"kind": "org", "text": "York"}]}, docs=[doc])
        row = json.loads(self.out.read_text())
        self.assertEqual(row["entities"], [{"kind": "org", "start": 0, "end": 8}])

    def test_explicit_exclusion_is_counted(self):
        (self.directory / "export-exclusions.json").write_text(json.dumps([{"id": "doc", "reason": "source terms pending review"}]))
        self.export({})
        self.assertEqual(self.out.read_text(), "")


class GoldTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.out = self.directory / "data/interim/review/gold-r2.jsonl"
        self.out.parent.mkdir(parents=True)
        self.doc = {"id": "doc", "source": "official", "country": "GB",
                    "text": "Jane Smith, Acme Ltd"}

    def gold(self, labels, problems=None, docs=None):
        old_cwd = Path.cwd()
        try:
            os.chdir(self.directory)
            with patch.object(agent_label, "load", return_value=[self.doc] if docs is None else docs), \
                 patch.object(agent_label, "read_pass", return_value=(labels, problems or [])), \
                 redirect_stdout(StringIO()):
                agent_label.gold(2, 1)
        finally:
            os.chdir(old_cwd)

    def test_clean_gold_contains_exact_spans(self):
        self.gold({"doc": [{"kind": "person", "text": "Jane Smith"},
                           {"kind": "org", "text": "Acme Ltd"}]})
        row = json.loads(self.out.read_text())
        self.assertEqual([(e["kind"], e["text"]) for e in row["expected"]],
                         [("person", "Jane Smith"), ("org", "Acme Ltd")])

    def test_missing_label_preserves_old_gold(self):
        self.out.write_text("old gold\n")
        with self.assertRaisesRegex(ValueError, "has no labels"):
            self.gold({})
        self.assertEqual(self.out.read_text(), "old gold\n")

    def test_cross_kind_overlap_preserves_old_gold(self):
        self.out.write_text("old gold\n")
        with self.assertRaisesRegex(ValueError, "overlapping labels"):
            self.gold({"doc": [{"kind": "person", "text": "Jane Smith"},
                               {"kind": "org", "text": "Smith"}]})
        self.assertEqual(self.out.read_text(), "old gold\n")

    def test_labels_for_unsampled_document_refuse_gold(self):
        with self.assertRaisesRegex(ValueError, "unsampled document"):
            self.gold({"doc": [], "other": []})

    def test_phone_after_printed_label_is_reachable(self):
        for value, number in (("Tel-599 85 71 70", "599 85 71 70"),
                              ("TEL.03-3561-1111", "03-3561-1111"),
                              ("Fax: 020 1234 5678", "020 1234 5678")):
            with self.subTest(value=value):
                doc = self.doc | {"text": value}
                self.gold({"doc": [{"kind": "phone", "text": number}]}, docs=[doc])
                self.assertEqual(json.loads(self.out.read_text())["expected"][0]["text"], number)

    def test_number_inside_identifier_remains_glued(self):
        doc = self.doc | {"text": "NRC-2025-1234"}
        with self.assertRaisesRegex(ValueError, "unreachable label"):
            self.gold({"doc": [{"kind": "phone", "text": "2025-1234"}]}, docs=[doc])


class SampleTests(TestCase):
    def test_short_noncontact_pool_fills_from_remaining_contact_pages(self):
        docs = [{"id": i, "text": f"Call 202-555-{i:04d}"} for i in range(20)]
        docs += [{"id": 20, "text": "A public announcement"}]
        chosen = agent_label.draw_contact_sample(docs, 12, random.Random(3))
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len({d["id"] for d in chosen}), 12)
        self.assertEqual(sum("Call" in d["text"] for d in chosen), 11)
