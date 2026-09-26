"""Tiny private fixtures for the source/label/export contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_export_gate as gate
from source_profiles import PROFILES, SourceProfile


class ExportGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw_dir = self.root / "data/raw/silver/unit"
        raw_dir.mkdir(parents=True)
        self.raw_path = raw_dir / "one.json"
        self.text = "Alice works at Acme."
        self.raw = {"source": "unit", "id": "one", "country": "US", "text": self.text,
                    "url": "https://example.org/one"}
        self.raw_path.write_text(json.dumps(self.raw))
        self.capture_path = self.root / "capture.html"
        self.capture_path.write_text('<div class="s-richtext js-richtext"><p>Alice works at Acme.</p></div>')
        self.profile_id = "unit_press_v1"
        profile = SourceProfile(
            self.profile_id, "unit", "one", "US", self.raw["url"],
            gate.digest(self.raw_path.read_bytes()), "capture.html",
            gate.digest(self.capture_path.read_bytes()), gate.digest(self.text.encode()),
            "de_sh_press_v1")
        profile_patch = patch.dict(PROFILES, {self.profile_id: profile})
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        self.candidate_path = self.root / "candidate.jsonl"
        self.candidate = {"source": "unit", "id": "one", "country": "US", "text": self.text,
                          "entities": []}
        self.lineage_path = self.root / "lineage.jsonl"
        self.qualification_path = self.root / "qualification.jsonl"
        self.labels_path = self.root / "labels.jsonl"
        self.source_proofs_path = self.root / "source-proofs.jsonl"
        self.eval_path = self.root / "evaluation.json"
        self.selection_path = self.root / "selection.json"
        self.gold_path = self.root / "gold.jsonl"
        self.index_path = self.root / "publisher-index.jsonl"
        self.dev_gold_path = self.root / "development-gold.jsonl"
        self.eval_hosts_path = self.root / "development-hosts.json"
        self.gold_path.write_bytes(gate.canonical({"id": "gold-1", "text": "Other Agency"}) + b"\n")
        self.dev_gold = {"name": "dev-1", "input": "Other development page", "country": "US",
                         "source_host": "US|other.example"}
        self.index = {"id": "gold-1", "publisher_group": "unit:other-agency",
                      "publisher_groups": ["unit:other-agency"]}
        self.lineage = {"candidate_line": 1, "source": "unit", "id": "one", "country": "US",
                        "candidate_text_sha256": gate.digest(self.text.encode()),
                        "raw_path": "data/raw/silver/unit/one.json",
                        "raw_file_sha256": gate.digest(self.raw_path.read_bytes()),
                        "raw_text_sha256": gate.digest(self.text.encode()),
                        "raw_url": self.raw["url"], "url_host": "example.org",
                        "raw_excerpt_start_byte": 0,
                        "raw_excerpt_end_byte": len(self.text.encode()),
                        "source_eligibility": "pending"}
        self.qualification = {"source": "unit", "id": "one",
                              "candidate_text_sha256": self.lineage["candidate_text_sha256"],
                              "raw_file_sha256": self.lineage["raw_file_sha256"],
                              "raw_url": self.raw["url"], "source_eligibility": "approved",
                              "publisher_group": "unit:example-agency",
                              "publisher_groups": ["unit:example-agency"],
                              "publisher_evidence_url": "https://example.org/about",
                              "qualification_evidence_url": "https://example.org/reuse",
                              "rights_scope": "full page reuse reviewed",
                              "reviewer": "fixture reviewer", "reviewed_at": "2026-09-26",
                              "decision_version": "fixture-v1"}
        self.source_proof = {"source": "unit", "id": "one",
                             "candidate_text_sha256": self.lineage["candidate_text_sha256"],
                             "raw_file_sha256": self.lineage["raw_file_sha256"],
                             "raw_url": self.raw["url"], "profile_id": self.profile_id,
                             "capture_sha256": profile.capture_sha256}
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.spans = [{"kind": "person", "start": 0, "end": 5},
                      {"kind": "org", "start": 15, "end": 19}]
        self.labels = {"source": "unit", "id": "one",
                       "candidate_text_sha256": self.lineage["candidate_text_sha256"],
                       "label_version": "fixture-v1", "reviewer": "fixture labeler",
                       "review_evidence": "two independent full-document reviews",
                       "entities": self.spans,
                       "entities_sha256": gate.digest(gate.canonical(self.spans))}
        self.evaluation = {"status": "sealed", "version": "fixture-v1",
                           "reviewer": "fixture reviewer", "reviewed_at": "2026-09-26",
                           "gold_path": str(self.gold_path),
                           "gold_sha256": gate.digest(self.gold_path.read_bytes()),
                           "publisher_index_path": str(self.index_path),
                           "publisher_index_sha256": None}
        self.selection = {"version": "fixture-v1", "reviewer": "fixture reviewer",
                          "reason": "tiny pilot", "candidate_sha256": None,
                          "lineage_sha256": None,
                          "selected": [{"source": "unit", "id": "one",
                                        "candidate_text_sha256": self.lineage["candidate_text_sha256"]}]}
        self.write()

    def write(self):
        self.candidate_path.write_bytes(gate.canonical(self.candidate) + b"\n")
        self.lineage_path.write_bytes(gate.canonical(self.lineage) + b"\n")
        self.lineage_path.with_suffix(".manifest.json").write_text(json.dumps({
            "candidate_sha256": gate.digest(self.candidate_path.read_bytes()),
            "lineage_sha256": gate.digest(self.lineage_path.read_bytes()), "rows": 1}))
        self.selection["candidate_sha256"] = gate.digest(self.candidate_path.read_bytes())
        self.selection["lineage_sha256"] = gate.digest(self.lineage_path.read_bytes())
        self.selection_path.write_text(json.dumps(self.selection))
        self.qualification_path.write_bytes(gate.canonical(self.qualification) + b"\n")
        self.source_proofs_path.write_bytes(gate.canonical(self.source_proof) + b"\n")
        self.labels_path.write_bytes(gate.canonical(self.labels) + b"\n")
        self.index_path.write_bytes(gate.canonical(self.index) + b"\n")
        self.evaluation["publisher_index_sha256"] = gate.digest(self.index_path.read_bytes())
        self.eval_path.write_text(json.dumps(self.evaluation))
        self.dev_gold_path.write_bytes(gate.canonical(self.dev_gold) + b"\n")
        self.trusted_dev_sha256 = gate.digest(self.dev_gold_path.read_bytes())
        self.eval_hosts_path.write_text(json.dumps({
            "status": "sealed", "version": "fixture-v1", "reviewer": "fixture reviewer",
            "reviewed_at": "2026-09-27", "gold_path": str(self.dev_gold_path),
            "gold_sha256": gate.digest(self.dev_gold_path.read_bytes()), "rows": 1}))

    def check(self, qualification=True, labels=True, selection=False):
        with patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256):
            return gate.validate_inputs(
                self.candidate_path, self.lineage_path,
                self.qualification_path if qualification else None,
                self.labels_path if labels else None, self.eval_path, self.root,
                self.selection_path if selection else None, self.eval_hosts_path,
                self.source_proofs_path)

    def test_approved_row_requires_allowlisted_replayed_source(self):
        self.source_proofs_path.write_bytes(b"")
        with patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path)
        self.assertTrue(any("missing typed source proof" in e for e in report["errors"]))
        self.write()
        self.source_proof["profile_id"] = "untrusted_script_v1"
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.write()
        _, report = self.check()
        self.assertTrue(any("not allowlisted" in e for e in report["errors"]))

    def test_malformed_profile_id_fails_with_report(self):
        self.source_proof["profile_id"] = []
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.write()
        _, report = self.check()
        self.assertTrue(any("not allowlisted" in e for e in report["errors"]))

    def test_capture_tamper_and_self_hashed_receipt_fail(self):
        self.capture_path.write_text(self.capture_path.read_text() + " ")
        _, report = self.check()
        self.assertTrue(any("capture SHA mismatch" in e for e in report["errors"]))
        self.source_proof["capture_sha256"] = gate.digest(self.capture_path.read_bytes())
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.write()
        _, report = self.check()
        self.assertTrue(any("not allowlisted" in e for e in report["errors"]))

    def test_duplicate_proof_and_unbound_qualification_fail(self):
        self.source_proofs_path.write_bytes(self.source_proofs_path.read_bytes() * 2)
        _, report = self.check()
        self.assertTrue(any("duplicate source/id" in e for e in report["errors"]))
        self.write()
        self.qualification["source_proof_sha256"] = "0" * 64
        self.write()
        _, report = self.check()
        self.assertTrue(any("qualification binding mismatch" in e for e in report["errors"]))

    def test_development_host_overlap_fails_even_when_qualifications_pass(self):
        self.dev_gold["source_host"] = "US|www.example.org"
        self.write()
        _, report = self.check()
        self.assertTrue(any("development host overlap" in e for e in report["errors"]))

    def test_development_gold_hash_and_exact_text_are_checked(self):
        self.dev_gold["input"] = self.text
        self.write()
        _, report = self.check()
        self.assertTrue(any("exact text overlaps development gold" in e for e in report["errors"]))
        self.dev_gold_path.write_bytes(self.dev_gold_path.read_bytes() + b" ")
        _, report = self.check()
        self.assertTrue(any("development host gold SHA mismatch" in e for e in report["errors"]))

    def test_missing_development_host_manifest_fails(self):
        _, report = gate.validate_inputs(self.candidate_path, self.lineage_path,
                                         self.qualification_path, self.labels_path,
                                         self.eval_path, self.root)
        self.assertTrue(any("missing sealed development host manifest" in e
                            for e in report["errors"]))

    def test_substituted_empty_development_gold_fails(self):
        self.dev_gold_path.write_bytes(b"")
        manifest = json.loads(self.eval_hosts_path.read_text())
        manifest["gold_sha256"] = gate.digest(b"")
        manifest["rows"] = 0
        self.eval_hosts_path.write_text(json.dumps(manifest))
        _, report = self.check()
        self.assertTrue(any("identity differs from pinned gold" in e
                            for e in report["errors"]))

    def test_invalid_candidate_host_fails(self):
        self.raw["url"] = "https://localhost/one"
        self.raw_path.write_text(json.dumps(self.raw))
        self.lineage["raw_url"] = self.raw["url"]
        self.lineage["url_host"] = "localhost"
        self.lineage["raw_file_sha256"] = gate.digest(self.raw_path.read_bytes())
        self.qualification["raw_url"] = self.raw["url"]
        self.qualification["raw_file_sha256"] = self.lineage["raw_file_sha256"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("invalid candidate URL host" in e for e in report["errors"]))

    def test_complete_fixture_passes_and_trainer_preflight_passes(self):
        output, report = self.check()
        self.assertEqual(report["errors"], [])
        self.assertEqual(output[0]["entities"], self.spans)
        binary = gate.ROOT / "target/release/trainer"
        if binary.exists():
            config = gate.ROOT / "internal/bench/silver/detector-v20-ge-contacts.toml"
            data, counts = gate.preflight(output, config, binary)
            self.assertEqual(counts["documents"], 1)
            self.assertEqual(counts["unreachable_spans"], 0)
            self.assertEqual(len(data.splitlines()), 1)

    def test_missing_or_held_decision_fails(self):
        _, report = self.check(qualification=False)
        self.assertEqual(report["counts"]["missing_qualification"], 1)
        self.qualification["source_eligibility"] = "held"
        self.write()
        self.source_proofs_path.write_bytes(b"")
        with patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path)
        self.assertFalse(any("source proof" in e for e in report["errors"]))
        _, report = self.check()
        self.assertEqual(report["counts"]["pending_or_held"], 1)

    def test_raw_tampering_and_excerpt_mismatch_fail(self):
        self.raw_path.write_text(self.raw_path.read_text() + " ")
        _, report = self.check()
        self.assertTrue(any("raw file SHA mismatch" in e for e in report["errors"]))
        self.raw_path.write_text(json.dumps(self.raw))
        self.lineage["raw_excerpt_start_byte"] = 1
        self.write()
        _, report = self.check()
        self.assertTrue(any("raw identity/text/excerpt/URL mismatch" in e for e in report["errors"]))

    def test_missing_labels_and_publisher_overlap_fail(self):
        _, report = self.check(labels=False)
        self.assertEqual(report["counts"]["missing_reviewed_labels"], 1)
        self.index["publisher_group"] = "unit:example-agency"
        self.index["publisher_groups"] = ["unit:example-agency"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("evaluation publisher overlap" in e for e in report["errors"]))

    def test_missing_evidence_and_stale_label_hash_fail(self):
        self.qualification["qualification_evidence_url"] = None
        self.labels["entities_sha256"] = "0" * 64
        self.write()
        _, report = self.check()
        self.assertTrue(any("qualification_evidence_url" in e for e in report["errors"]))
        self.assertTrue(any("reviewed entity SHA mismatch" in e for e in report["errors"]))

    def test_sealed_eval_gold_and_index_hashes_are_checked(self):
        self.gold_path.write_bytes(self.gold_path.read_bytes() + b" ")
        _, report = self.check()
        self.assertTrue(any("sealed evaluation gold SHA mismatch" in e for e in report["errors"]))
        self.gold_path.write_bytes(gate.canonical({"id": "gold-1", "text": "Other Agency"}) + b"\n")
        self.index_path.write_bytes(self.index_path.read_bytes() + b" ")
        _, report = self.check()
        self.assertTrue(any("sealed evaluation publisher index SHA mismatch" in e for e in report["errors"]))

    def test_duplicate_sealed_eval_text_fails(self):
        duplicate = {"id": "gold-2", "text": "Other Agency"}
        self.gold_path.write_bytes(self.gold_path.read_bytes() + gate.canonical(duplicate) + b"\n")
        self.index_path.write_bytes(self.index_path.read_bytes() + gate.canonical({
            "id": "gold-2", "publisher_group": "unit:other",
            "publisher_groups": ["unit:other"]}) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.evaluation["publisher_index_sha256"] = gate.digest(self.index_path.read_bytes())
        self.eval_path.write_text(json.dumps(self.evaluation))
        _, report = self.check()
        self.assertTrue(any("duplicate exact text" in e for e in report["errors"]))

    def test_trainer_gold_name_input_schema_passes(self):
        self.gold_path.write_bytes(gate.canonical({"name": "gold-1", "input": "Other Agency",
                                                   "expected": [], "country": "GE",
                                                   "doc_type": "bounded_office_contact_excerpt"}) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.write()
        output, report = self.check()
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(output), 1)

    def test_mixed_gold_aliases_must_agree(self):
        row = {"name": "gold-1", "input": "Other Agency", "id": "gold-1",
               "text": "Different Agency", "expected": [], "country": "GE", "doc_type": "page"}
        self.gold_path.write_bytes(gate.canonical(row) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.write()
        _, report = self.check()
        self.assertTrue(any("gold aliases disagree" in e for e in report["errors"]))

    def test_partial_native_gold_pair_fails(self):
        self.gold_path.write_bytes(gate.canonical({"name": "gold-1", "text": "Other Agency"}) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.write()
        _, report = self.check()
        self.assertTrue(any("gold row missing name/input" in e for e in report["errors"]))

    def test_native_gold_exact_text_overlap_fails(self):
        self.gold_path.write_bytes(gate.canonical({"name": "gold-1", "input": self.text,
                                                   "expected": [], "country": "US",
                                                   "doc_type": "page"}) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.write()
        _, report = self.check()
        self.assertTrue(any("exact text overlaps sealed evaluation gold" in e for e in report["errors"]))

    def test_selection_scopes_out_unselected_rows(self):
        second = {"source": "unit", "id": "two", "country": "US",
                  "text": "Bob works elsewhere.", "entities": []}
        second_lineage = dict(self.lineage, id="two", candidate_line=2,
                              candidate_text_sha256=gate.digest(second["text"].encode()))
        self.candidate_path.write_bytes(self.candidate_path.read_bytes() + gate.canonical(second) + b"\n")
        self.lineage_path.write_bytes(self.lineage_path.read_bytes() + gate.canonical(second_lineage) + b"\n")
        self.lineage_path.with_suffix(".manifest.json").write_text(json.dumps({
            "candidate_sha256": gate.digest(self.candidate_path.read_bytes()),
            "lineage_sha256": gate.digest(self.lineage_path.read_bytes()), "rows": 2}))
        self.selection["candidate_sha256"] = gate.digest(self.candidate_path.read_bytes())
        self.selection["lineage_sha256"] = gate.digest(self.lineage_path.read_bytes())
        self.selection_path.write_text(json.dumps(self.selection))
        output, report = self.check(selection=True)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["counts"]["out_of_scope"], 1)
        self.assertEqual(len(output), 1)
        _, no_selection = self.check()
        self.assertEqual(no_selection["counts"]["missing_qualification"], 1)

    def test_selection_tamper_and_missing_selected_decision_fail(self):
        self.selection["selected"][0]["candidate_text_sha256"] = "0" * 64
        self.selection_path.write_text(json.dumps(self.selection))
        _, report = self.check(selection=True)
        self.assertTrue(any("selection text SHA mismatch" in e for e in report["errors"]))
        self.selection["selected"][0]["candidate_text_sha256"] = self.lineage["candidate_text_sha256"]
        self.selection["candidate_sha256"] = "0" * 64
        self.selection_path.write_text(json.dumps(self.selection))
        _, report = self.check(selection=True)
        self.assertTrue(any("selection candidate/lineage SHA mismatch" in e for e in report["errors"]))
        self.write()
        _, report = self.check(qualification=False, selection=True)
        self.assertEqual(report["counts"]["missing_qualification"], 1)

    def test_exact_train_eval_text_overlap_fails(self):
        self.gold_path.write_bytes(gate.canonical({"id": "gold-1", "text": self.text}) + b"\n")
        self.evaluation["gold_sha256"] = gate.digest(self.gold_path.read_bytes())
        self.write()
        _, report = self.check(selection=True)
        self.assertTrue(any("exact text overlaps sealed evaluation gold" in e for e in report["errors"]))

    def test_secondary_linked_publisher_overlap_fails(self):
        self.qualification["publisher_groups"].append("unit:shared")
        self.index["publisher_groups"].append("unit:shared")
        self.write()
        _, report = self.check(selection=True)
        self.assertTrue(any("evaluation publisher overlap" in e for e in report["errors"]))

    def test_missing_or_noncanonical_publisher_ids_fail(self):
        self.assertTrue(gate.publisher_ids(["fr:agency:209", "govuk:7cd6bf12-bbe9-4118-8523-f927b0442156"]))
        self.qualification["publisher_groups"] = []
        self.write()
        _, report = self.check(selection=True)
        self.assertTrue(any("linked publisher_groups" in e for e in report["errors"]))
        self.qualification["publisher_groups"] = ["unit:Example Agency"]
        self.write()
        _, report = self.check(selection=True)
        self.assertTrue(any("linked publisher_groups" in e for e in report["errors"]))

    def test_primary_publisher_must_be_in_linked_set(self):
        self.qualification["publisher_group"] = "unit:hidden-primary"
        self.write()
        _, report = self.check()
        self.assertTrue(any("primary publisher_group absent" in e for e in report["errors"]))

    def test_sealed_eval_primary_publisher_must_be_in_linked_set(self):
        self.index["publisher_group"] = "unit:hidden-primary"
        self.write()
        _, report = self.check()
        self.assertTrue(any("publisher index incomplete or invalid" in e for e in report["errors"]))

    def test_duplicate_exact_text_fails(self):
        second = dict(self.candidate, id="two")
        second_lineage = dict(self.lineage, id="two", candidate_line=2)
        self.candidate_path.write_bytes(gate.canonical(self.candidate) + b"\n" + gate.canonical(second) + b"\n")
        self.lineage_path.write_bytes(gate.canonical(self.lineage) + b"\n" + gate.canonical(second_lineage) + b"\n")
        self.lineage_path.with_suffix(".manifest.json").write_text(json.dumps({
            "candidate_sha256": gate.digest(self.candidate_path.read_bytes()),
            "lineage_sha256": gate.digest(self.lineage_path.read_bytes()), "rows": 2}))
        _, report = self.check()
        self.assertTrue(any("duplicate exact text" in e for e in report["errors"]))

    def test_unreachable_semantic_positive_rejected_by_trainer(self):
        binary = gate.ROOT / "target/release/trainer"
        if not binary.exists():
            self.skipTest("trainer binary unavailable")
        output = [{"source": "unit", "id": "one", "country": "US",
                   "text": "Acme\n\nOffice", "entities": [{"kind": "org", "start": 0, "end": 12}]}]
        config = gate.ROOT / "internal/bench/silver/detector-v20-ge-contacts.toml"
        with self.assertRaisesRegex(ValueError, "unreachable"):
            gate.preflight(output, config, binary)


if __name__ == "__main__":
    unittest.main()
