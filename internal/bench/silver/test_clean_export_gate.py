"""Tiny private fixtures for the source/label/export contract."""

import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_export_gate as gate
from source_profiles import PROFILES, SourceProfile


def publisher_record(issuers, parents=(), origins=(), families=(), subjects=(),
                     governance=()):
    return {"roles": {"issuers": list(issuers), "controlling_parents": list(parents),
                      "origin_publishers": list(origins), "source_families": list(families),
                      "subjects": list(subjects), "governance": list(governance)},
            "completeness": {role: "complete" for role in
                             ("issuers", "controlling_parents", "origin_publishers",
                              "source_families")},
            "reviewers": ["reviewer a", "reviewer b"], "evidence": "fixture page credits"}


class ExportGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw_dir = self.root / "data/raw/silver/unit"
        raw_dir.mkdir(parents=True)
        self.raw_path = raw_dir / "one.json"
        self.text = "Alice works at Acme. " + " ".join(
            f"Section {i} records a separate office contact procedure."
            for i in range(5))
        self.raw = {"source": "unit", "id": "one", "country": "US", "text": self.text,
                    "url": "https://example.org/one"}
        self.raw_path.write_text(json.dumps(self.raw))
        self.capture_path = self.root / "capture.html"
        self.capture_path.write_text(
            f'<div class="s-richtext js-richtext"><p>{self.text}</p></div>')
        self.profile_id = "unit_press_v1"
        profile = SourceProfile(
            self.profile_id, "unit", "one", "US", self.raw["url"],
            gate.digest(self.raw_path.read_bytes()), "capture.html",
            gate.digest(self.capture_path.read_bytes()), gate.digest(self.text.encode()),
            "de_sh_press_v1", publisher_group="unit:example-agency",
            required_publisher_groups=("unit:example-agency",))
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
        self.dev_publishers_path = self.root / "development-publishers.json"
        self.dev_index_path = self.root / "development-publisher-index.jsonl"
        self.dev_source_index_path = self.root / "development-source-index.jsonl"
        self.registry_path = self.root / "publisher-registry.json"
        self.registry = {"status": "sealed", "entities": [
            {"id": "unit:example-agency", "kind": "institution", "aliases": []},
            {"id": "unit:other-agency", "kind": "institution", "aliases": []},
            {"id": "unit:parent", "kind": "institution", "aliases": []},
            {"id": "unit:shared", "kind": "institution", "aliases": []},
            {"id": "unit:example-host", "kind": "source_family", "aliases": []},
            {"id": "unit:other-host", "kind": "source_family", "aliases": []}]}
        self.gold_path.write_bytes(gate.canonical({"id": "gold-1", "text": "Other Agency"}) + b"\n")
        self.dev_gold = {"name": "dev-1", "input": "Other development page", "country": "US",
                         "source_host": "US|other.example"}
        self.dev_raw_path = self.root / "data/raw/review/unit/dev-1.json"
        self.dev_raw_path.parent.mkdir(parents=True)
        self.dev_raw_url = "https://other.example/dev-1"
        self.dev_publisher = publisher_record(["unit:other-agency"],
                                               families=["unit:other-host"])
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
        self.qualification["publisher"] = publisher_record(
            ["unit:example-agency"], families=["unit:example-host"])
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
        eval_gold_patch = patch.object(gate, "EVAL_GOLD_SHA256",
                                       gate.digest(self.gold_path.read_bytes()))
        eval_index_patch = patch.object(gate, "EVAL_PUBLISHER_INDEX_SHA256",
                                        gate.digest(self.index_path.read_bytes()))
        eval_gold_patch.start()
        eval_index_patch.start()
        self.addCleanup(eval_gold_patch.stop)
        self.addCleanup(eval_index_patch.stop)

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
        self.dev_raw_path.write_text(json.dumps({
            "id": "dev-1", "source": "unit-dev", "country": "US",
            "text": self.dev_gold["input"], "url": self.dev_raw_url}))
        source = {"id": "dev-1", "source": "unit-dev", "country": "US",
                  "gold_text_sha256": gate.digest(self.dev_gold["input"].encode()),
                  "raw_path": "data/raw/review/unit/dev-1.json",
                  "raw_url": self.dev_raw_url,
                  "raw_file_sha256": gate.digest(self.dev_raw_path.read_bytes()),
                  "source_host": "other.example"}
        self.dev_source_index_path.write_bytes(gate.canonical(source) + b"\n")
        self.dev_index_path.write_bytes(gate.canonical({
            "id": "dev-1", "text_sha256": source["gold_text_sha256"],
            "raw_path": source["raw_path"], "raw_url": source["raw_url"],
            "raw_file_sha256": source["raw_file_sha256"],
            "publisher": self.dev_publisher}) + b"\n")
        self.registry_path.write_text(json.dumps(self.registry))
        self.dev_publishers_path.write_text(json.dumps({
            "status": "sealed", "version": "fixture-v1", "reviewer": "fixture reviewer",
            "reviewed_at": "2026-09-27", "gold_sha256": self.trusted_dev_sha256,
            "index_path": str(self.dev_index_path),
            "index_sha256": gate.digest(self.dev_index_path.read_bytes()),
            "source_index_path": str(self.dev_source_index_path),
            "source_index_sha256": gate.digest(self.dev_source_index_path.read_bytes()),
            "registry_path": str(self.registry_path),
            "registry_sha256": gate.digest(self.registry_path.read_bytes()), "rows": 1,
            "source_family_policy": "record_only"}))

    def publisher_pins(self, manifest_sha=None, source_index_sha=None):
        return {
            "manifest": manifest_sha or gate.digest(self.dev_publishers_path.read_bytes()),
            "index": gate.digest(self.dev_index_path.read_bytes()),
            "registry": gate.digest(self.registry_path.read_bytes()),
            "source_index": source_index_sha or gate.digest(self.dev_source_index_path.read_bytes()),
        }

    def check(self, qualification=True, labels=True, selection=False):
        with (patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256),
              patch.dict(gate.DEV_PUBLISHER_PINS_BY_GOLD, {
                  self.trusted_dev_sha256: self.publisher_pins()})):
            return gate.validate_inputs(
                self.candidate_path, self.lineage_path,
                self.qualification_path if qualification else None,
                self.labels_path if labels else None, self.eval_path, self.root,
                self.selection_path if selection else None, self.eval_hosts_path,
                self.source_proofs_path, self.dev_publishers_path)

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

    def test_approved_row_requires_profile_publisher_and_parent(self):
        profile = replace(PROFILES[self.profile_id],
                          publisher_group="unit:example-agency",
                          required_publisher_groups=("unit:example-agency", "unit:parent"))
        with patch.dict(PROFILES, {self.profile_id: profile}):
            _, report = self.check()
            self.assertTrue(any("omits source-profile publisher or parent" in e
                                for e in report["errors"]))
            self.qualification["publisher_groups"].append("unit:parent")
            self.write()
            _, report = self.check()
            self.assertFalse(any("omits source-profile publisher or parent" in e
                                 for e in report["errors"]))
            self.qualification["publisher_group"] = "unit:parent"
            self.write()
            _, report = self.check()
            self.assertTrue(any("omits source-profile publisher or parent" in e
                                for e in report["errors"]))

    def test_approved_row_rejects_profile_without_publisher_pin(self):
        profile = replace(PROFILES[self.profile_id], publisher_group="",
                          required_publisher_groups=())
        with patch.dict(PROFILES, {self.profile_id: profile}):
            _, report = self.check()
            self.assertTrue(any("source profile lacks a pinned publisher chain" in e
                                for e in report["errors"]))

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

    def test_normalized_development_copy_fails_without_gold_entities(self):
        self.dev_gold["input"] = "  " + self.text.upper().replace(" ", "  \n") + "  "
        self.dev_gold["expected"] = []
        self.write()
        _, report = self.check()
        self.assertTrue(any("near-copy overlaps development gold dev-1 (normalized whole text)" in e
                            for e in report["errors"]))
        self.dev_gold_path.write_bytes(self.dev_gold_path.read_bytes() + b" ")
        _, report = self.check()
        self.assertTrue(any("development host gold SHA mismatch" in e for e in report["errors"]))
        self.assertEqual(gate.normalized_near_text(" CAFE\u0301 \n"),
                         gate.normalized_near_text("café"))

    def test_character_near_copy_catches_edits_and_excerpts(self):
        gold = "Alice works at Acme. " + " ".join(
            f"Section {i} records a distinct contact procedure for office {i}."
            for i in range(18))
        normalized = gate.normalized_near_text(gold)
        pages = [("dev-long", normalized, gate.character_shingles(normalized))]
        edited = gold.replace("Alice", "Alicx", 1)
        self.assertIsNotNone(gate.development_near_copy(edited, pages))
        self.assertIsNotNone(gate.development_near_copy(gold[100:700], pages))
        self.assertIsNone(gate.development_near_copy(gold[100:220], pages))

    def test_cross_country_edited_copy_fails(self):
        self.dev_gold["country"] = "JP"
        self.dev_gold["source_host"] = "JP|other.example"
        self.dev_gold["input"] = self.text.replace("Alice", "Alicx", 1)
        self.write()
        _, report = self.check()
        self.assertTrue(any("near-copy overlaps development gold dev-1" in e
                            for e in report["errors"]))

    def test_short_approved_row_needs_code_pinned_split_exception(self):
        self.assertGreaterEqual(len(gate.normalized_near_text(self.text)),
                                gate.NEAR_COPY_MIN_CHARS)
        short = "Alice works at Acme."
        self.assertLess(len(gate.normalized_near_text(short)), gate.NEAR_COPY_MIN_CHARS)
        self.assertIsNone(gate.development_near_copy(short, []))
        self.text = short
        self.raw["text"] = short
        self.raw_path.write_text(json.dumps(self.raw))
        self.capture_path.write_text(
            f'<div class="s-richtext js-richtext"><p>{short}</p></div>')
        self.candidate["text"] = short
        text_sha = gate.digest(short.encode())
        raw_sha = gate.digest(self.raw_path.read_bytes())
        capture_sha = gate.digest(self.capture_path.read_bytes())
        self.lineage.update(candidate_text_sha256=text_sha, raw_text_sha256=text_sha,
                            raw_file_sha256=raw_sha, raw_excerpt_end_byte=len(short.encode()))
        self.qualification.update(candidate_text_sha256=text_sha, raw_file_sha256=raw_sha)
        self.source_proof.update(candidate_text_sha256=text_sha, raw_file_sha256=raw_sha,
                                 capture_sha256=capture_sha)
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.labels["candidate_text_sha256"] = text_sha
        PROFILES[self.profile_id] = replace(PROFILES[self.profile_id], raw_sha256=raw_sha,
                                            capture_sha256=capture_sha, text_sha256=text_sha)
        self.write()
        _, report = self.check()
        self.assertTrue(any("requires a code-pinned development split exception" in e
                            for e in report["errors"]))
        self.assertFalse(any("source proof" in e for e in report["errors"]))

    def test_approved_long_row_rejects_development_near_copy(self):
        self.text = "Alice works at Acme. " + " ".join(
            f"Section {i} records a distinct office contact procedure."
            for i in range(18))
        self.raw["text"] = self.text
        self.raw_path.write_text(json.dumps(self.raw))
        self.capture_path.write_text(
            f'<div class="s-richtext js-richtext"><p>{self.text}</p></div>')
        self.candidate["text"] = self.text
        text_sha = gate.digest(self.text.encode())
        raw_sha = gate.digest(self.raw_path.read_bytes())
        capture_sha = gate.digest(self.capture_path.read_bytes())
        self.lineage.update(candidate_text_sha256=text_sha, raw_text_sha256=text_sha,
                            raw_file_sha256=raw_sha,
                            raw_excerpt_end_byte=len(self.text.encode()))
        self.qualification.update(candidate_text_sha256=text_sha, raw_file_sha256=raw_sha)
        self.source_proof.update(candidate_text_sha256=text_sha, raw_file_sha256=raw_sha,
                                 capture_sha256=capture_sha)
        self.qualification["source_proof_sha256"] = gate.digest(gate.canonical(self.source_proof))
        self.labels["candidate_text_sha256"] = text_sha
        PROFILES[self.profile_id] = replace(PROFILES[self.profile_id], raw_sha256=raw_sha,
                                            capture_sha256=capture_sha, text_sha256=text_sha)
        self.dev_gold["input"] = self.text.replace("Alice", "Alicx", 1)
        self.write()
        _, report = self.check()
        self.assertTrue(any("near-copy overlaps development gold dev-1" in e
                            for e in report["errors"]))
        self.assertFalse(any("source proof" in e for e in report["errors"]))

    def test_shared_legal_boilerplate_does_not_trigger_near_copy(self):
        legal = " ".join(
            f"Term {i} explains standard use of official online information."
            for i in range(18))
        gold = "Technische Universität München, Arcisstraße 21. " + legal
        candidate = "Bavarian Ministry, Rosenheimer Straße 4. " + legal + " " + " ".join(
            f"Additional office contact provision {i} applies to this ministry."
            for i in range(14))
        normalized = gate.normalized_near_text(gold)
        pages = [("dev-imprint", normalized, gate.character_shingles(normalized))]
        self.assertIsNone(gate.development_near_copy(candidate, pages))

    def test_self_hashed_alternate_evaluation_publisher_index_fails(self):
        self.index["publisher_group"] = "unit:forged-agency"
        self.index["publisher_groups"] = ["unit:forged-agency"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("identity differs from pinned seal" in e
                            for e in report["errors"]))

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

    def test_whitespace_only_development_text_reports_error(self):
        self.dev_gold["input"] = " \n "
        self.write()
        _, report = self.check()
        self.assertTrue(any("development gold empty normalized text dev-1" in e
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

    def test_stage_multiline_silver_array(self):
        config = ('[detector]\n'
                  'silver = [\n'
                  '    "old-one.jsonl",\n'
                  '    "old-two.jsonl",\n'
                  ']\n'
                  'silver_repeat = 10\n')
        staged = gate.stage_silver_config(config, Path('/tmp/new.jsonl'))
        self.assertIn('silver = ["/tmp/new.jsonl"]\n', staged)
        self.assertNotIn('old-one.jsonl', staged)
        self.assertIn('silver_repeat = 10', staged)
        commented = ('[detector]\n'
                     "silver = [ # reviewed sources\n"
                     "  'old-one.jsonl', # first\n"
                     '  "old]two.jsonl",\n'
                     '] # end\n')
        staged = gate.stage_silver_config(commented, Path('/tmp/new.jsonl'))
        self.assertIn('silver = ["/tmp/new.jsonl"] # end', staged)
        self.assertNotIn('old]two.jsonl', staged)
        with self.assertRaisesRegex(ValueError, "exactly one silver array"):
            gate.stage_silver_config(config + config, Path('/tmp/new.jsonl'))
        with self.assertRaisesRegex(ValueError, "only paths"):
            gate.stage_silver_config('[detector]\nsilver = ["old.jsonl", 7]\n',
                                     Path('/tmp/new.jsonl'))

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

    def test_missing_eval_gold_reports_error_without_crashing(self):
        self.gold_path.unlink()
        _, report = self.check()
        self.assertTrue(any("sealed evaluation gold path missing" in e
                            for e in report["errors"]))

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
        with patch.object(gate, "EVAL_GOLD_SHA256", self.evaluation["gold_sha256"]):
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

    def test_development_issuer_alias_overlap_fails(self):
        self.registry["entities"][0]["aliases"] = ["unit:example-alias"]
        self.dev_publisher["roles"]["issuers"] = ["unit:example-agency"]
        self.qualification["publisher"]["roles"]["issuers"] = ["unit:example-alias"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("development publisher overlap" in e for e in report["errors"]))

    def test_development_joint_issuer_and_origin_overlap_fail(self):
        self.dev_publisher["roles"]["issuers"].append("unit:example-agency")
        self.write()
        _, report = self.check()
        self.assertTrue(any("development publisher overlap" in e for e in report["errors"]))
        self.dev_publisher["roles"]["issuers"].remove("unit:example-agency")
        self.dev_publisher["roles"]["origin_publishers"] = ["unit:example-agency"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("development publisher overlap" in e for e in report["errors"]))

    def test_shared_parent_blocks_but_subject_does_not(self):
        self.dev_publisher["roles"]["subjects"] = ["unit:example-agency"]
        self.write()
        _, report = self.check()
        self.assertEqual(report["errors"], [])
        self.dev_publisher["roles"]["controlling_parents"] = ["unit:parent"]
        self.qualification["publisher"]["roles"]["controlling_parents"] = ["unit:parent"]
        self.qualification["publisher_groups"].append("unit:parent")
        self.write()
        _, report = self.check()
        self.assertTrue(any("development publisher overlap" in e for e in report["errors"]))

    def test_source_family_needs_explicit_disjoint_policy(self):
        self.dev_publisher["roles"]["source_families"] = ["unit:example-host"]
        self.write()
        _, report = self.check()
        self.assertEqual(report["errors"], [])
        manifest = json.loads(self.dev_publishers_path.read_text())
        manifest["source_family_policy"] = "disjoint"
        self.dev_publishers_path.write_text(json.dumps(manifest))
        _, report = self.check()
        self.assertTrue(any("development source-family overlap" in e
                            for e in report["errors"]))

    def test_manifest_policy_change_cannot_self_seal(self):
        original_sha = gate.digest(self.dev_publishers_path.read_bytes())
        manifest = json.loads(self.dev_publishers_path.read_text())
        manifest["source_family_policy"] = "disjoint"
        self.dev_publishers_path.write_text(json.dumps(manifest))
        with (patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256),
              patch.dict(gate.DEV_PUBLISHER_PINS_BY_GOLD, {
                  self.trusted_dev_sha256: self.publisher_pins(manifest_sha=original_sha)})):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path,
                dev_publishers_path=self.dev_publishers_path)
        self.assertTrue(any("manifest SHA differs from code pin" in e
                            for e in report["errors"]))

    def test_incomplete_roles_and_registry_conflict_fail(self):
        self.qualification["publisher"]["completeness"]["origin_publishers"] = "unresolved"
        self.write()
        _, report = self.check()
        self.assertTrue(any("lack complete independent review" in e
                            for e in report["errors"]))
        self.qualification["publisher"]["completeness"]["origin_publishers"] = "complete"
        self.registry["entities"][1]["aliases"] = ["unit:example-agency"]
        self.write()
        _, report = self.check()
        self.assertTrue(any("duplicate or invalid publisher alias" in e
                            for e in report["errors"]))

    def test_self_hashed_development_index_without_code_pin_fails(self):
        self.dev_publisher["roles"]["issuers"] = ["unit:example-agency"]
        self.write()
        with patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path,
                dev_publishers_path=self.dev_publishers_path)
        self.assertTrue(any("lack code pins" in e for e in report["errors"]))

    def test_development_publisher_index_requires_exact_gold_rows(self):
        self.write()
        self.dev_index_path.write_bytes(gate.canonical({
            "id": "wrong-id", "text_sha256": gate.digest(self.dev_gold["input"].encode()),
            "publisher": self.dev_publisher}) + b"\n")
        manifest = json.loads(self.dev_publishers_path.read_text())
        manifest["index_sha256"] = gate.digest(self.dev_index_path.read_bytes())
        self.dev_publishers_path.write_text(json.dumps(manifest))
        _, report = self.check()
        self.assertTrue(any("development publisher index ID mismatch" in e
                            for e in report["errors"]))

    def test_development_publisher_row_binds_raw_url_and_hash(self):
        self.write()
        row = json.loads(self.dev_index_path.read_text())
        row["raw_url"] = "https://other.example/forged"
        self.dev_index_path.write_bytes(gate.canonical(row) + b"\n")
        manifest = json.loads(self.dev_publishers_path.read_text())
        manifest["index_sha256"] = gate.digest(self.dev_index_path.read_bytes())
        self.dev_publishers_path.write_text(json.dumps(manifest))
        _, report = self.check()
        self.assertTrue(any("development publisher raw/source binding mismatch" in e
                            for e in report["errors"]))
        row["raw_url"] = self.dev_raw_url
        row["raw_file_sha256"] = "0" * 64
        self.dev_index_path.write_bytes(gate.canonical(row) + b"\n")
        manifest["index_sha256"] = gate.digest(self.dev_index_path.read_bytes())
        self.dev_publishers_path.write_text(json.dumps(manifest))
        _, report = self.check()
        self.assertTrue(any("development publisher raw/source binding mismatch" in e
                            for e in report["errors"]))

    def test_development_source_index_replays_raw_file(self):
        self.dev_raw_path.write_text(self.dev_raw_path.read_text() + " ")
        _, report = self.check()
        self.assertTrue(any("development source raw/gold mismatch" in e
                            for e in report["errors"]))

    def test_development_source_index_has_independent_code_pin(self):
        with (patch.object(gate, "DEV_GOLD_SHA256", self.trusted_dev_sha256),
              patch.dict(gate.DEV_PUBLISHER_PINS_BY_GOLD, {
                  self.trusted_dev_sha256: self.publisher_pins(source_index_sha="0" * 64)})):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path,
                dev_publishers_path=self.dev_publishers_path)
        self.assertTrue(any("identity differs from code pins" in e
                            for e in report["errors"]))

    def test_development_publisher_pins_follow_gold_version(self):
        v1_sha = "1" * 64
        v2_sha = self.trusted_dev_sha256
        with (patch.object(gate, "DEV_GOLD_SHA256", v1_sha),
              patch.object(gate, "DEV_V2_HOSTED_GOLD_SHA256", v2_sha),
              patch.dict(gate.DEV_PUBLISHER_PINS_BY_GOLD, {
                  v1_sha: {"manifest": None, "index": None, "registry": None,
                           "source_index": "0" * 64},
                  v2_sha: self.publisher_pins()})):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path,
                dev_publishers_path=self.dev_publishers_path)
        self.assertEqual(report["errors"], [])
        with (patch.object(gate, "DEV_GOLD_SHA256", v1_sha),
              patch.object(gate, "DEV_V2_HOSTED_GOLD_SHA256", v2_sha),
              patch.dict(gate.DEV_PUBLISHER_PINS_BY_GOLD, {
                  v1_sha: self.publisher_pins(),
                  v2_sha: {"manifest": None, "index": None, "registry": None,
                           "source_index": "0" * 64}})):
            _, report = gate.validate_inputs(
                self.candidate_path, self.lineage_path, self.qualification_path,
                self.labels_path, self.eval_path, self.root,
                eval_hosts_path=self.eval_hosts_path,
                source_proofs_path=self.source_proofs_path,
                dev_publishers_path=self.dev_publishers_path)
        self.assertTrue(any("lack code pins" in e for e in report["errors"]))

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
