"""Replay the frozen official captures through the tracked source profiles."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_profiles import PROFILES, html_200_headers, project, verify_source_profile


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "data/interim/silver/qa-dedup-20260926/train.jsonl"
LINEAGE = ROOT / "data/interim/silver/qa-dedup-20260926/source-lineage-pending-v2.jsonl"
DERIVED = ROOT / "data/interim/silver/clean-derived-pilot-20260927-v1"
DERIVED_PROFILES = {
    "ge-q0389-school-address-excerpt-v1": "ge_q0389_school_excerpt_v1",
    "jp-jma-office-table-v1": "jp_jma_office_table_v1",
    "jp-gsi-3d-contact-v1": "jp_gsi_3d_contact_v1",
    "GBR001": "gb_phs_foi_contact_v1",
}
NEW_DERIVED = ROOT / "data/interim/silver/clean-new-ge-jp-pilot-20260927-v1"
NEW_DERIVED_PROFILES = {
    "GE-RUSTAVI-SCHOOLS-001": "ge_rustavi_schools_v1",
    "jpx-jbaudit-contact-v1": "jp_jbaudit_contact_v1",
    "jpx-jbaudit-staff-v1": "jp_jbaudit_staff_v1",
}


class OfficialCaptureReplayTest(unittest.TestCase):
    def derived_rows(self):
        candidates = {row["id"]: row for row in map(
            json.loads, (DERIVED / "candidate-v1.jsonl").read_text().splitlines())}
        lineage = {row["id"]: row for row in map(
            json.loads, (DERIVED / "source-lineage-v1.jsonl").read_text().splitlines())}
        self.assertEqual(set(candidates), set(DERIVED_PROFILES))
        self.assertEqual(set(lineage), set(DERIVED_PROFILES))
        return candidates, lineage

    def test_held_derived_profiles_replay_real_captures(self):
        candidates, lineage = self.derived_rows()
        for row_id, profile_id in DERIVED_PROFILES.items():
            with self.subTest(profile=profile_id):
                self.assertEqual(verify_source_profile(
                    ROOT, profile_id, candidates[row_id], lineage[row_id]),
                    PROFILES[profile_id])

    def test_held_derived_profiles_fail_on_missing_or_changed_evidence(self):
        candidates, lineage = self.derived_rows()
        for row_id, profile_id in DERIVED_PROFILES.items():
            profile = PROFILES[profile_id]
            with self.subTest(profile=profile_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = [profile.capture_path, profile.source_map_path]
                if profile.header_path:
                    paths.append(profile.header_path)
                if profile.original_raw_path:
                    paths.append(profile.original_raw_path)
                for path in paths:
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / path, target)
                self.assertEqual(verify_source_profile(
                    root, profile_id, candidates[row_id], lineage[row_id]), profile)
                capture = root / profile.capture_path
                original = capture.read_bytes()
                capture.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, "capture SHA mismatch"):
                    verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])
                capture.unlink()
                with self.assertRaises(FileNotFoundError):
                    verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])
                capture.write_bytes(original)
                source_map = root / profile.source_map_path
                source_map.write_bytes(source_map.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "map SHA mismatch"):
                    verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])

    def test_derived_parent_and_header_pins_fail_closed(self):
        candidates, lineage = self.derived_rows()
        for row_id in DERIVED_PROFILES:
            profile_id = DERIVED_PROFILES[row_id]
            profile = PROFILES[profile_id]
            evidence_path = profile.original_raw_path or profile.header_path
            with self.subTest(profile=profile_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for path in (profile.capture_path, profile.source_map_path, evidence_path):
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / path, target)
                target = root / evidence_path
                target.write_bytes(target.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA mismatch|headers mismatch"):
                    verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])

    def test_final_http_response_accepts_html_200_over_http2(self):
        redirected = (b"HTTP/1.1 301 Moved\r\nLocation: https://example.org/new\r\n\r\n"
                      b"HTTP/2 200 \r\ncontent-type: text/html; charset=UTF-8\r\n\r\n")
        self.assertTrue(html_200_headers(redirected))
        self.assertFalse(html_200_headers(
            b"HTTP/2 200 \r\ncontent-type: text/html\r\n\r\n"
            b"HTTP/2 403 \r\ncontent-type: text/html\r\n\r\n"))

    def test_ge_scopes_project_visible_text_and_reject_dropped_content(self):
        eqe = (b'<main><div class="block block--height box-shadow">'
               b'<h2>Contact</h2><p>Alice</p></div><footer>Outside</footer></main>')
        self.assertEqual(project(eqe, "ge_eqe_card_v1"), "Contact\n\nAlice")
        with_break = (b'<div class="block block--height box-shadow">'
                      b'<p>Line<br>Next</p></div>')
        self.assertEqual(project(with_break, "ge_eqe_card_v1"), "Line\nNext")
        gardabani = (b'<main><article class="post-10022"><h1>Office</h1>'
                     b'<p>Main Street 1</p></article><footer>Outside</footer></main>')
        self.assertEqual(project(gardabani, "ge_gardabani_article_v1"),
                         "Office\n\nMain Street 1")
        with self.assertRaisesRegex(ValueError, "omits substantive text"):
            project(eqe.replace(b"<p>Alice</p>", b"<script>Alice</script>"),
                    "ge_eqe_card_v1")

    def test_new_ge_jp_scopes_replay_frozen_review_text(self):
        packets = (
            ("data/raw/ge-municipal-school-pilot-20260927-v1/rustavi.html",
             "ge_rustavi_schools_v1",
             "data/interim/silver/qa-ge-rustavi-schools-20260927-v1/blind-text-v1.jsonl"),
            ("data/raw/jp-jbaudit-contact-20260927-v1/contact.html",
             "jp_jbaudit_contact_v1",
             "data/interim/silver/qa-jp-jbaudit-contact-20260927-v1/blind-text-v1.jsonl"),
            ("data/raw/jp-jbaudit-contact-20260927-v1/staff-messages.html",
             "jp_jbaudit_staff_headings_v1",
             "data/interim/silver/qa-jp-jbaudit-staff-20260927-v1/blind-text-v1.jsonl"),
        )
        for capture, projection, packet in packets:
            with self.subTest(projection=projection):
                if not (ROOT / capture).exists() or not (ROOT / packet).exists():
                    self.skipTest("private capture or blind packet unavailable")
                text = json.loads((ROOT / packet).read_text())["text"]
                self.assertEqual(project((ROOT / capture).read_bytes(), projection), text)

    def test_new_held_profiles_bind_capture_map_and_terms(self):
        candidates = {row["id"]: row for row in map(
            json.loads, (NEW_DERIVED / "candidate-v1.jsonl").read_text().splitlines())}
        lineage = {row["id"]: row for row in map(
            json.loads, (NEW_DERIVED / "source-lineage-v1.jsonl").read_text().splitlines())}
        self.assertEqual(set(candidates), set(NEW_DERIVED_PROFILES))
        self.assertEqual(set(lineage), set(NEW_DERIVED_PROFILES))
        for row_id, profile_id in NEW_DERIVED_PROFILES.items():
            with self.subTest(profile=profile_id):
                self.assertEqual(verify_source_profile(
                    ROOT, profile_id, candidates[row_id], lineage[row_id]), PROFILES[profile_id])

    def test_board_terms_capture_and_source_map_fail_closed(self):
        candidates = {row["id"]: row for row in map(
            json.loads, (NEW_DERIVED / "candidate-v1.jsonl").read_text().splitlines())}
        lineage = {row["id"]: row for row in map(
            json.loads, (NEW_DERIVED / "source-lineage-v1.jsonl").read_text().splitlines())}
        row_id, profile_id = "jpx-jbaudit-contact-v1", "jp_jbaudit_contact_v1"
        profile = PROFILES[profile_id]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (profile.capture_path, profile.header_path, profile.source_map_path,
                         profile.terms_path, profile.terms_header_path):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, target)
            self.assertEqual(verify_source_profile(
                root, profile_id, candidates[row_id], lineage[row_id]), profile)
            terms = root / profile.terms_path
            terms.write_bytes(terms.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "terms SHA mismatch"):
                verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])
            shutil.copyfile(ROOT / profile.terms_path, terms)
            source_map = root / profile.source_map_path
            source_map.write_bytes(source_map.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "map SHA mismatch"):
                verify_source_profile(root, profile_id, candidates[row_id], lineage[row_id])

    def test_all_pinned_profiles_replay_exact_candidate_text(self):
        if not CANDIDATE.exists() or not LINEAGE.exists():
            self.skipTest("private candidate and lineage snapshots unavailable")
        wanted = {profile.row_id for profile in PROFILES.values()
                  if profile.profile_id != "unit_press_v1" and not profile.source_map_path}
        with CANDIDATE.open() as handle:
            candidates = {row["id"]: row for row in map(json.loads, handle)
                          if row["id"] in wanted}
        with LINEAGE.open() as handle:
            lineage = {row["id"]: row for row in map(json.loads, handle)
                       if row["id"] in wanted}
        self.assertEqual(set(candidates), wanted)
        self.assertEqual(set(lineage), wanted)
        for profile in PROFILES.values():
            if profile.profile_id == "unit_press_v1" or profile.source_map_path:
                continue
            with self.subTest(profile=profile.profile_id):
                self.assertEqual(
                    verify_source_profile(ROOT, profile.profile_id,
                                          candidates[profile.row_id], lineage[profile.row_id]),
                    profile)

    def test_unknown_profile_and_identity_substitution_fail(self):
        profile = PROFILES["de_sh_press_q0042_v1"]
        doc = {"source": profile.source, "id": profile.row_id,
               "country": profile.country, "text": ""}
        info = {"raw_url": profile.url, "raw_file_sha256": profile.raw_sha256,
                "candidate_text_sha256": profile.text_sha256}
        with self.assertRaisesRegex(ValueError, "unknown source proof profile"):
            verify_source_profile(ROOT, "caller_script.py", doc, info)
        info["raw_url"] = "https://example.org/forged"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            verify_source_profile(ROOT, profile.profile_id, doc, info)


if __name__ == "__main__":
    unittest.main()
