"""Replay the frozen official captures through the tracked source profiles."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_profiles import PROFILES, project, verify_source_profile


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "data/interim/silver/qa-dedup-20260926/train.jsonl"
LINEAGE = ROOT / "data/interim/silver/qa-dedup-20260926/source-lineage-pending-v2.jsonl"


class OfficialCaptureReplayTest(unittest.TestCase):
    def test_ge_scopes_project_visible_text_and_reject_dropped_content(self):
        eqe = (b'<main><div class="block block--height box-shadow">'
               b'<h2>Contact</h2><p>Alice</p></div><footer>Outside</footer></main>')
        self.assertEqual(project(eqe, "ge_eqe_card_v1"), "Contact\n\nAlice")
        gardabani = (b'<main><article class="post-10022"><h1>Office</h1>'
                     b'<p>Main Street 1</p></article><footer>Outside</footer></main>')
        self.assertEqual(project(gardabani, "ge_gardabani_article_v1"),
                         "Office\n\nMain Street 1")
        with self.assertRaisesRegex(ValueError, "omits substantive text"):
            project(eqe.replace(b"<p>Alice</p>", b"<script>Alice</script>"),
                    "ge_eqe_card_v1")

    def test_all_pinned_profiles_replay_exact_candidate_text(self):
        if not CANDIDATE.exists() or not LINEAGE.exists():
            self.skipTest("private candidate and lineage snapshots unavailable")
        wanted = {profile.row_id for profile in PROFILES.values()
                  if profile.profile_id != "unit_press_v1"}
        with CANDIDATE.open() as handle:
            candidates = {row["id"]: row for row in map(json.loads, handle)
                          if row["id"] in wanted}
        with LINEAGE.open() as handle:
            lineage = {row["id"]: row for row in map(json.loads, handle)
                       if row["id"] in wanted}
        self.assertEqual(set(candidates), wanted)
        self.assertEqual(set(lineage), wanted)
        for profile in PROFILES.values():
            if profile.profile_id == "unit_press_v1":
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
