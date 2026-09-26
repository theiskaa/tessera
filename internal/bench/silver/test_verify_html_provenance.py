"""Structural provenance checks against a small fixture and the frozen SA capture."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_html_provenance import ProvenanceError, verify_packet  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "data/interim/silver/qa-ge-sa-address-eval-20260926"
FROZEN_RAW = ROOT / "data/raw/sa-service-agency/erteulebi-20260926.html"
GEOSTAT_DERIVED = ROOT / "data/interim/silver/geostat-bureau-raw-derived-v2.jsonl"


def sha(data):
    return hashlib.sha256(data).hexdigest()


class HtmlProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw = self.base / "raw.html"
        self.mapping = self.base / "mapping.jsonl"
        self.packet = self.base / "packet.jsonl"
        self.source_url = "https://sa.gov.ge/p/erteulebi"
        self.publisher = "sa:official"
        self.raw_bytes = ("<!DOCTYPE html><div class=\"content\">"
                          "<h5><b>სტრუქტურული ერთეულები:</b></h5>"
                          "<p><b>რუსთავის ცენტრალური დეპარტამენტი</b></p>"
                          "<p>მის.: ქ. რუსთავი, ალექსანდრე ლობჟანიძის ქუჩა N8 "
                          "(ყოფილი მისამართი: ძველი ქუჩა N1)</p>"
                          "<p><b>თბილისის სამმართველო</b></p>"
                          "<p>მის.: ქ. თბილისი, პ. ქავთარაძის ქუჩა N21</p>"
                          "<p>service description with another place</p>"
                          "<p>მის.: ქ. თბილისი, დ. აღმაშენებლის ხეივანი 270</p>"
                          "<h5><b>სამუშაო საათები:</b></h5>"
                          "<footer><p>მის.: footer address N5</p></footer></div>").encode("utf-8")
        self.raw.write_bytes(self.raw_bytes)
        paragraphs = [
            ("heading", "<p><b>რუსთავის ცენტრალური დეპარტამენტი</b></p>"),
            ("current_address", "<p>მის.: ქ. რუსთავი, ალექსანდრე ლობჟანიძის ქუჩა N8 (ყოფილი მისამართი: ძველი ქუჩა N1)</p>"),
            ("heading", "<p><b>თბილისის სამმართველო</b></p>"),
            ("current_address", "<p>მის.: ქ. თბილისი, პ. ქავთარაძის ქუჩა N21</p>"),
            ("current_address", "<p>მის.: ქ. თბილისი, დ. აღმაშენებლის ხეივანი 270</p>"),
        ]
        evidence = []
        for kind, text in paragraphs:
            p = text.encode("utf-8")
            start = self.raw_bytes.index(p)
            evidence.append({"kind": kind, "byte_range": [start, start + len(p)], "raw_sha256": sha(p)})
        self.text_rows = [
            {"row_id": "SAE001", "selected_text": "რუსთავის ცენტრალური დეპარტამენტი\nმის.: ქ. რუსთავი, ალექსანდრე ლობჟანიძის ქუჩა N8"},
            {"row_id": "SAE002", "selected_text": "თბილისის სამმართველო\nმის.: ქ. თბილისი, პ. ქავთარაძის ქუჩა N21\nმის.: ქ. თბილისი, დ. აღმაშენებლის ხეივანი 270"},
        ]
        self.map_rows = []
        for row, spans in zip(self.text_rows, (evidence[:2], evidence[2:])):
            self.map_rows.append({"row_id": row["row_id"], "source_url": self.source_url,
                                  "publisher_group": self.publisher, "raw_html_sha256": sha(self.raw_bytes),
                                  "selected_text_sha256": sha(row["selected_text"].encode("utf-8")),
                                  "source_paragraphs": spans})
        self.freeze()

    def freeze(self):
        self.mapping.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in self.map_rows))
        self.packet.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in self.text_rows))

    def verify(self, **overrides):
        args = dict(raw_path=self.raw, mapping_path=self.mapping, packet_path=self.packet,
                    expected_raw_sha256=sha(self.raw_bytes),
                    expected_mapping_sha256=sha(self.mapping.read_bytes()),
                    expected_packet_sha256=sha(self.packet.read_bytes()),
                    expected_source_url=self.source_url,
                    expected_publisher_group=self.publisher, profile="sa_offices_v1")
        args.update(overrides)
        return verify_packet(**args)

    def test_valid_projection_excludes_history_description_and_footer(self):
        result = self.verify()
        self.assertEqual(result.rows, 2)
        self.assertNotIn("ყოფილი", self.text_rows[0]["selected_text"])
        self.assertNotIn("footer", self.text_rows[1]["selected_text"])

    def test_raw_body_tamper_fails_pinned_sha(self):
        self.raw.write_bytes(self.raw_bytes.replace(b"N21", b"N22"))
        with self.assertRaisesRegex(ProvenanceError, "raw HTML SHA mismatch"):
            self.verify()

    def test_mapping_and_packet_tamper_fail_pinned_sha(self):
        mapping_sha = sha(self.mapping.read_bytes())
        packet_sha = sha(self.packet.read_bytes())
        self.mapping.write_bytes(self.mapping.read_bytes().replace(b"sa:official", b"xx:official", 1))
        with self.assertRaisesRegex(ProvenanceError, "mapping SHA mismatch"):
            self.verify(expected_mapping_sha256=mapping_sha)
        self.freeze()
        self.packet.write_bytes(self.packet.read_bytes().replace(b"N21", b"N22", 1))
        with self.assertRaisesRegex(ProvenanceError, "packet SHA mismatch"):
            self.verify(expected_packet_sha256=packet_sha)

    def test_wrong_paragraph_range_fails_even_with_rehashed_mapping(self):
        self.map_rows[0]["source_paragraphs"][1]["byte_range"][0] += 1
        self.freeze()
        with self.assertRaisesRegex(ProvenanceError, "paragraph sequence/range mismatch"):
            self.verify()

    def test_wrong_paragraph_hash_fails_even_with_rehashed_mapping(self):
        self.map_rows[0]["source_paragraphs"][1]["raw_sha256"] = "0" * 64
        self.freeze()
        with self.assertRaisesRegex(ProvenanceError, "paragraph sequence/range mismatch"):
            self.verify()

    def test_historical_address_reinsertion_fails_projection(self):
        self.text_rows[0]["selected_text"] += " (ყოფილი მისამართი: ძველი ქუჩა N1)"
        self.map_rows[0]["selected_text_sha256"] = sha(self.text_rows[0]["selected_text"].encode())
        self.freeze()
        with self.assertRaisesRegex(ProvenanceError, "parsed text projection mismatch"):
            self.verify()

    def test_missing_office_fails_even_with_rehashed_files(self):
        self.text_rows.pop()
        self.map_rows.pop()
        self.freeze()
        with self.assertRaisesRegex(ProvenanceError, "source/packet row count mismatch"):
            self.verify()

    def test_generic_paragraph_profile_replays_html_entities(self):
        raw = "<main><p>Georgia &amp; partners</p><p>მისამართი: 8 Main Street</p></main>".encode()
        self.raw.write_bytes(raw)
        paragraphs = ["<p>Georgia &amp; partners</p>".encode(),
                      "<p>მისამართი: 8 Main Street</p>".encode()]
        spans = []
        for p in paragraphs:
            start = raw.index(p)
            spans.append({"kind": "text", "byte_range": [start, start + len(p)], "raw_sha256": sha(p)})
        self.text_rows = [{"row_id": "GEO001", "selected_text": "Georgia & partners\nმისამართი: 8 Main Street"}]
        self.map_rows = [{"row_id": "GEO001", "source_url": "https://www.geostat.ge/example",
                          "publisher_group": "geostat:official", "raw_html_sha256": sha(raw),
                          "selected_text_sha256": sha(self.text_rows[0]["selected_text"].encode()),
                          "source_paragraphs": spans}]
        self.freeze()
        result = self.verify(expected_raw_sha256=sha(raw),
                             expected_source_url="https://www.geostat.ge/example",
                             expected_publisher_group="geostat:official", profile="paragraphs_v1")
        self.assertEqual(result.rows, 1)

    def test_source_and_publisher_spoof_fail_pinned_identity(self):
        for key, value, message in (("source_url", "https://other.example/", "source URL mismatch"),
                                    ("publisher_group", "other:official", "publisher group mismatch")):
            rows = copy.deepcopy(self.map_rows)
            rows[0][key] = value
            self.map_rows = rows
            self.freeze()
            with self.assertRaisesRegex(ProvenanceError, message):
                self.verify()
            self.map_rows[0][key] = self.source_url if key == "source_url" else self.publisher

    @unittest.skipUnless(FROZEN_RAW.is_file() and (FROZEN / "source-mapping-v2.jsonl").is_file(),
                         "private frozen SA artifacts are absent")
    def test_actual_frozen_sa_capture(self):
        result = verify_packet(
            raw_path=FROZEN_RAW, mapping_path=FROZEN / "source-mapping-v2.jsonl",
            packet_path=FROZEN / "blind-text-v1.jsonl",
            expected_raw_sha256="98132a606f8b26afbac8443f2769c23e805d6c4437a2c057202b88824ca90734",
            expected_mapping_sha256="fb48a7e27352a05687ae6972b623694101e80558207f4de7afe5a23927d564d8",
            expected_packet_sha256="1b802f46ac7d3637c285d7ee45df9c4ed8170ce1c8ae89aca2fb72d56a7ef34f",
            expected_source_url="https://sa.gov.ge/p/erteulebi",
            expected_publisher_group="sa:official", profile="sa_offices_v1")
        self.assertEqual(result.rows, 12)

    @unittest.skipUnless(GEOSTAT_DERIVED.is_file(), "private frozen Geostat artifacts are absent")
    def test_actual_geostat_paragraphs_with_generic_profile(self):
        derived_bytes = GEOSTAT_DERIVED.read_bytes()
        self.assertEqual(sha(derived_bytes),
                         "eeb9980678c5b41299fc623ed1116d40807916671c71027cea164fedb2b5c252")
        records = [json.loads(line) for line in derived_bytes.splitlines()]
        self.assertEqual(len(records), 11)
        for row in records:
            with self.subTest(row_id=row["row_id"]):
                raw = ROOT.joinpath(row["raw_html_path"]).read_bytes()
                self.assertEqual(sha(raw), row["raw_html_sha256"])
                start, end = row["raw_address_p_start_byte"], row["raw_address_p_end_byte"]
                paragraph = raw[start:end]
                self.raw.write_bytes(raw)
                self.text_rows = [{"row_id": row["row_id"], "selected_text": row["selected_text"]}]
                self.map_rows = [{
                    "row_id": row["row_id"], "source_url": row["source_url"],
                    "publisher_group": "geostat:official", "raw_html_sha256": row["raw_html_sha256"],
                    "selected_text_sha256": row["selected_text_sha256"],
                    "source_paragraphs": [{"kind": "text", "byte_range": [start, end],
                                           "raw_sha256": sha(paragraph)}],
                }]
                self.freeze()
                verified = self.verify(expected_raw_sha256=row["raw_html_sha256"],
                                       expected_source_url=row["source_url"],
                                       expected_publisher_group="geostat:official",
                                       profile="paragraphs_v1")
                self.assertEqual(verified.rows, 1)


if __name__ == "__main__":
    unittest.main()
