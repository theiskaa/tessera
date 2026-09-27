"""Focused tests for the scratch synthetic overlap policy."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import filter_synthetic_overlap as subject


def span_row(doc_id, text, value, kind="org"):
    encoded = text.encode("utf-8")
    start = encoded.index(value.encode("utf-8"))
    return {"doc_id": doc_id, "text": text, "entities_json": json.dumps([
        {"kind": kind, "start": start, "end": start + len(value.encode("utf-8"))}
    ])}


class OverlapPolicyTests(unittest.TestCase):
    def setUp(self):
        self.names = {"org": {"tbilisi state university", "state university", "bvg",
                              "ministry of economy", "department of health",
                              "state audit office", "ministry of internal affairs",
                              "revenue service", "supreme court",
                              "new york times", "washington post",
                              "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტი",
                              "ეკონომიკისა და ბიზნესის ფაკულტეტი",
                              "ილიას სახელმწიფო უნივერსიტეტი",
                              "სახელმწიფო სერვისების განვითარების სააგენტო",
                              "საქართველოს ეკონომიკისა და მდგრადი განვითარების სამინისტრო",
                              "თიბისი ბანკი", "პროკრედიტ ბანკი",
                              "royal mail group", "punch taverns",
                              "აგრარული უნივერსიტეტი", "procredit bank",
                              "european commission", "მათემატიკის ინსტიტუტი",
                              "freie und hansestadt hamburg", "data protection unit",
                              "საერთაშორისო ურთიერთობების დეპარტამენტი",
                              "総務省", "農林水産省"},
                      "person": {"jane doe"}, "address": {"123 main street"}}
        self.policy = subject.embedded_org_policy(self.names)

    def test_specific_multiword_and_japanese_compounds(self):
        cases = {
            "Tbilisi State University-ის": "embedded_multiword_org",
            "Tbilisi State University-ში": "embedded_multiword_org",
            "Ivane Javakhishvili Tbilisi State University": "embedded_multiword_org",
            "農林水産省 産業振興係 環境保全課": "jp_ministry_unit_compound",
            "総務省総合通信基盤局": "jp_ministry_unit_compound",
            "State Audit Office of Georgia": "ge_jurisdiction_alias",
            "Ministry of Internal Affairs of Georgia-ის": "ge_jurisdiction_alias",
            "Revenue Service of Georgia": "ge_jurisdiction_alias",
            "Supreme Court of Georgia-თან": "ge_jurisdiction_alias",
            "The New York Times": "leading_article_alias",
            "The Washington Post": "leading_article_alias",
            "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტის": "ge_attested_inflection",
            "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტში": "ge_attested_inflection",
            "ეკონომიკისა და ბიზნესის ფაკულტეტის": "ge_attested_inflection",
            "ეკონომიკისა და ბიზნესის ფაკულტეტში": "ge_attested_inflection",
            "ილიას სახელმწიფო უნივერსიტეტის": "ge_attested_inflection",
            "ილიას სახელმწიფო უნივერსიტეტში": "ge_attested_inflection",
            "სახელმწიფო სერვისების განვითარების სააგენტოს": "ge_attested_inflection",
            "სახელმწიფო სერვისების განვითარების სააგენტომ": "ge_attested_inflection",
            "სახელმწიფო სერვისების განვითარების სააგენტოსთან": "ge_attested_inflection",
            "საქართველოს ეკონომიკისა და მდგრადი განვითარების სამინისტროსთან": "ge_attested_inflection",
            "თიბისი ბანკთან": "ge_attested_inflection",
            "სს თიბისი ბანკი": "ge_attested_legal_prefix",
            "სს პროკრედიტ ბანკი": "ge_attested_legal_prefix",
            "Royal Mail Group Limited": "gb_attested_legal_suffix",
            "Punch Taverns PLC": "gb_attested_legal_suffix",
            "საქართველოს აგრარული უნივერსიტეტი": "additional_attested_alias",
            "ProCredit Bank-თან": "additional_attested_alias",
            "European Commission-ის": "additional_attested_alias",
            "Tbilisi State University Foundation": "strict_broad_protected_literal",
            "Ministry of Economy, Trade and Industry": "strict_broad_protected_literal",
            "Department of Health and Social Care": "strict_broad_protected_literal",
            "Service Agency of the Ministry of Internal Affairs of Georgia": "strict_broad_protected_literal",
            "International School of Economics at Tbilisi State University": "strict_broad_protected_literal",
            "Internal Revenue Service": "strict_curated_protected_literal",
            "სახელმწიფო სერვისების განვითარების სააგენტოს დეპარტამენტი": "strict_sda_inflection_literal",
            "新総務省総合通信基盤局": "strict_jp_head_literal",
        }
        for value, rule in cases.items():
            with self.subTest(value=value):
                self.assertEqual(subject.overlap(value, "org", self.names, self.policy)[0], rule)

    def test_short_generic_and_partial_words_do_not_match(self):
        for value in ("NotTbilisi State University", "Tbilisi State UniversityXYZ",
                      "State University Hospital", "BVG Media",
                      "City St George’s, University Of London",
                      "St Anne's College In The University Of Oxford",
                      "Shota Rustaveli National Science Foundation",
                      "The New York Times Company",
                      "The Washington Post Company",
                      "The New York Times-ის",
                      "New York Times Company",
                      "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტის კლინიკა",
                      "სს თიბისი ბანკი ფილიალი",
                      "სს სხვა ბანკი",
                      "Punch Taverns PLC Holdings",
                      "საქართველოს აგრარული უნივერსიტეტის კლინიკა",
                      "ProCredit Bank-თან Group",
                      "European Commission-ის Directorate",
                      "თბილისის მათემატიკის ინსტიტუტი",
                      "თბილისის მერიის საერთაშორისო ურთიერთობების დეპარტამენტიX"):
            with self.subTest(value=value):
                self.assertIsNone(subject.overlap(value, "org", self.names, self.policy))
        self.assertEqual(subject.overlap("BVG", "org", self.names, self.policy)[0], "exact")
        self.assertEqual(subject.overlap("Jane Doe", "person", self.names, self.policy)[0], "exact")
        self.assertIsNone(subject.overlap("Dr Jane Doe", "person", self.names, self.policy))

    def test_parquet_filter_drops_entire_rows_and_keeps_near_misses(self):
        rows = [
            span_row(1, "At Tbilisi State University-ის today", "Tbilisi State University-ის"),
            span_row(2, "農林水産省 産業振興係 環境保全課", "農林水産省 産業振興係 環境保全課"),
            span_row(3, "State University Hospital", "State University Hospital"),
            span_row(4, "BVG Media", "BVG Media"),
            span_row(5, "Jane Doe", "Jane Doe", "person"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "small.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            with patch.object(subject, "OUTPUT", root / "out"):
                report = subject.filter_shard(source, self.names, self.policy)
            result = pq.read_table(root / "out/small.parquet").to_pylist()
        self.assertEqual([row["doc_id"] for row in result], [3, 4])
        self.assertEqual(report["source_rows"], 5)
        self.assertEqual(report["kept_rows"], 2)
        self.assertEqual(report["removed"]["any"], 3)
        self.assertEqual(report["matching_spans_by_rule"], {
            "embedded_multiword_org": 1, "jp_ministry_unit_compound": 1, "exact": 1,
        })

    def test_pinned_gold_hash_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/interim/review/qa-sa-bounded-20260926/gold-v1.jsonl"
            path.parent.mkdir(parents=True)
            original = (json.dumps({"expected": [{"kind": "org", "text": "Tbilisi State University"}]})
                        + "\n").encode()
            path.write_bytes(original)
            with (patch.object(subject, "ROOT", root), patch.object(subject, "GOLD", [path]),
                  patch.object(subject, "PINNED_GOLD", {str(path.relative_to(root)):
                      hashlib.sha256(original).hexdigest()})):
                names, gold_hash = subject.gold_names()
                self.assertIn("tbilisi state university", names["org"])
                self.assertEqual(len(gold_hash), 64)
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "pinned gold changed"):
                    subject.gold_names()

    def test_pinned_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            path.write_bytes(b"source")
            with patch.object(subject, "PINNED_SOURCE", {"train.parquet":
                    hashlib.sha256(b"source").hexdigest()}):
                subject.checked_source(path)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pinned source changed"):
                    subject.checked_source(path)

    def test_full_text_policy_counts_non_org_hits_separately(self):
        address_text = "Office\nFreie und Hansestadt Hamburg"
        crossed_text = "Michigan Department of Data Protection\nUnit 21"
        self.assertEqual(subject.full_text_overlap(address_text, self.names, self.policy),
                         ("strict_full_text_broad_literal", "freie und hansestadt hamburg"))
        self.assertEqual(subject.full_text_overlap(crossed_text, self.names, self.policy),
                         ("strict_full_text_broad_literal", "data protection unit"))
        self.assertIsNone(subject.full_text_overlap(
            "Office\nFreie und Hansestadt Hamburger", self.names, self.policy))
        rows = [span_row(1, address_text, "Freie und Hansestadt Hamburg", "address"),
                span_row(2, "Ordinary address", "Ordinary address", "address")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "small.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            with patch.object(subject, "OUTPUT", root / "out"):
                report = subject.filter_shard(source, self.names, self.policy)
            result = pq.read_table(root / "out/small.parquet").to_pylist()
        self.assertEqual([row["doc_id"] for row in result], [2])
        self.assertEqual(report["matching_spans_by_rule"],
                         {"strict_full_text_broad_literal": 1})
        self.assertEqual(report["removed"], {"full_text": 1, "any": 1})

    def test_curated_and_japanese_heads_apply_outside_org_spans(self):
        self.assertEqual(subject.full_text_overlap(
            "Return address: Internal Revenue Service", self.names, self.policy),
            ("strict_full_text_curated_literal", "revenue service"))
        self.assertEqual(subject.full_text_overlap(
            "所在地: 総務省", self.names, self.policy),
            ("strict_full_text_jp_head_literal", "総務省"))
        self.assertEqual(subject.full_text_overlap(
            "სახელმწიფო სერვისების განვითარების სააგენტოს დეპარტამენტი",
            self.names, self.policy),
            ("strict_full_text_sda_inflection_literal",
             "სახელმწიფო სერვისების განვითარების სააგენტო"))
        self.assertIsNone(subject.full_text_overlap(
            "Internal Revenue Servicex", self.names, self.policy))

    def test_explicit_ge_aliases_require_protected_gold_base(self):
        names = {"org": set(), "person": set(), "address": set()}
        policy = subject.embedded_org_policy(names)
        for value in ("თიბისი ბანკთან", "სს თიბისი ბანკი", "სს პროკრედიტ ბანკი",
                      "სახელმწიფო სერვისების განვითარების სააგენტოს",
                      "Royal Mail Group Limited", "Punch Taverns PLC",
                      "საქართველოს აგრარული უნივერსიტეტი", "ProCredit Bank-თან",
                      "European Commission-ის"):
            with self.subTest(value=value):
                self.assertIsNone(subject.overlap(value, "org", names, policy))


if __name__ == "__main__":
    unittest.main()
