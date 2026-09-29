"""Merge reviewed contact cases with adjudicated full-year additions."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from build_silver import family
from freeze_2025_contact_packet import RAW
from freeze_2025_ready_contact_packet import OUT as READY_PACKET
from freeze_2025_year_contact_delta import OUT as DELTA_PACKET
from freeze_2025_year_contact_packet import OUT as YEAR_PACKET


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
READY_GOLD = REVIEW / "us-2025-ready-contact-gold-v1.jsonl"
REVIEW_A = REVIEW / "us-2025-year-contact-delta-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-2025-year-contact-delta-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2025-year-contact-delta-adjudication-v1.json"
OUT = REVIEW / "us-2025-year-contact-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    decisions = json.loads(DECISIONS.read_text())
    for path, key in ((DELTA_PACKET, "packet_sha256"),
                      (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != decisions[key]:
            raise ValueError(f"blind year contact input changed: {path.name}")
    ready_manifest = json.loads(READY_GOLD.with_suffix(".manifest.json").read_text())
    if (digest(READY_GOLD.read_bytes()) != ready_manifest["sha256"] or
            digest(READY_PACKET.read_bytes()) != ready_manifest["packet_sha256"] or
            ready_manifest["training_eligible"]):
        raise ValueError("previously reviewed contact gold changed")
    year_manifest = json.loads(YEAR_PACKET.with_suffix(".manifest.json").read_text())
    if (digest(YEAR_PACKET.read_bytes()) != year_manifest["sha256"] or
            year_manifest["training_eligible"]):
        raise ValueError("frozen full-year contact packet changed")
    year = read_rows(YEAR_PACKET)
    ready = {row["name"]: row for row in read_rows(READY_GOLD)}
    ready_packet = {row["name"]: row for row in read_rows(READY_PACKET)}
    delta = read_rows(DELTA_PACKET)
    a = read_rows(REVIEW_A)
    b = read_rows(REVIEW_B)
    delta_names = [row["name"] for row in delta]
    if (delta_names != [row["name"] for row in a] or
            delta_names != [row["name"] for row in b] or
            len(set(delta_names)) != len(delta_names) or
            len({row["name"] for row in year}) != len(year)):
        raise ValueError("full-year contact review order differs")
    delta_map = {row["name"]: (row, left, right)
                 for row, left, right in zip(delta, a, b)}
    if set(delta_map) != {row["name"] for row in year} - set(ready_packet):
        raise ValueError("contact delta does not cover all new year cases")
    choices = {row["name"]: row for row in decisions["decisions"]}
    audits = {row["name"]: row for row in decisions["agreed_audits"]}
    if (len(choices) != len(decisions["decisions"]) or
            len(audits) != len(decisions["agreed_audits"])):
        raise ValueError("duplicate year contact adjudication")
    disputed = set()
    agreed = set()
    reused = 0
    result = []
    for source in year:
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"full-year case differs from source: {name}")
        if name in ready:
            old = ready[name]
            if (ready_packet[name] != source or old["input"] != source["input"] or
                    old["source_id"] != source["source_id"] or
                    old["source_url"] != source["source_url"]):
                raise ValueError(f"reviewed contact changed in full year: {name}")
            spans = validate_spans(name, source["input"], old["expected"])
            reused += 1
        else:
            frozen, left, right = delta_map[name]
            if frozen != source:
                raise ValueError(f"new contact changed in full year: {name}")
            a_spans = validate_spans(name, source["input"], left["entities"])
            b_spans = validate_spans(name, source["input"], right["entities"])
            if a_spans != b_spans:
                disputed.add(name)
                choice = choices.get(name)
                if choice is None or choice.get("take") not in {"a", "b"} or not choice.get("reason"):
                    raise ValueError(f"unadjudicated year contact: {name}")
                spans = a_spans if choice["take"] == "a" else b_spans
            else:
                agreed.add(name)
                spans = a_spans
        result.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"], "source_url": source["source_url"],
            "source_group": family(source["source_url"]),
        })
    if set(choices) != disputed or not set(audits) <= agreed or reused != len(year) - len(delta):
        raise ValueError("year contact adjudications or reused cases differ")
    if any(not row.get("reason") for row in audits.values()):
        raise ValueError("year contact agreement audit lacks reason")
    return result, reused, len(disputed)


def main():
    rows, reused, adjudications = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_year_contact_gold_v1",
        "cases": len(rows), "reused_reviewed_cases": reused,
        "new_cases": len(rows) - reused, "adjudications": adjudications,
        "entity_counts": dict(sorted(counts.items())),
        "year_packet_sha256": digest(YEAR_PACKET.read_bytes()),
        "ready_gold_sha256": digest(READY_GOLD.read_bytes()),
        "delta_packet_sha256": digest(DELTA_PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_group",
        "training_eligible": False,
        "strict_entity_holdout": False,
        "intended_use": "split_candidate_pending_overlap_audit",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen year contact gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2025 year contact gold changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2025 year contact gold manifest changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed full-year contacts, {len(rows)-reused} new, {adjudications} adjudications")


if __name__ == "__main__":
    main()
