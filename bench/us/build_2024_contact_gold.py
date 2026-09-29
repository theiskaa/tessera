"""Reconcile two independent label passes on 2024 contact excerpts."""

import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from freeze_2024_contact_packet import RAW


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/interim/silver/r24/us-2024-contact-blind-v2.jsonl"
REVIEW_A = ROOT / "data/interim/review/us-2024-contact-review-a-v2.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-2024-contact-review-b-v2.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2024-contact-adjudication-v3.json"
OUT = ROOT / "data/interim/review/us-2024-contact-gold-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"blind input changed: {path.name}")
    packet, review_a, review_b = map(read_rows, (PACKET, REVIEW_A, REVIEW_B))
    names = [row["name"] for row in packet]
    if (len(names) != len(set(names)) or
            [row["name"] for row in review_a] != names or
            [row["name"] for row in review_b] != names):
        raise ValueError("blind reviews do not match the frozen packet order")
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate adjudication")
    disputed = set()
    result = []
    for source, a, b in zip(packet, review_a, review_b):
        name = source["name"]
        raw_path = RAW / f"{source['source_id']}.json"
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                not raw["date"].startswith("2024-") or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"contact case differs from raw source: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a_spans != b_spans or a.get("uncertain") or b.get("uncertain"):
            disputed.add(name)
            decision = choices.get(name)
            if decision is None or decision.get("take") not in {"a", "b"} or not decision.get("reason"):
                raise ValueError(f"unadjudicated blind review: {name}")
            spans = a_spans if decision["take"] == "a" else b_spans
        else:
            spans = a_spans
        result.append({"name": name, "country": "US", "doc_type": "notice_contact",
                       "input": source["input"], "expected": spans,
                       "source_id": source["source_id"],
                       "source_url": source["source_url"]})
    if set(choices) != disputed:
        raise ValueError("adjudications differ from disputed or uncertain cases")
    return result, len(disputed)


def main():
    rows, adjudications = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2024 contact gold changed after freezing")
    if not OUT.exists():
        OUT.write_bytes(data)
    manifest = {
        "kind": "us_2024_contact_gold_v3",
        "cases": len(rows), "adjudications": adjudications,
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_id",
        "training_eligible": False,
        "strict_entity_holdout": False,
        "intended_use": "split_candidate_pending_overlap_audit",
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2024 contact gold manifest changed")
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reconciled 2024 contacts, {adjudications} adjudications: {digest(data)}")


if __name__ == "__main__":
    main()
