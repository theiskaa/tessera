"""Reconcile independent acronym labels after the source-family split correction."""

import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from freeze_2024_acronym_packet import RAW


ROOT = Path(__file__).resolve().parents[2]
PACKET_V1 = ROOT / "data/interim/silver/r24/us-2024-acronym-prose-blind-v1.jsonl"
PACKET_V2 = ROOT / "data/interim/silver/r24/us-2024-acronym-prose-blind-v2.jsonl"
REVIEW_A = ROOT / "data/interim/review/us-2024-acronym-prose-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-2024-acronym-prose-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2024-acronym-prose-adjudication-v2.json"
OUT = ROOT / "data/interim/review/us-2024-acronym-prose-gold-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET_V1, "packet_v1_sha256"),
                      (PACKET_V2, "packet_v2_sha256"),
                      (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"blind input changed: {path.name}")
    original, selected, review_a, review_b = map(
        read_rows, (PACKET_V1, PACKET_V2, REVIEW_A, REVIEW_B))
    original_names = [row["name"] for row in original]
    selected_names = [row["name"] for row in selected]
    if (len(set(original_names)) != len(original_names) or
            len(set(selected_names)) != len(selected_names) or
            [row["name"] for row in review_a] != original_names or
            [row["name"] for row in review_b] != original_names):
        raise ValueError("blind reviews do not match the original packet order")
    original_by_name = {row["name"]: row for row in original}
    if any(original_by_name.get(row["name"]) != row for row in selected):
        raise ValueError("corrected packet changed a reviewed case")
    review_a_by_name = {row["name"]: row for row in review_a}
    review_b_by_name = {row["name"]: row for row in review_b}
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate adjudication")
    disputed = set()
    result = []
    for source in selected:
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].encode()[source["source_byte_start"]:
                                     source["source_byte_end"]].decode() != source["input"]):
            raise ValueError(f"acronym case differs from raw source: {name}")
        a, b = review_a_by_name[name], review_b_by_name[name]
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
        result.append({"name": name, "country": "US", "doc_type": "notice_prose",
                       "input": source["input"], "expected": spans,
                       "source_id": source["source_id"],
                       "source_url": source["source_url"],
                       "source_group": source["source_group"]})
    if set(choices) != disputed:
        raise ValueError("adjudications differ from disputed or uncertain cases")
    return result, len(disputed)


def main():
    rows, adjudications = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2024 acronym gold changed after freezing")
    if not OUT.exists():
        OUT.write_bytes(data)
    manifest = {
        "kind": "us_2024_acronym_prose_gold",
        "cases": len(rows), "adjudications": adjudications,
        "packet_v1_sha256": digest(PACKET_V1.read_bytes()),
        "packet_v2_sha256": digest(PACKET_V2.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "reviewed_training_candidate_pending_data_audit",
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2024 acronym gold manifest changed")
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reconciled 2024 acronym cases, {adjudications} adjudications: {digest(data)}")


if __name__ == "__main__":
    main()
