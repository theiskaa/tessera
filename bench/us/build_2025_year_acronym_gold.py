"""Reconcile independent organization labels on 2025 action prose."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from freeze_2024_acronym_packet import RAW


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PACKET = ROOT / "data/interim/silver/r24/us-2025-year-acronym-prose-blind-v3.jsonl"
PACKET = ROOT / "data/interim/silver/r24/us-2025-year-acronym-prose-blind-v4.jsonl"
REVIEW_A = ROOT / "data/interim/review/us-2025-year-acronym-prose-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-2025-year-acronym-prose-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2025-year-acronym-prose-adjudication-v1.json"
PREVIOUS_OUT = ROOT / "data/interim/review/us-2025-year-acronym-prose-gold-v1.jsonl"
OUT = ROOT / "data/interim/review/us-2025-year-acronym-prose-gold-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((ORIGINAL_PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"blind acronym input changed: {path.name}")
    selected_manifest = json.loads(PACKET.with_suffix(".manifest.json").read_text())
    if (selected_manifest["sha256"] != digest(PACKET.read_bytes()) or
            selected_manifest["supersedes_sha256"] != digest(ORIGINAL_PACKET.read_bytes()) or
            selected_manifest["training_eligible"]):
        raise ValueError("corrected acronym packet changed")
    original, packet, left, right = map(
        read_rows, (ORIGINAL_PACKET, PACKET, REVIEW_A, REVIEW_B))
    original_names = [row["name"] for row in original]
    names = [row["name"] for row in packet]
    if (len(set(original_names)) != len(original_names) or
            len(set(names)) != len(names) or
            [row["name"] for row in left] != original_names or
            [row["name"] for row in right] != original_names):
        raise ValueError("2025 acronym reviews do not match packet order")
    original_by_name = {row["name"]: row for row in original}
    if any(original_by_name.get(row["name"]) != row for row in packet):
        raise ValueError("corrected acronym packet changed a reviewed case")
    left_by_name = {row["name"]: row for row in left}
    right_by_name = {row["name"]: row for row in right}
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate acronym adjudication")
    disputed = set()
    rows = []
    for source in packet:
        name = source["name"]
        a, b = left_by_name[name], right_by_name[name]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].encode()[source["source_byte_start"]:
                                     source["source_byte_end"]].decode() != source["input"] or
                raw["text"].encode()[source["expansion_byte_start"]:
                                     source["expansion_byte_end"]].decode() !=
                source["expansion_evidence"]):
            raise ValueError(f"acronym case differs from source: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a_spans != b_spans or a.get("uncertain") or b.get("uncertain"):
            disputed.add(name)
            choice = choices.get(name)
            if choice is None or choice.get("take") not in {"a", "b"} or not choice.get("reason"):
                raise ValueError(f"unadjudicated 2025 acronym case: {name}")
            spans = a_spans if choice["take"] == "a" else b_spans
        else:
            spans = a_spans
        rows.append({"name": name, "country": "US", "doc_type": "notice_prose",
                     "input": source["input"], "expected": spans,
                     "source_id": source["source_id"],
                     "source_url": source["source_url"],
                     "source_group": source["source_group"]})
    if set(choices) != disputed:
        raise ValueError("acronym adjudications differ from review disputes")
    return rows, len(disputed)


def main():
    rows, adjudications = gold_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha or previous_manifest["training_eligible"]:
        raise ValueError("previous acronym gold changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_year_acronym_prose_gold_v2",
        "supersedes_sha256": previous_sha,
        "cases": len(rows), "adjudications": adjudications,
        "entity_counts": dict(sorted(counts.items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "original_packet_sha256": digest(ORIGINAL_PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data), "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "reviewed_training_candidate_pending_split_audit",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen 2025 acronym gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2025 acronym gold changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2025 acronym gold inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reconciled 2025 acronym cases, {adjudications} adjudications")


if __name__ == "__main__":
    main()
