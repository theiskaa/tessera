"""Freeze independently reviewed Federal Register narrative passages."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION, ROOT, rows
from freeze_us_person_long_v3_silver import digest, label_key, read_review


PACKET = ROOT / "data/raw/candidates/us-long-prose-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
PACKET_SHA256 = "c5c1a1260e79ff81af7e0472ba1b81327a9cee40018b595bede5d86fd17f42e9"
REVIEW = ROOT / "data/interim/review"
REVIEWS = {name: REVIEW / f"us-long-prose-v1-review-{name}.jsonl" for name in "ab"}
NOTES = tuple(REVIEW / f"us-long-prose-v1-review-{name}-uncertain.json" for name in "ab")
DECISIONS = REVIEW / "us-long-prose-v1-resolutions.json"
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
OUT = ROOT / "data/interim/silver/us-long-prose-reviewed-v1.jsonl"


def reviewed_rows():
    """Require policy-pinned decisions supported by complete independent agreement."""
    manifest = json.loads(PACKET_MANIFEST.read_text())
    if (digest(PACKET) != PACKET_SHA256 or manifest["packet_sha256"] != PACKET_SHA256 or
            manifest["kind"] != "us_long_prose_blind_v1" or manifest["cases"] != 3 or
            manifest["label_status"] != "unlabeled" or manifest["training_eligible"] or
            manifest["evaluation_eligible"]):
        raise ValueError("long prose packet changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"long prose screening input changed: {name}")
    original = rows(PACKET)
    packet = {row["source_id"]: row for row in original}
    if len(packet) != 3 or len(original) != 3:
        raise ValueError("long prose packet repeats or omits sources")
    for row in original:
        if digest(ROOT / row["source_file"]) != row["raw_sha256"]:
            raise ValueError(f"long prose raw source changed: {row['source_id']}")
    reviews = {name: read_review(path, packet, set(packet))
               for name, path in REVIEWS.items()}
    decisions = json.loads(DECISIONS.read_text())
    if (decisions["packet_sha256"] != PACKET_SHA256 or
            decisions["policy_sha256"] != digest(POLICY) or
            decisions["review_sha256"] != {name: digest(path) for name, path in REVIEWS.items()}):
        raise ValueError("long prose decisions refer to different inputs")
    accepted, excluded = decisions["accepted"], decisions["excluded"]
    if (not accepted or set(accepted) & set(excluded) or
            set(accepted) | set(excluded) != set(packet) or
            any(not reason.strip() for reason in excluded.values())):
        raise ValueError("long prose decisions do not cover the packet")
    selected = []
    for row in original:
        source_id = row["source_id"]
        if source_id not in accepted:
            continue
        decision = accepted[source_id]
        if decision["review"] not in reviews or not decision.get("reason", "").strip():
            raise ValueError(f"missing long prose decision: {source_id}")
        chosen = reviews[decision["review"]][source_id]
        agreement = sum(source_id in review and label_key(review[source_id]) == label_key(chosen)
                        for review in reviews.values())
        if agreement < 2 or len(chosen["input"].split()) < 200:
            raise ValueError(f"long prose lacks independent positive supervision: {source_id}")
        selected.append(chosen)
    hits = heldout_reasons(selected, [row for path in EVALUATION for row in rows(path)])
    if hits:
        raise ValueError(f"long prose labels overlap evaluation: {hits}")
    return selected, excluded


def main():
    """Keep originals intact and replay the same approved training examples."""
    selected, excluded = reviewed_rows()
    silver = []
    for row in selected:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected", "training_eligible", "evaluation_eligible"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "entities": [{key: span[key] for key in ("kind", "start", "end")}
                                    for span in row["expected"]]})
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in silver).encode()
    raw_sources = tuple(ROOT / row["source_file"] for row in selected)
    inputs = (PACKET, PACKET_MANIFEST, *raw_sources, *REVIEWS.values(), *NOTES, DECISIONS, POLICY,
              *EVALUATION, Path(__file__), ROOT / "bench/us/freeze_us_person_long_v3_silver.py",
              ROOT / "bench/us/check_us_blind_packet_holdout.py")
    manifest = {
        "kind": "us_long_prose_reviewed_v1", "training_eligible": True,
        "active_in_training_config": False, "cases": len(silver),
        "source_documents": len(silver), "unresolved_labels_excluded": excluded,
        "spans_by_kind": dict(sorted(collections.Counter(
            span["kind"] for row in silver for span in row["entities"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = OUT.with_suffix(".manifest.json")
    if OUT.exists() != manifest_path.exists():
        raise ValueError("long prose silver or manifest is missing")
    if OUT.exists() and (OUT.read_bytes() != data or manifest_path.read_bytes() != encoded):
        raise ValueError("frozen long prose labels or inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        manifest_path.write_bytes(encoded)
    print(f"{len(silver)} long narrative passages frozen; {manifest['spans_by_kind']}")


if __name__ == "__main__":
    main()
