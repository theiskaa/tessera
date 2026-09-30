"""Freeze independently reviewed long US address passages without starting training."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons, validate_spans
from check_us_next_data_split import EVALUATION, ROOT, rows
from source_identity import document_key


PACKET = ROOT / "data/raw/candidates/us-address-long-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
PACKET_SHA256 = "0e0b7ebefc856187ae7bf2d8a76da642433452068f3d339777e8d743d4f42daa"
REVIEW = ROOT / "data/interim/review"
REVIEW_VERSIONS = {"a": "a-policy-v2", "b": "b-policy-v2", "c": "c"}
REVIEWS = {name: REVIEW / f"us-address-long-v1-review-{version}.jsonl"
           for name, version in REVIEW_VERSIONS.items()}
NOTES = tuple(REVIEW / f"us-address-long-v1-review-{version}-uncertain.json"
              for version in REVIEW_VERSIONS.values())
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
RESOLUTIONS = REVIEW / "us-address-long-v1-resolutions.json"
OUT = ROOT / "data/interim/silver/us-address-long-reviewed-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
SOURCE_EXCLUSIONS = {
    "2026-08096": "broken HTML in captured source text",
    "2025-21350": "predominantly foreign address roster outside this US prose batch",
}


def digest(path):
    """Hash exact saved file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_key(row):
    """Compare complete annotation sets, including every occurrence and boundary."""
    return tuple(sorted((span["kind"], span["start"], span["end"], span["text"])
                        for span in row["expected"]))


def read_review(path, packet, expected_ids):
    """Require unchanged sources, complete membership and valid UTF-8 spans."""
    reviewed = rows(path)
    mapped = {row["source_id"]: row for row in reviewed}
    if len(mapped) != len(reviewed) or set(mapped) != expected_ids:
        raise ValueError(f"review membership changed: {path.name}")
    for source_id, row in mapped.items():
        if {key: value for key, value in row.items() if key != "expected"} != packet[source_id]:
            raise ValueError(f"review changed blind source: {source_id}")
        validate_spans(row, row["expected"])
    return mapped


def silver_rows():
    """Accept explicitly resolved rows supported by two complete independent passes."""
    manifest = json.loads(PACKET_MANIFEST.read_text())
    if (manifest["packet_sha256"] != PACKET_SHA256 or digest(PACKET) != PACKET_SHA256 or
            manifest["kind"] != "us_address_long_blind_v1" or
            manifest["label_status"] != "unlabeled" or manifest["cases"] != 23 or
            manifest["training_eligible"] or manifest["evaluation_eligible"]):
        raise ValueError("long address blind packet changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"long address screening input changed: {name}")
    source_rows = rows(PACKET)
    packet = {row["source_id"]: row for row in source_rows}
    if len(packet) != len(source_rows) or len(packet) != 23:
        raise ValueError("long address packet repeats or omits sources")
    expected_ids = set(packet) - SOURCE_EXCLUSIONS.keys()
    reviews = {name: read_review(path, packet, expected_ids)
               for name, path in REVIEWS.items()}
    decisions = json.loads(RESOLUTIONS.read_text())
    if (decisions["packet_sha256"] != digest(PACKET) or
            decisions["policy_sha256"] != digest(POLICY) or
            decisions["review_sha256"] != {name: digest(path) for name, path in REVIEWS.items()}):
        raise ValueError("address decisions refer to different packet, policy or reviews")
    accepted, excluded = decisions["accepted"], decisions["excluded"]
    if (set(accepted) & set(excluded) or set(accepted) | set(excluded) != expected_ids or
            not accepted):
        raise ValueError("address decisions do not cover the reviewed packet")
    selected = []
    for source in source_rows:
        source_id = source["source_id"]
        if source_id not in accepted:
            continue
        decision = accepted[source_id]
        if not decision.get("reason", "").strip() or decision["review"] not in reviews:
            raise ValueError(f"missing address review decision: {source_id}")
        chosen = reviews[decision["review"]][source_id]
        agreement = sum(label_key(review[source_id]) == label_key(chosen)
                        for review in reviews.values())
        if agreement < 2:
            raise ValueError(f"no complete independent agreement: {source_id}")
        if len(chosen["input"].split()) < 200:
            raise ValueError(f"address passage is not long: {source_id}")
        if not any(span["kind"] == "address" for span in chosen["expected"]):
            raise ValueError(f"address passage has no reviewed address: {source_id}")
        selected.append(chosen)
    if any(not reason.strip() for reason in excluded.values()):
        raise ValueError("missing reason for excluded address labels")
    hits = heldout_reasons(selected, [row for path in EVALUATION for row in rows(path)])
    if hits:
        raise ValueError(f"address labels overlap reserved evaluation: {hits}")
    silver = []
    for row in selected:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected", "training_eligible",
                                   "evaluation_eligible"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "entities": [{key: span[key] for key in ("kind", "start", "end")}
                                    for span in row["expected"]]})
    if len({document_key(row) for row in silver}) != len(silver):
        raise ValueError("reviewed addresses repeat a source document")
    return silver, excluded


def main():
    """Pin reviewed labels and replay identical bytes on subsequent runs."""
    silver, excluded = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in silver).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS.values(), *NOTES, RESOLUTIONS, POLICY,
              *EVALUATION, Path(__file__),
              ROOT / "bench/us/check_us_blind_packet_holdout.py")
    manifest = {
        "kind": "us_address_long_reviewed_v1",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(silver),
        "source_documents": len(silver),
        "spans_by_kind": dict(sorted(collections.Counter(
            span["kind"] for row in silver for span in row["entities"]).items())),
        "source_quality_exclusions": SOURCE_EXCLUSIONS,
        "unresolved_labels_excluded": excluded,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("address silver or manifest is missing")
    if OUT.exists() and (OUT.read_bytes() != data or MANIFEST.read_bytes() != encoded):
        raise ValueError("frozen address silver or its inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded)
    print(f"{len(silver)} long address passages frozen; {manifest['spans_by_kind']}")


if __name__ == "__main__":
    main()
