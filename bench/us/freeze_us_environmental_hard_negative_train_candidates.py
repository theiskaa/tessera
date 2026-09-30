"""Freeze agreed environmental title and role negatives without source leakage."""

import collections
import json
from pathlib import Path

from active_sources import PRIOR_DOL_REMAINING_SILVER
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_environmental_hard_negative_packet import (
    MANIFEST as PACKET_MANIFEST, OUT as PACKET, packet_rows,
)
from screen_org_rich_contact_candidates import STRICT
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW_A = ROOT / "data/interim/review/us-environmental-hard-negative-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-environmental-hard-negative-review-b-v1.jsonl"
OUT = ROOT / "data/interim/silver/r26/us-environmental-hard-negative-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-environmental-negative-2024-27844": "reviewer uncertain about address detail",
    "us-environmental-negative-2026-05489": "excerpt starts mid-sentence",
    "us-environmental-negative-2026-15270": "BLM organization overlaps evaluation",
}


def candidate_rows():
    """Rebuild only agreed, complete snippets from source and blind reviews."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled" or
            packet != packet_rows()):
        raise ValueError("environmental packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names or
            not set(EXCLUDED) <= set(names)):
        raise ValueError("environmental reviews do not cover packet")
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    evaluation_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in evaluation_paths}
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_DOL_REMAINING_SILVER]
    prior_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                    for path in active_paths}
    if (len(evaluation) != 339 or
            packet_manifest["evaluation_sha256"] != evaluation_hashes or
            packet_manifest["active_silver_sha256"] != prior_hashes):
        raise ValueError("environmental screening inputs changed")
    near = NearTextIndex()
    for row in evaluation:
        near.add(row["name"], row["input"])
    for path in active_paths:
        for row in lines(path):
            near.add(row["id"], row["text"])
    result = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if name in EXCLUDED:
            continue
        if a["uncertain"] or b["uncertain"] or a_spans != b_spans:
            raise ValueError(f"unresolved environmental labels: {name}")
        if near.prior(source["input"]):
            raise ValueError(f"near-duplicate environmental snippet: {name}")
        result.append({
            "name": name, "country": "US", "doc_type": "notice_prose",
            "input": source["input"], "expected": a_spans,
            "source_id": source["source_id"], "source_url": source["source_url"],
            "source_path": source["source_path"],
            "source_start": source["source_start"],
            "source_end": source["source_end"],
            "raw_sha256": source["raw_sha256"],
        })
        near.add(name, source["input"])
    if len(result) != 5:
        raise ValueError("environmental training membership changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in result]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("environmental labels overlap evaluation")
    return result, evaluation_hashes, prior_hashes


def main():
    """Pin training candidate bytes and reviewer lineage."""
    rows, evaluation_hashes, prior_hashes = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_environmental_hard_negative_candidate_v1",
        "cases": len(rows), "excluded": EXCLUDED,
        "labels": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": evaluation_hashes,
        "prior_silver_sha256": prior_hashes,
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("environmental candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("environmental candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("environmental candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed environmental snippets pass the split gate")


if __name__ == "__main__":
    main()
