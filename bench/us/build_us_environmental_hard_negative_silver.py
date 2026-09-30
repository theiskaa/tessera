"""Export reviewed environmental prose negatives as US silver."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, validate_spans
from freeze_us_environmental_hard_negative_train_candidates import (
    MANIFEST as CANDIDATE_MANIFEST, OUT as CANDIDATE,
    PACKET, REVIEW_A, REVIEW_B, candidate_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/silver/us-environmental-hard-negatives-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MODEL_KINDS = {"person", "org", "address"}


def silver_rows():
    """Rebuild negative and positive spans from the pinned candidate."""
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    candidates, evaluation_hashes, prior_hashes = candidate_rows()
    frozen = [json.loads(line) for line in CANDIDATE.read_text().splitlines()]
    if (candidate_manifest["sha256"] != digest(CANDIDATE.read_bytes()) or
            candidate_manifest["training_eligible"] or frozen != candidates or
            candidate_manifest["cases"] != len(candidates) or
            candidate_manifest["packet_sha256"] != digest(PACKET.read_bytes()) or
            candidate_manifest["review_a_sha256"] != digest(REVIEW_A.read_bytes()) or
            candidate_manifest["review_b_sha256"] != digest(REVIEW_B.read_bytes()) or
            candidate_manifest["evaluation_sha256"] != evaluation_hashes or
            candidate_manifest["prior_silver_sha256"] != prior_hashes):
        raise ValueError("environmental candidate changed")
    rows = []
    for row in candidates:
        expected = validate_spans(row["name"], row["input"], row["expected"])
        rows.append({
            "id": row["name"], "source": "federal-register",
            "source_id": row["source_id"],
            "source_url": row["source_url"],
            "source_group": row["source_id"],
            "source_path": row["source_path"],
            "source_start": row["source_start"],
            "source_end": row["source_end"],
            "raw_sha256": row["raw_sha256"],
            "country": "US", "text": row["input"],
            "entities": [{"kind": span["kind"], "start": span["start"],
                          "end": span["end"]} for span in expected],
        })
    return rows, candidate_manifest, evaluation_hashes, prior_hashes


def main():
    """Pin reviewed negative examples and their source lineage."""
    rows, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["entities"])
    manifest = {
        "kind": "us_environmental_hard_negatives_silver_v1",
        "candidate_sha256": candidate_manifest["sha256"],
        "review_a_sha256": candidate_manifest["review_a_sha256"],
        "review_b_sha256": candidate_manifest["review_b_sha256"],
        "evaluation_sha256": evaluation_hashes,
        "prior_silver_sha256": prior_hashes,
        "documents": len(rows),
        "labels": dict(sorted(counts.items())),
        "model_labels": {kind: counts[kind] for kind in sorted(MODEL_KINDS)
                         if counts[kind]},
        "rule_labels": {kind: counts[kind] for kind in ("email", "phone")},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("environmental silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("environmental silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("environmental silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} environmental snippets with {dict(sorted(counts.items()))} labels")


if __name__ == "__main__":
    main()
