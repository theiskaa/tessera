"""Export independently reviewed, source-backed US contacts as detector silver."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, validate_spans
from freeze_org_rich_contact_train_candidates import (
    MANIFEST as ORG_MANIFEST, OUT as ORG_CANDIDATE,
    PACKET as ORG_PACKET, REVIEW_A as ORG_REVIEW_A,
    REVIEW_B as ORG_REVIEW_B,
    candidate_rows as org_candidate_rows,
)
from freeze_us_error_target_train_candidates import (
    MANIFEST as ERROR_MANIFEST, OUT as ERROR_CANDIDATE,
    PACKET as ERROR_PACKET, REVIEW_A as ERROR_REVIEW_A,
    REVIEW_B as ERROR_REVIEW_B,
    candidate_rows as error_candidate_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/silver/us-r25-reviewed-contacts-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MODEL_KINDS = {"person", "org", "address"}


def silver_rows():
    """Rebuild every label from the two pinned candidate sets."""
    candidates = []
    candidate_hashes = {}
    review_hashes = {}
    evaluation_hashes = None
    for path, manifest_path, packet, review_a, review_b, rebuild in (
            (ORG_CANDIDATE, ORG_MANIFEST, ORG_PACKET,
             ORG_REVIEW_A, ORG_REVIEW_B, org_candidate_rows),
            (ERROR_CANDIDATE, ERROR_MANIFEST, ERROR_PACKET,
             ERROR_REVIEW_A, ERROR_REVIEW_B, error_candidate_rows)):
        manifest = json.loads(manifest_path.read_text())
        rows, _ = rebuild()
        frozen = [json.loads(line) for line in path.read_text().splitlines()]
        if (manifest["training_eligible"] or
                manifest["sha256"] != digest(path.read_bytes()) or
                manifest["packet_sha256"] != digest(packet.read_bytes()) or
                manifest["review_a_sha256"] != digest(review_a.read_bytes()) or
                manifest["review_b_sha256"] != digest(review_b.read_bytes()) or
                frozen != rows or manifest["cases"] != len(rows)):
            raise ValueError(f"reviewed contact candidate changed: {path.name}")
        if evaluation_hashes is None:
            evaluation_hashes = manifest["evaluation_sha256"]
        elif evaluation_hashes != manifest["evaluation_sha256"]:
            raise ValueError("reviewed contact candidates use different evaluation sets")
        candidates.extend(rows)
        candidate_hashes[path.name] = manifest["sha256"]
        review_hashes[path.name] = {
            "a": manifest["review_a_sha256"],
            "b": manifest["review_b_sha256"],
        }
    if len(candidates) != 17 or len({row["name"] for row in candidates}) != 17:
        raise ValueError("reviewed contact membership changed")
    result = []
    for row in candidates:
        spans = validate_spans(row["name"], row["input"], row["expected"])
        result.append({
            "id": row["name"], "source": "federal-register",
            "source_id": row["source_id"], "source_url": row["source_url"],
            "source_group": row["source_group"], "country": "US",
            "text": row["input"],
            "entities": [{"kind": span["kind"], "start": span["start"],
                          "end": span["end"]} for span in spans],
        })
    return result, candidate_hashes, review_hashes, evaluation_hashes


def main():
    """Freeze reviewed rows and their lineage for v3 training inputs."""
    rows, candidate_hashes, review_hashes, evaluation_hashes = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["entities"])
    manifest = {
        "kind": "us_r25_reviewed_contacts_silver_v1",
        "candidate_sha256": candidate_hashes,
        "review_sha256": review_hashes,
        "evaluation_sha256": evaluation_hashes,
        "documents": len(rows),
        "labels": dict(sorted(counts.items())),
        "model_labels": {kind: counts[kind] for kind in sorted(MODEL_KINDS)},
        "rule_labels": {kind: counts[kind] for kind in ("email", "phone")},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("reviewed contact silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("reviewed contact silver changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("reviewed contact inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed contacts with {dict(sorted(counts.items()))} labels")


if __name__ == "__main__":
    main()
