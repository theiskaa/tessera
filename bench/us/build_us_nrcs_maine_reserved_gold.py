"""Freeze four Maine office tables outside training for layout diagnosis."""

import json
from pathlib import Path

from build_full_notice_gold import digest
from freeze_us_nrcs_maine_train_candidates import (
    PACKET, REVIEW_A, REVIEW_B, candidate_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/review/us-nrcs-maine-reserved-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    """Rebuild the office-disjoint reserve from the agreed blind spans."""
    _, reserve, _, _ = candidate_rows()
    return [{
        "name": row["name"], "country": "US",
        "doc_type": "agency_mailing_address", "input": row["input"],
        "expected": row["expected"],
        "source_group": "us-nrcs-maine-directory-v1",
    } for row in reserve]


def main():
    """Pin gold bytes and explicitly prohibit training use."""
    rows = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_nrcs_maine_reserved_gold_v1", "cases": len(rows),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("USDA Maine reserved gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("USDA Maine reserved gold changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("USDA Maine reserved gold inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} USDA Maine offices reserved outside training")


if __name__ == "__main__":
    main()
