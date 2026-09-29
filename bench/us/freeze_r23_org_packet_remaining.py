"""Freeze the remaining source-disjoint US organization paragraphs for blind review."""

import hashlib
import json
from pathlib import Path

from freeze_r23_org_packet import FROZEN_SHA256 as FIRST_SHA256
from freeze_r23_org_packet import OUT as FIRST
from freeze_r23_org_packet import TARGET, selected_cases


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/silver/r23/us-org-blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
FROZEN_SHA256 = "2645f390cd12420bff83030c94d1adae54388606274cbdf1d6ab13379bcde0c3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if digest(FIRST.read_bytes()) != FIRST_SHA256:
        raise ValueError("first blind packet changed")
    selected = selected_cases()[TARGET:]
    if not selected:
        raise ValueError("no remaining source-disjoint paragraphs")
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError(f"second blind packet differs: {digest(data)}")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("second blind packet changed")
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_r23_blind_org_packet",
                                    "first_packet_sha256": FIRST_SHA256,
                                    "cases": len(selected), "sha256": digest(data)},
                                   indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} remaining blind US paragraphs frozen: {digest(data)}")


if __name__ == "__main__":
    main()
