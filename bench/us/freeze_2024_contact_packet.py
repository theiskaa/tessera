"""Freeze complete contact paragraphs from unused 2024 Federal Register notices."""

import json
import re

from build_contact_snippets import lines
from freeze_2025_contact_packet import (GOLD, PACKET_DIR, RAW, SILVER,
                                        SILVER_SOURCES, digest, selected_cases)
from freeze_full_notice_packet import source_ids


SNAPSHOT = PACKET_DIR / "us-2024-source-snapshot-v1.json"
OUT = PACKET_DIR / "us-2024-contact-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
NEW_HOLDOUT = PACKET_DIR.parent.parent / "review/us-2025-contact-holdout-v1.jsonl"
OLD_HOLDOUT = SILVER / "r23/us-full-notice-blind-v1.jsonl"


def source_snapshot():
    if SNAPSHOT.exists():
        snapshot = json.loads(SNAPSHOT.read_text())
    else:
        snapshot = {path.stem: digest(path.read_bytes())
                    for path in sorted(RAW.glob("2024-*.json"))}
        if not snapshot:
            raise ValueError("no 2024 Federal Register source capture")
        PACKET_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    for source_id, expected in snapshot.items():
        path = RAW / f"{source_id}.json"
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"2024 source changed: {source_id}")
    return snapshot


def excluded_source_ids():
    used, _ = source_ids()
    for name in SILVER_SOURCES:
        for row in lines(SILVER / f"{name}.jsonl"):
            match = re.search(r"2024-\d{5}", row["id"])
            if match:
                used.add(match.group())
    for path in (*GOLD, OLD_HOLDOUT, NEW_HOLDOUT):
        for row in lines(path):
            if row.get("source_id", "").startswith("2024-"):
                used.add(row["source_id"])
            match = re.search(r"2024-\d{5}", row["name"])
            if match:
                used.add(match.group())
    return used


def main():
    snapshot = source_snapshot()
    used = excluded_source_ids()
    selected, counts, _ = selected_cases(
        snapshot, year=2024, extra_gold=lines(NEW_HOLDOUT), excluded_sources=used)
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2024 contact packet changed")
    manifest = {
        "kind": "us_2024_contact_blind_packet",
        "intended_use": "training_candidate_after_two_blind_reviews",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in (*GOLD, OLD_HOLDOUT, NEW_HOLDOUT)},
        "source_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                          for name in SILVER_SOURCES},
        "capture_count": len(snapshot),
        "excluded_source_count": len(used),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2024 contact packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind 2024 contact paragraphs frozen: {digest(data)}")


if __name__ == "__main__":
    main()
