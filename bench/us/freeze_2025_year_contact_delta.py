"""Freeze the full-year contact cases absent from the reviewed ten-month packet."""

import json

from build_contact_snippets import lines
from freeze_2025_ready_contact_packet import MANIFEST as READY_MANIFEST
from freeze_2025_ready_contact_packet import OUT as READY_PACKET
from freeze_2025_year_contact_packet import MANIFEST as YEAR_MANIFEST
from freeze_2025_year_contact_packet import OUT as YEAR_PACKET
from freeze_2025_year_contact_packet import PACKET_DIR, RAW, digest


OUT = PACKET_DIR / "us-2025-year-contact-delta-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    ready_manifest = json.loads(READY_MANIFEST.read_text())
    year_manifest = json.loads(YEAR_MANIFEST.read_text())
    if (digest(READY_PACKET.read_bytes()) != ready_manifest["sha256"] or
            digest(YEAR_PACKET.read_bytes()) != year_manifest["sha256"] or
            ready_manifest["training_eligible"] or year_manifest["training_eligible"]):
        raise ValueError("source contact packets changed")
    ready = {row["name"]: row for row in lines(READY_PACKET)}
    year = lines(YEAR_PACKET)
    if len(ready) != ready_manifest["cases"] or len(year) != year_manifest["cases"]:
        raise ValueError("source contact packet case count changed")
    shared = [row for row in year if row["name"] in ready]
    if any(ready[row["name"]] != row for row in shared):
        raise ValueError("previously reviewed contact case changed")
    delta = [row for row in year if row["name"] not in ready]
    for row in delta:
        raw_bytes = (RAW / f"{row['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != row["raw_sha256"] or
                raw["id"] != row["source_id"] or
                raw["url"] != row["source_url"] or
                raw["text"].count(row["input"]) != 1):
            raise ValueError(f"new contact case differs from source: {row['name']}")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in delta).encode()
    manifest = {
        "kind": "us_2025_year_contact_delta_blind_v1",
        "intended_use": "two_blind_reviews_then_full_year_gold",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "ready_packet_sha256": ready_manifest["sha256"],
        "year_packet_sha256": year_manifest["sha256"],
        "reused_reviewed_cases": len(shared),
        "cases": len(delta),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen year contact delta or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen year contact delta changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen year contact delta inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(delta)} new contact cases need blind review; {len(shared)} reuse reviewed labels")


if __name__ == "__main__":
    main()
