"""Freeze 2025 contact passages from months with sufficient captured coverage."""

import collections
import json
import os
import tempfile

from build_contact_snippets import lines
from build_silver import family
from freeze_2025_contact_packet import GOLD, HOLDOUT, HOLDOUT_REVIEWS, RAW, SILVER, SILVER_SOURCES, digest, selected_cases
from freeze_2025_year_contact_packet import (CAPTURE_QUIET_SECONDS, EXTRA_TRAIN,
                                             MAX_REVIEW_PER_MONTH,
                                             MIN_REVIEW_PER_MONTH, PACKET_DIR,
                                             PRIOR_2025_PACKETS, REVIEWED_2025,
                                             source_snapshot)


MONTHS = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)
MONTH_KEYS = {f"2025-{month:02d}" for month in MONTHS}
SNAPSHOT = PACKET_DIR / "us-2025-ready-contact-source-snapshot-v1.json"
OUT = PACKET_DIR / "us-2025-ready-contact-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen packet changed: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    full_snapshot = source_snapshot(persist=False)
    snapshot = {source_id: sha for source_id, sha in full_snapshot.items()
                if json.loads((RAW / f"{source_id}.json").read_text())["date"][:7]
                in MONTH_KEYS}
    reviewed = lines(REVIEWED_2025)
    previous = [row for path in PRIOR_2025_PACKETS for row in lines(path)]
    excluded = reviewed + previous
    selected, counts, _ = selected_cases(
        snapshot, extra_gold=reviewed,
        excluded_sources={row["source_id"] for row in excluded},
        excluded_families={family(row["source_url"]) for row in excluded},
        extra_train=(row for path in (REVIEWED_2025, *PRIOR_2025_PACKETS,
                                     *EXTRA_TRAIN) for row in lines(path)),
        month_cap=MAX_REVIEW_PER_MONTH)
    months = collections.Counter(json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                                 for row in selected)
    if any(months[month] < MIN_REVIEW_PER_MONTH for month in MONTH_KEYS):
        raise ValueError(f"ready contact selection lacks month coverage: {dict(sorted(months.items()))}")
    if source_snapshot(persist=False) != full_snapshot:
        raise ValueError("2025 source capture changed during ready packet selection")
    snapshot_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    manifest = {
        "kind": "us_2025_ready_month_contact_blind_packet",
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "months": sorted(MONTH_KEYS),
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(snapshot_bytes),
        "reviewed_2025_sha256": digest(REVIEWED_2025.read_bytes()),
        "prior_2025_sha256": {path.name: digest(path.read_bytes())
                              for path in PRIOR_2025_PACKETS},
        "extra_train_sha256": {path.name: digest(path.read_bytes())
                               for path in EXTRA_TRAIN},
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in GOLD},
        "reserved_sha256": {path.name: digest(path.read_bytes())
                            for path in (HOLDOUT, *HOLDOUT_REVIEWS)},
        "source_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                          for name in SILVER_SOURCES},
        "capture_count": len(snapshot),
        "minimum_review_per_month": MIN_REVIEW_PER_MONTH,
        "maximum_review_per_month": MAX_REVIEW_PER_MONTH,
        "capture_quiet_seconds": CAPTURE_QUIET_SECONDS,
        "review_cases_by_month": dict(sorted(months.items())),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    for path, content in ((SNAPSHOT, snapshot_bytes), (OUT, data),
                          (MANIFEST, manifest_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"frozen packet changed: {path}")
    write_once(SNAPSHOT, snapshot_bytes)
    write_once(OUT, data)
    write_once(MANIFEST, manifest_bytes)
    print(f"{len(selected)} ten-month contact cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
