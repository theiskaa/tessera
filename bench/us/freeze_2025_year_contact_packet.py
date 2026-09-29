"""Freeze balanced 2025 US contact passages after the month-by-month capture finishes."""

import collections
import json
import os
import tempfile
import time

from address_keys import address_keys_in_text
from build_contact_snippets import lines
from build_silver import family
from freeze_2025_contact_packet import (GOLD, HOLDOUT, HOLDOUT_REVIEWS,
                                        PACKET_DIR, RAW, SILVER, SILVER_SOURCES,
                                        digest, selected_cases)


SNAPSHOT = PACKET_DIR / "us-2025-source-snapshot-v2.json"
OUT = PACKET_DIR / "us-2025-year-contact-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
REVIEWED_2025 = PACKET_DIR.parent.parent / "review/us-2025-contact-gold-v5.jsonl"
EXTRA_TRAIN = (
    PACKET_DIR.parent.parent / "review/us-2024-contact-gold-v3.jsonl",
    PACKET_DIR.parent.parent / "review/us-2024-acronym-prose-gold-v2.jsonl",
)
PRIOR_2025_PACKETS = (
    PACKET_DIR / "us-2025-context-blind-v1.jsonl",
    *(PACKET_DIR / f"us-2025-contact-blind-v{version}.jsonl"
      for version in range(1, 5)),
    PACKET_DIR / "us-2025-contact-train-candidate-v1.jsonl",
    PACKET_DIR / "us-2025-contact-train-candidate-v2.jsonl",
)
MIN_PER_MONTH = 250
MAX_REVIEW_PER_MONTH = 15
MIN_REVIEW_PER_MONTH = 8
CAPTURE_QUIET_SECONDS = 120


def source_snapshot(persist=True):
    if SNAPSHOT.exists():
        snapshot = json.loads(SNAPSHOT.read_text())
        current = {path.stem for path in RAW.glob("2025-*.json")}
        if current != set(snapshot):
            raise ValueError("2025 source set changed after snapshot")
    else:
        files = sorted(RAW.glob("2025-*.json"))
        counts = collections.Counter()
        snapshot = {}
        now = time.time()
        for path in files:
            if now - path.stat().st_mtime < CAPTURE_QUIET_SECONDS:
                raise ValueError("2025 source capture is still writing")
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
            if (raw["id"] != path.stem or raw["source"] != "federal-register" or
                    not raw["date"].startswith("2025-") or
                    not raw["url"].startswith("https://www.federalregister.gov/documents/") or
                    f"/{raw['id']}/" not in raw["url"] or
                    raw["text_url"] !=
                    f"https://www.govinfo.gov/content/pkg/FR-{raw['date']}/html/{raw['id']}.htm" or
                    len(raw["text"]) < 200):
                raise ValueError(f"2025 source metadata differs: {path.name}")
            counts[raw["date"][:7]] += 1
            snapshot[path.stem] = digest(raw_bytes)
        if any(counts[f"2025-{month:02d}"] < MIN_PER_MONTH
               for month in range(1, 13)):
            raise ValueError(f"2025 capture incomplete by month: {dict(sorted(counts.items()))}")
        if persist:
            write_snapshot(snapshot)
    for source_id, expected in snapshot.items():
        if digest((RAW / f"{source_id}.json").read_bytes()) != expected:
            raise ValueError(f"2025 source changed after snapshot: {source_id}")
    return snapshot


def write_snapshot(snapshot):
    data = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=PACKET_DIR, prefix=".2025-source-snapshot-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if SNAPSHOT.exists() and SNAPSHOT.read_text() != data:
            raise ValueError("2025 source snapshot changed")
        if not SNAPSHOT.exists():
            os.replace(temporary, SNAPSHOT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    snapshot = source_snapshot(persist=False)
    reviewed = lines(REVIEWED_2025)
    previous = [row for path in PRIOR_2025_PACKETS for row in lines(path)]
    source_ids = [row["source_id"] for row in reviewed]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source in reviewed 2025 contacts")
    excluded = reviewed + previous
    selected, counts, _ = selected_cases(
        snapshot, extra_gold=reviewed,
        excluded_sources={row["source_id"] for row in excluded},
        excluded_families={family(row["source_url"]) for row in excluded},
        extra_train=(row for path in (REVIEWED_2025, *PRIOR_2025_PACKETS, *EXTRA_TRAIN)
                     for row in lines(path)),
        month_cap=MAX_REVIEW_PER_MONTH, address_screen=address_keys_in_text,
        address_pair_screen=True)
    months = collections.Counter(json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                                 for row in selected)
    if any(value > MAX_REVIEW_PER_MONTH for value in months.values()):
        raise ValueError("contact review packet exceeds a month cap")
    if any(months[f"2025-{month:02d}"] < MIN_REVIEW_PER_MONTH
           for month in range(1, 13)):
        raise ValueError(f"2025 contact selection lacks month coverage: {dict(sorted(months.items()))}")
    if source_snapshot(persist=False) != snapshot:
        raise ValueError("2025 source capture changed during selection")
    write_snapshot(snapshot)
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    manifest = {
        "kind": "us_2025_year_contact_blind_packet",
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
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
        "minimum_capture_per_month": MIN_PER_MONTH,
        "maximum_review_per_month": MAX_REVIEW_PER_MONTH,
        "minimum_review_per_month": MIN_REVIEW_PER_MONTH,
        "review_cases_by_month": dict(sorted(months.items())),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen year contact packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen year contact packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen year contact packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} balanced 2025 contact cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
