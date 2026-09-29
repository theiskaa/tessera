"""Freeze agency-action prose from unused 2025 notice families for blind review."""

import collections
import json

from address_keys import address_keys_in_text
from build_contact_snippets import lines
from build_silver import family
from freeze_2024_acronym_packet import (GOLD, HOLDOUT, HOLDOUT_REVIEWS, RAW,
                                        REVIEW, SILVER, SILVER_SOURCES,
                                        digest, selected_cases)
from freeze_2025_year_contact_packet import (PACKET_DIR, PRIOR_2025_PACKETS,
                                             SNAPSHOT, source_snapshot)
from freeze_2025_ready_contact_packet import OUT as READY_CONTACT_PACKET


CONTACT_PACKET = PACKET_DIR / "us-2025-year-contact-blind-v1.jsonl"
REVIEWED_2025 = REVIEW / "us-2025-contact-gold-v5.jsonl"
CROSS_YEAR_FAMILIES = (
    PACKET_DIR / "us-2024-contact-blind-v1.jsonl",
    PACKET_DIR / "us-2024-contact-blind-v2.jsonl",
    REVIEW / "us-2024-contact-gold-v3.jsonl",
    PACKET_DIR / "us-2024-acronym-prose-blind-v1.jsonl",
    PACKET_DIR / "us-2024-acronym-prose-blind-v2.jsonl",
    REVIEW / "us-2024-acronym-prose-gold-v2.jsonl",
    *(SILVER / f"{name}.jsonl" for name in SILVER_SOURCES),
)
EXTRA_TRAIN = (
    REVIEW / "us-2024-contact-gold-v3.jsonl",
    REVIEW / "us-2024-acronym-prose-gold-v2.jsonl",
    CONTACT_PACKET,
    READY_CONTACT_PACKET,
    *PRIOR_2025_PACKETS,
)
QUALITY_EXCLUSIONS = {
    "2025-02626": "work is a noun after the acronym",
    "2025-22898": "passive work attribution in an unrelated multi-clause sentence",
}
PREVIOUS_OUT = PACKET_DIR / "us-2025-year-acronym-prose-blind-v3.jsonl"
PREVIOUS_MANIFEST = PREVIOUS_OUT.with_suffix(".manifest.json")
OUT = PACKET_DIR / "us-2025-year-acronym-prose-blind-v4.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MAX_REVIEW_PER_MONTH = 10
MAX_PER_ACRONYM = 2


def main():
    previous = json.loads(PREVIOUS_MANIFEST.read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_sha != previous["sha256"] or previous["training_eligible"]:
        raise ValueError("original 2025 acronym packet changed")
    snapshot = source_snapshot()
    reviewed = lines(REVIEWED_2025)
    contacts = lines(CONTACT_PACKET)
    excluded = reviewed + contacts + lines(READY_CONTACT_PACKET) + [row for path in PRIOR_2025_PACKETS
                                      for row in lines(path)]
    source_ids = {row["source_id"] for row in excluded}
    families = {family(row["source_url"]) for row in excluded}
    if len({row["source_id"] for row in reviewed}) != len(reviewed):
        raise ValueError("reviewed 2025 contacts repeat a source")
    selected, counts, input_hashes, _ = selected_cases(
        snapshot, year=2025, extra_gold=reviewed,
        extra_train=(row for path in EXTRA_TRAIN for row in lines(path)),
        excluded_sources=source_ids | set(QUALITY_EXCLUSIONS), excluded_families=families,
        month_cap=MAX_REVIEW_PER_MONTH, max_acronym_uses=MAX_PER_ACRONYM,
        used_paths=(*PRIOR_2025_PACKETS, READY_CONTACT_PACKET),
        strict_expansions=True,
        address_screen=address_keys_in_text, address_pair_screen=True,
        family_paths=CROSS_YEAR_FAMILIES,
        ignored_paths=(SILVER / "us-2025-year-acronym-prose-v1.jsonl",
                       SILVER / "us-2025-year-acronym-prose-v2.jsonl"))
    if (any(row["source_id"] in source_ids or row["source_group"] in families
            for row in selected) or
            len({row["source_id"] for row in selected}) != len(selected) or
            len({row["source_group"] for row in selected}) != len(selected)):
        raise ValueError("acronym packet overlaps a reviewed contact source family")
    months = collections.Counter(json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                                 for row in selected)
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    manifest = {
        "kind": "us_2025_year_acronym_prose_blind_packet",
        "version": 4,
        "supersedes_sha256": previous_sha,
        "quality_exclusions": QUALITY_EXCLUSIONS,
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_group",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "contact_packet_sha256": digest(CONTACT_PACKET.read_bytes()),
        "reviewed_2025_sha256": digest(REVIEWED_2025.read_bytes()),
        "extra_train_sha256": {path.name: digest(path.read_bytes())
                               for path in EXTRA_TRAIN},
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in GOLD},
        "reserved_sha256": {path.name: digest(path.read_bytes())
                            for path in (HOLDOUT, *HOLDOUT_REVIEWS)},
        "input_sha256": input_hashes,
        "cross_year_family_sha256": {path.name: digest(path.read_bytes())
                                     for path in CROSS_YEAR_FAMILIES},
        "source_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                          for name in SILVER_SOURCES},
        "capture_count": len(snapshot),
        "maximum_review_per_month": MAX_REVIEW_PER_MONTH,
        "maximum_review_per_acronym": MAX_PER_ACRONYM,
        "review_cases_by_month": dict(sorted(months.items())),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen year acronym packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen year acronym packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen year acronym packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} balanced 2025 acronym prose cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
