"""Freeze reviewed 2026 contacts with no labeled training-surface overlap."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from active_sources import ACTIVE_SILVER
from check_holdout_overlap import (SILVER, SYNTHETIC, collisions, contact_collisions,
                                   current_training_sources)


ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/interim/review/us-2026-contact-gold-v1.jsonl"
GOLD_MANIFEST = GOLD.with_suffix(".manifest.json")
OUT = ROOT / "data/interim/review/us-2026-contact-strict-holdout-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def strict_rows(processed):
    """Select reviewed contacts disjoint from the specified prepared synthetic set."""
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    gold_bytes = GOLD.read_bytes()
    if (gold_manifest["sha256"] != digest(gold_bytes) or
            gold_manifest["training_eligible"] or gold_manifest["cases"] != 40):
        raise ValueError("reviewed 2026 contact gold changed")
    rows = read_rows(GOLD)
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("duplicate 2026 contact gold case")
    for row in rows:
        validate_spans(row["name"], row["input"], row["expected"])
    labeled = collisions(rows, current_training_sources(processed))
    contacts = contact_collisions(rows, current_training_sources(processed))
    rejected = {}
    for row in rows:
        name = row["name"]
        kinds = set(labeled.get(name, {})) | set(contacts.get(name, {}))
        if kinds:
            rejected[name] = sorted(kinds)
    kept = [row for row in rows if row["name"] not in rejected]
    months = collections.Counter(row["month"] for row in kept)
    if (len(kept) < 25 or len(months) != 8 or
            len({row["source_id"] for row in kept}) != len(kept) or
            len({row["source_group"] for row in kept}) != len(kept)):
        raise ValueError("2026 strict contact holdout lost required source diversity")
    return kept, rejected, months, gold_bytes


def main():
    """Pin the strict subset and every prepared input used to screen it."""
    kept, rejected, months, gold_bytes = strict_rows(SYNTHETIC)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in kept).encode()
    input_paths = [*(SILVER / f"{name}.jsonl" for name, _ in ACTIVE_SILVER),
                   SYNTHETIC / "train.parquet", SYNTHETIC / "valid.parquet"]
    manifest = {
        "kind": "us_2026_contact_strict_holdout_v1",
        "cases": len(kept),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in kept for span in row["expected"]).items())),
        "cases_by_month": dict(sorted(months.items())),
        "rejected_training_surface_overlap": dict(sorted(rejected.items())),
        "gold_sha256": digest(gold_bytes),
        "prepared_input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                  for path in input_paths},
        "sha256": digest(data),
        "source_group_key": "source_id",
        "strict_entity_holdout": True,
        "training_eligible": False,
        "intended_use": "source_distinct_diagnostic_only",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("2026 strict holdout or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2026 strict holdout changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2026 strict holdout inputs changed after freezing")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} strict 2026 contacts frozen; {len(rejected)} overlaps excluded")


if __name__ == "__main__":
    main()
