"""Exclude long evaluation names seen in prepared synthetic training or validation."""

import collections
import json

from build_us_full_eval_exclusions import digest
from check_holdout_overlap import collisions, contact_collisions, synthetic_sources
from freeze_2024_2025_long_org_eval_gold_v3 import MANIFEST as PREVIOUS_MANIFEST
from freeze_2024_2025_long_org_eval_gold_v3 import OUT as PREVIOUS_GOLD
from freeze_2024_2025_long_org_eval_gold_v3 import ROOT
from freeze_2024_2025_long_org_eval_gold_v3 import main as verify_previous_gold
from source_identity import document_key


OUT = ROOT / "data/interim/review/us-long-org-eval-gold-v4.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
SYNTHETIC = ROOT / "data/processed/detector-us-v4"
EXCLUDED = {
    "2025-01760": "CFPB is also a labeled organization in prepared synthetic training or validation.",
    "2025-21130": "McNamara and Moore are also labeled people in prepared synthetic training or validation.",
}


def main():
    """Pin evaluation rows with no labeled-name overlap against prepared V4 training."""
    rows = verify_previous_gold()
    removed = {row["source_id"] for row in rows if row["source_id"] in EXCLUDED}
    kept = [row for row in rows if row["source_id"] not in EXCLUDED]
    if (len(rows) != 17 or len(kept) != 15 or removed != set(EXCLUDED) or
            len({document_key(row) for row in kept}) != len(kept)):
        raise ValueError("long evaluation synthetic exclusions changed")
    if (collisions(kept, synthetic_sources(SYNTHETIC)) or
            contact_collisions(kept, synthetic_sources(SYNTHETIC))):
        raise ValueError("long evaluation labels overlap prepared synthetic data")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in kept).encode()
    inputs = (PREVIOUS_GOLD, PREVIOUS_MANIFEST,
              SYNTHETIC / "train.parquet", SYNTHETIC / "valid.parquet")
    manifest = {
        "kind": "us_long_org_eval_gold_v4",
        "training_eligible": False,
        "cases": len(kept),
        "source_documents": len({document_key(row) for row in kept}),
        "excluded_synthetic_name_overlap": EXCLUDED,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in kept for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long evaluation gold v4 or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long evaluation gold v4 changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long evaluation gold v4 inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} synthetic-disjoint long US evaluation cases frozen")
    return kept


if __name__ == "__main__":
    main()
