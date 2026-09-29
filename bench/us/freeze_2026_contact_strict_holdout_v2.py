"""Freeze reviewed 2026 contacts against the coherent US synthetic corpus."""

import collections
import json
from pathlib import Path

from active_sources import ACTIVE_SILVER
from build_full_notice_gold import digest, read_rows
from check_holdout_overlap import CURRENT_SYNTHETIC, SILVER
from freeze_2026_contact_strict_holdout import (
    MANIFEST as PRIOR_MANIFEST,
    OUT as PRIOR,
    strict_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/review/us-2026-contact-strict-holdout-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
GENERATOR = ROOT / "data/manifests/detector-synthetic.json"


def main():
    """Pin the same reviewed cases to the new synthetic training inputs."""
    prior_manifest = json.loads(PRIOR_MANIFEST.read_text())
    if (prior_manifest["sha256"] != digest(PRIOR.read_bytes()) or
            not prior_manifest["strict_entity_holdout"] or
            prior_manifest["training_eligible"]):
        raise ValueError("prior 2026 strict holdout changed")
    generator = json.loads(GENERATOR.read_text())
    if (generator.get("version") != 2 or
            generator.get("template_policy") !=
            "exclude_unverified_us_unit_parent_v1"):
        raise ValueError("US synthetic coherence policy is not active")
    kept, rejected, months, gold_bytes = strict_rows(CURRENT_SYNTHETIC)
    if [row["name"] for row in kept] != [row["name"] for row in read_rows(PRIOR)]:
        raise ValueError("2026 strict holdout membership changed under v2 inputs")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in kept).encode()
    input_paths = [*(SILVER / f"{name}.jsonl" for name, _ in ACTIVE_SILVER),
                   CURRENT_SYNTHETIC / "train.parquet",
                   CURRENT_SYNTHETIC / "valid.parquet"]
    manifest = {
        "kind": "us_2026_contact_strict_holdout_v2",
        "cases": len(kept),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in kept for span in row["expected"]).items())),
        "cases_by_month": dict(sorted(months.items())),
        "rejected_training_surface_overlap": dict(sorted(rejected.items())),
        "gold_sha256": digest(gold_bytes),
        "prepared_input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                  for path in input_paths},
        "generator_manifest_sha256": digest(GENERATOR.read_bytes()),
        "previous_holdout_sha256": digest(PRIOR.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_id",
        "strict_entity_holdout": True,
        "training_eligible": False,
        "intended_use": "source_distinct_diagnostic_only",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("2026 v2 holdout or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2026 v2 holdout changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2026 v2 holdout inputs changed after freezing")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} strict 2026 contacts frozen against v2 training inputs")


if __name__ == "__main__":
    main()
