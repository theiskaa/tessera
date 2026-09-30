"""Remove a referent overlap missed by the historical US V4 alias screen."""

import hashlib
import json
from pathlib import Path

from check_holdout_overlap import collisions


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/interim/silver/us-federal-org-reviewed-safe-v1.jsonl"
SOURCE_MANIFEST = SOURCE.with_suffix(".manifest.json")
EVALUATION = tuple(ROOT / "data/interim/review" / name for name in (
    "us-eval-exclusions-v4.jsonl",
    "us-long-org-eval-gold-v4.jsonl",
    "us-doe-org-overviews-strict-gold-v1.jsonl",
))
OUT = ROOT / "data/interim/silver/us-federal-org-reviewed-safe-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
KNOWN_OVERLAP = {"raw-us-org-2024-29294-1"}


def digest(data):
    """Hash exact input and output bytes."""
    return hashlib.sha256(data).hexdigest()


def rows(path):
    """Read a frozen JSONL source."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_rows():
    """Retain only reviewed rows disjoint under the expanded alias policy."""
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if source_manifest["sha256"] != digest(SOURCE.read_bytes()):
        raise ValueError("historical V4 organization silver changed")
    original = rows(SOURCE)
    evaluation = [row for path in EVALUATION for row in rows(path)]
    if len(original) != 112:
        raise ValueError("historical V4 federal organization membership changed")
    heldout = collisions(evaluation, ((row["id"], row["text"], row["entities"])
                                   for row in original), strict_aliases=True)
    blocked = {origin for kinds in heldout.values() for values in kinds.values()
               for origin, _ in values}
    if blocked != KNOWN_OVERLAP:
        raise ValueError(f"strict organization overlap changed: {sorted(blocked)}")
    selected = [row for row in original if row["id"] not in blocked]
    if len(selected) != 111 or collisions(
            evaluation, ((row["id"], row["text"], row["entities"])
                         for row in selected), strict_aliases=True):
        raise ValueError("strict organization silver is not evaluation-disjoint")
    return selected, heldout


def main():
    """Freeze a safe replacement without changing the historical V4 source."""
    selected, heldout = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    inputs = (SOURCE, SOURCE_MANIFEST, *EVALUATION,
              ROOT / "bench/us/org_aliases.py",
              ROOT / "bench/us/check_holdout_overlap.py")
    manifest = {
        "kind": "us_federal_org_reviewed_safe_v2_strict",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(selected),
        "excluded_ids": sorted(KNOWN_OVERLAP),
        "affected_evaluation_cases": sorted(heldout),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("strict organization silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("strict organization silver changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("strict organization silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(selected)} federal organization rows frozen; 1 referent overlap removed")


if __name__ == "__main__":
    main()
