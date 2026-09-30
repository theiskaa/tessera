"""Freeze safe copies of US silver with held-out referent aliases removed."""

import hashlib
import json
from pathlib import Path

from check_holdout_overlap import collisions


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
EVALUATION = tuple(ROOT / "data/interim/review" / name for name in (
    "us-eval-exclusions-v4.jsonl",
    "us-long-org-eval-gold-v4.jsonl",
    "us-doe-org-overviews-strict-gold-v1.jsonl",
))
SOURCES = (
    ("us-reviewed-org-paragraphs-v4", "us-reviewed-org-paragraphs-strict-v1", 48,
     ("2024-26472-org-paragraph-3",)),
    ("us-v4-reviewed-additions-v2", "us-v4-reviewed-additions-strict-v1", 113,
     ("us-contact-expansion-2026-06802-1",
      "us-contact-expansion-2026-08844-1",
      "us-contact-expansion-2026-09237-1",
      "us-contact-expansion-2026-10507-1",
      "us-contact-expansion-2026-13868-2",
      "us-contact-expansion-2026-14276-1",
      "us-contact-expansion-2026-17800-2")),
    ("us-v4-reviewed-historical-acronyms-v1",
     "us-v4-reviewed-historical-acronyms-strict-v1", 15,
     ("us-historical-acronym-2025-04960-3881",)),
    ("us-v5-reviewed-long-org-prose-v1",
     "us-v5-reviewed-long-org-prose-strict-v1", 35,
     ("us-long-org-prose-2025-03136-5722",)),
    ("us-reviewed-extra-long-org-prose-v2",
     "us-reviewed-extra-long-org-prose-strict-v1", 29,
     ("us-long-org-prose-2025-02441-2384",
      "us-long-org-prose-2025-05342-3839")),
    ("us-reviewed-two-paragraph-org-prose-v3",
     "us-reviewed-two-paragraph-org-prose-strict-v1", 18,
     ("us-long-org-prose-2024-30692-8284",
      "us-long-org-prose-2025-23176-2774")),
)


def digest(path):
    """Hash exact frozen bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    """Read JSONL rows in their original order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outputs():
    """Derive and validate strict replacements before writing any files."""
    evaluation = [row for path in EVALUATION for row in rows(path)]
    if len(evaluation) != 371:
        raise ValueError("reserved US evaluation changed")
    result = []
    for source_name, output_name, expected_count, expected_blocked in SOURCES:
        source = SILVER / f"{source_name}.jsonl"
        source_manifest = source.with_suffix(".manifest.json")
        manifest = json.loads(source_manifest.read_text())
        if manifest["sha256"] != digest(source):
            raise ValueError(f"source silver changed: {source_name}")
        original = rows(source)
        if len(original) != expected_count:
            raise ValueError(f"source membership changed: {source_name}")
        hits = collisions(
            evaluation, ((row["id"], row["text"], row["entities"])
                         for row in original), strict_aliases=True)
        blocked = {origin for kinds in hits.values() for values in kinds.values()
                   for origin, _ in values}
        if blocked != set(expected_blocked):
            raise ValueError(f"strict overlap changed in {source_name}: {sorted(blocked)}")
        selected = [row for row in original if row["id"] not in blocked]
        if collisions(evaluation, ((row["id"], row["text"], row["entities"])
                                   for row in selected), strict_aliases=True):
            raise ValueError(f"strict replacement still overlaps: {source_name}")
        result.append((source, source_manifest, SILVER / f"{output_name}.jsonl",
                       selected, blocked, hits))
    return result


def main():
    """Write pinned safe replacements without modifying historical silver."""
    results = outputs()
    shared_inputs = (*EVALUATION, ROOT / "bench/us/org_aliases.py",
                     ROOT / "bench/us/check_holdout_overlap.py")
    for source, source_manifest, output, selected, blocked, hits in results:
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                       for row in selected).encode()
        manifest = {
            "kind": "us_strict_referent_safe_silver_v1",
            "training_eligible": True,
            "active_in_training_config": False,
            "source": str(source.relative_to(ROOT)),
            "cases": len(selected),
            "excluded_ids": sorted(blocked),
            "affected_evaluation_cases": sorted(hits),
            "input_sha256": {str(path.relative_to(ROOT)): digest(path)
                             for path in (source, source_manifest, *shared_inputs)},
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() != manifest_path.exists():
            raise ValueError(f"strict silver or manifest missing: {output}")
        if output.exists() and output.read_bytes() != data:
            raise ValueError(f"strict silver changed: {output}")
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_data:
            raise ValueError(f"strict silver inputs changed: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.write_bytes(data)
        if not manifest_path.exists():
            manifest_path.write_bytes(manifest_data)
        print(f"{output.stem}: {len(selected)} kept, {len(blocked)} excluded")


if __name__ == "__main__":
    main()
