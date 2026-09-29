"""Gate reviewed 2025 acronym prose against strict US evaluation names."""

import collections
import json
from pathlib import Path

from build_2025_year_acronym_gold import MANIFEST as GOLD_MANIFEST
from build_2025_year_acronym_gold import OUT as GOLD
from build_2025_year_acronym_gold import gold_rows
from build_full_notice_gold import digest, read_rows
from build_silver import family
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import EVALUATION


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_OUT = ROOT / "data/interim/silver/r24/us-2025-year-acronym-prose-train-candidate-v2.jsonl"
OUT = ROOT / "data/interim/silver/r24/us-2025-year-acronym-prose-train-candidate-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-2025-acronym-prose-2025-02534-396": "ITA aliases strict evaluation International Trade Administration",
    "us-2025-acronym-prose-2025-16644-5476": "AOUSC aliases strict evaluation Administrative Office of the U.S. Courts",
}


def candidate_rows():
    manifest = json.loads(GOLD_MANIFEST.read_text())
    gold = read_rows(GOLD)
    rebuilt, _ = gold_rows()
    if (manifest["sha256"] != digest(GOLD.read_bytes()) or
            manifest["training_eligible"] or gold != rebuilt):
        raise ValueError("reviewed acronym gold changed")
    names = {row["name"] for row in gold}
    if len(names) != len(gold) or set(EXCLUDED) & names or len(gold) != 46:
        raise ValueError("acronym exclusions differ from reviewed gold")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    eval_ids = {row["source_id"] for row in evaluation if row.get("source_id")}
    eval_groups = {family(row["source_url"]) for row in evaluation
                   if row.get("source_url")}
    candidates = gold
    if any(row["source_id"] in eval_ids or row["source_group"] in eval_groups
           for row in candidates):
        raise ValueError("acronym source overlaps evaluation")
    hits = collisions(evaluation,
                      ((row["name"], row["input"], row["expected"])
                       for row in candidates))
    if hits:
        raise ValueError(f"acronym labels overlap evaluation: {sorted(hits)}")
    return candidates


def main():
    rows = candidate_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha or previous_manifest["training_eligible"]:
        raise ValueError("previous acronym candidates changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_year_acronym_prose_train_candidate_v3",
        "supersedes_sha256": previous_sha,
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in EVALUATION},
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "training_candidate_after_input_integration_gate",
        "excluded": EXCLUDED,
        "cases": len(rows), "entity_counts": dict(sorted(counts.items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen acronym candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen acronym candidates changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen acronym candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed acronym cases pass strict evaluation split")


if __name__ == "__main__":
    main()
