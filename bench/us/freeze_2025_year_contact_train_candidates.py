"""Gate reviewed full-year contacts against the strict US evaluation set."""

import collections
import json
from pathlib import Path

from build_2025_year_contact_gold import MANIFEST as GOLD_MANIFEST
from build_2025_year_contact_gold import OUT as GOLD
from build_full_notice_gold import digest, read_rows
from build_silver import family
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import (
    EVALUATION, EXCLUDED, contact_surfaces, holdout_reserved_names, street_pairs,
)


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_OUT = ROOT / "data/interim/silver/r24/us-2025-year-contact-train-candidate-v5.jsonl"
OUT = ROOT / "data/interim/silver/r24/us-2025-year-contact-train-candidate-v6.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ADDITIONAL_EXCLUDED = {
    "us-2025-contact-2025-22725-1": "organizational acronym variant of strict evaluation",
    "us-2025-contact-2025-02229-1": "organizational long-form variant of strict evaluation",
    "us-2025-contact-2025-02510-1": "organizational unit variant of strict evaluation",
    "us-2025-contact-2025-09357-2": "person appears in synthetic detector validation",
}


def candidate_rows():
    gold_bytes = GOLD.read_bytes()
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    if gold_manifest["sha256"] != digest(gold_bytes) or gold_manifest["training_eligible"]:
        raise ValueError("reviewed full-year contact gold changed")
    rows = read_rows(GOLD)
    names = {row["name"] for row in rows}
    excluded = {name: reason for name, reason in (EXCLUDED | ADDITIONAL_EXCLUDED).items()
                if name in names}
    if len(names) != len(rows) or len(excluded) != 11:
        raise ValueError("full-year contact exclusions changed")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    reserved = holdout_reserved_names()
    eval_sources = {row["source_id"] for row in evaluation if row.get("source_id")}
    eval_groups = {family(row["source_url"]) for row in evaluation
                   if row.get("source_url")}
    candidates = [row for row in rows
                  if row["name"] not in excluded and row["name"] not in reserved]
    if any(row["source_id"] in eval_sources or row["source_group"] in eval_groups
           for row in candidates):
        raise ValueError("full-year contact source overlaps evaluation")
    hits = collisions(evaluation,
                      ((row["name"], row["input"], row["expected"])
                       for row in candidates))
    if hits:
        raise ValueError(f"full-year contact labels overlap evaluation: {sorted(hits)}")
    shared_streets = street_pairs(candidates) & street_pairs(evaluation)
    if shared_streets:
        raise ValueError(f"full-year contact streets overlap evaluation: {sorted(shared_streets)}")
    for kind in ("email", "phone"):
        shared = contact_surfaces(candidates, kind) & contact_surfaces(evaluation, kind)
        if shared - ({"711"} if kind == "phone" else set()):
            raise ValueError(f"full-year contact {kind} overlaps evaluation")
    return candidates, excluded


def main():
    rows, excluded = candidate_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha or previous_manifest["training_eligible"]:
        raise ValueError("previous full-year contact candidates changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    groups = collections.Counter(row["source_group"] for row in rows)
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_year_contact_train_candidate_v6",
        "supersedes_sha256": previous_sha,
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in EVALUATION},
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "training_candidate_after_input_integration_gate",
        "excluded": excluded,
        "holdout_reserved_cases": len(holdout_reserved_names()),
        "cases": len(rows),
        "entity_counts": dict(sorted(counts.items())),
        "repeated_source_groups": sum(count > 1 for count in groups.values()),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen year contact candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen year contact candidates changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen year contact candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed full-year contacts pass strict evaluation split: {digest(data)}")


if __name__ == "__main__":
    main()
