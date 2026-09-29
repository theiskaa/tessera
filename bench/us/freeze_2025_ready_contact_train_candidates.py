"""Gate reviewed 2025 contact examples against the strict US evaluation set."""

import collections
import json
import re
from pathlib import Path

from address_keys import address_keys_in_text
from build_2025_ready_contact_gold import MANIFEST as GOLD_MANIFEST
from build_2025_ready_contact_gold import OUT as GOLD
from build_full_notice_gold import digest, read_rows
from build_silver import family
from check_holdout_overlap import collisions


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
PREVIOUS_OUT = ROOT / "data/interim/silver/r24/us-2025-ready-contact-train-candidate-v4.jsonl"
OUT = ROOT / "data/interim/silver/r24/us-2025-ready-contact-train-candidate-v5.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
YEAR_HOLDOUT = REVIEW / "us-2025-year-contact-holdout-v2.jsonl"
EVALUATION = (REVIEW / "us-eval-exclusions-v1.jsonl",
              REVIEW / "us-2025-contact-holdout-v2.jsonl",
              YEAR_HOLDOUT)
EXCLUDED = {
    "us-2025-contact-2025-01447-2": "unverified organizational routing code",
    "us-2025-contact-2025-07423-1": "source prints an extra-digit phone number",
    "us-2025-contact-2025-14408-2": "unverified combined organizational routing code",
    "us-2025-contact-2025-05441-1": "same numbered street as strict evaluation",
    "us-2025-contact-2025-09472-1": "same numbered street as strict evaluation despite a different ZIP",
    "us-2025-contact-2025-11815-2": "same numbered street as strict evaluation despite a different ZIP",
    "us-2025-contact-2025-11759-2": "organizational punctuation variant of strict evaluation",
    "us-2025-contact-2025-19298-1": "organizational prefix variant of strict evaluation",
    "us-2025-contact-2025-09554-1": "organizational acronym variant of strict evaluation",
    "us-2025-contact-2025-07318-1": "organizational long-form variant of strict evaluation",
}


def street_pairs(rows):
    return {key[1:] for row in rows for span in row["expected"]
            if span["kind"] == "address"
            for key in address_keys_in_text(span["text"])}


def contact_surfaces(rows, kind):
    if kind == "email":
        return {span["text"].casefold() for row in rows for span in row["expected"]
                if span["kind"] == kind}
    return {re.sub(r"\D", "", span["text"]) for row in rows
            for span in row["expected"] if span["kind"] == kind}


def holdout_reserved_names():
    manifest = json.loads(YEAR_HOLDOUT.with_suffix(".manifest.json").read_text())
    holdout = read_rows(YEAR_HOLDOUT)
    if (manifest["sha256"] != digest(YEAR_HOLDOUT.read_bytes()) or
            manifest["training_eligible"] or not manifest["strict_entity_holdout"] or
            manifest["cases"] != len(holdout)):
        raise ValueError("strict 2025 contact holdout changed")
    held_names = {row["name"] for row in holdout}
    reserved = {name for group in manifest["reserved_groups"] for name in group}
    if len(held_names) != len(holdout) or not held_names <= reserved:
        raise ValueError("strict 2025 contact holdout groups changed")
    return reserved


def candidate_rows():
    gold_bytes = GOLD.read_bytes()
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    if (gold_manifest["sha256"] != digest(gold_bytes) or
            gold_manifest["training_eligible"] or
            not set(gold_manifest["training_excluded"]) <= set(EXCLUDED)):
        raise ValueError("reviewed contact gold or exclusions changed")
    rows = read_rows(GOLD)
    names = {row["name"] for row in rows}
    if len(names) != len(rows) or not set(EXCLUDED) <= names:
        raise ValueError("reviewed contact cases or exclusions changed")
    reserved = holdout_reserved_names()
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    holdout_groups = {row["source_group"] for row in read_rows(YEAR_HOLDOUT)}
    eval_groups = {family(row["source_url"]) for row in evaluation
                   if row.get("source_url")}
    candidates = [row for row in rows
                  if row["name"] not in EXCLUDED and row["name"] not in reserved
                  and row["source_group"] not in holdout_groups]
    if any(row["source_group"] in eval_groups for row in candidates):
        raise ValueError("reviewed source family overlaps evaluation")
    hits = collisions(evaluation,
                      ((row["name"], row["input"], row["expected"])
                       for row in candidates))
    if hits:
        raise ValueError(f"reviewed labels overlap evaluation: {sorted(hits)}")
    shared_streets = street_pairs(candidates) & street_pairs(evaluation)
    if shared_streets:
        raise ValueError(f"reviewed streets overlap evaluation: {sorted(shared_streets)}")
    if contact_surfaces(candidates, "email") & contact_surfaces(evaluation, "email"):
        raise ValueError("reviewed email overlaps evaluation")
    generic_relay = {"711"}
    if (contact_surfaces(candidates, "phone") &
            contact_surfaces(evaluation, "phone")) - generic_relay:
        raise ValueError("reviewed phone overlaps evaluation")
    return candidates


def main():
    rows = candidate_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha or previous_manifest["training_eligible"]:
        raise ValueError("previous reviewed contact candidates changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    groups = collections.Counter(row["source_group"] for row in rows)
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_ready_contact_train_candidate_v5",
        "supersedes_sha256": previous_sha,
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in EVALUATION},
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "training_candidate_after_input_integration_gate",
        "excluded": EXCLUDED,
        "holdout_reserved_cases": len(holdout_reserved_names() & {row["name"] for row in read_rows(GOLD)}),
        "holdout_source_family_cases": sum(row["source_group"] in {
            held["source_group"] for held in read_rows(YEAR_HOLDOUT)}
            for row in read_rows(GOLD)),
        "cases": len(rows),
        "entity_counts": dict(sorted(counts.items())),
        "repeated_source_groups": sum(count > 1 for count in groups.values()),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen contact candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen contact candidates changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen contact candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed contact candidates pass strict evaluation split: {digest(data)}")


if __name__ == "__main__":
    main()
