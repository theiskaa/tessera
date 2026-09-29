"""Freeze reviewed targeted contacts after strict split and source checks."""

import collections
import json
from pathlib import Path

import pyarrow.parquet as parquet

from active_sources import ACTIVE_SILVER, BASE_SILVER
from build_2025_targeted_contact_gold import MANIFEST as GOLD_MANIFEST
from build_2025_targeted_contact_gold import OUT as GOLD, gold_rows
from build_full_notice_gold import digest, read_rows
from build_silver import family
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import (
    EVALUATION, contact_surfaces, street_pairs,
)
from org_aliases import active_aliases, normalized
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
PROCESSED = ROOT / "data/processed/detector-us-v1"
SILVER_NAME = "us-2025-targeted-contacts-v1"
OUT = SILVER / "r24/us-2025-targeted-contact-train-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def prior_hashes():
    return {name: digest((SILVER / f"{name}.jsonl").read_bytes())
            for name, _ in ACTIVE_SILVER if name != SILVER_NAME}


def candidate_rows():
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    gold = read_rows(GOLD)
    rebuilt, _ = gold_rows()
    if (gold_manifest["sha256"] != digest(GOLD.read_bytes()) or
            gold_manifest["training_eligible"] or gold != rebuilt or
            len(gold) != 5 or len({row["name"] for row in gold}) != len(gold)):
        raise ValueError("targeted contact gold changed")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    eval_ids = {row.get("source_id") for row in evaluation}
    eval_families = {family(row["source_url"]) for row in evaluation
                     if row.get("source_url")}
    if (any(row["source_id"] in eval_ids or row["source_group"] in eval_families
            for row in gold) or
            collisions(evaluation, ((row["name"], row["input"], row["expected"])
                                    for row in gold)) or
            street_pairs(gold) & street_pairs(evaluation)):
        raise ValueError("targeted contacts overlap frozen evaluation")
    for kind in ("email", "phone"):
        shared = contact_surfaces(gold, kind) & contact_surfaces(evaluation, kind)
        if shared - ({"711"} if kind == "phone" else set()):
            raise ValueError(f"targeted contact {kind} overlaps evaluation")
    active = [row for name, _ in ACTIVE_SILVER if name != SILVER_NAME
              for row in read_rows(SILVER / f"{name}.jsonl")]
    base = [row for name, _ in BASE_SILVER
            for row in read_rows(SILVER / f"{name}.jsonl")]
    prior = active + base
    prior_ids = {row.get("source_id") for row in prior}
    prior_families = {row.get("source_group") for row in prior}
    if any(row["source_id"] in prior_ids or row["source_group"] in prior_families
           for row in gold):
        raise ValueError("targeted contact reuses a silver source family")
    gold_org = {normalized(span["text"]) for row in gold for span in row["expected"]
                if span["kind"] == "org"}
    eval_org = {normalized(span["text"]) for row in evaluation for span in row["expected"]
                if span["kind"] == "org"}
    if (gold_org & active_aliases(evaluation, include_roster=True) or
            eval_org & active_aliases(gold, include_roster=True)):
        raise ValueError("targeted contact organization aliases overlap evaluation")
    prior_as_gold = [{"name": row["id"], "input": row["text"],
                      "expected": [{**span, "text": row["text"].encode()[
                          span["start"]:span["end"]].decode()}
                                   for span in row["entities"]]}
                     for row in prior]
    if (collisions(gold, ((row["id"], row["text"], row["entities"])
                          for row in prior)) or
            collisions(prior_as_gold, ((row["name"], row["input"], row["expected"])
                                       for row in gold))):
        raise ValueError("targeted contacts overlap prior silver labels")
    near = NearTextIndex()
    for row in prior:
        near.add(row["id"], row["text"])
    for row in gold:
        if near.prior(row["input"]):
            raise ValueError(f"targeted contact repeats prior text: {row['name']}")
        near.add(row["name"], row["input"])
    for split in ("valid", "test"):
        file = parquet.ParquetFile(PROCESSED / f"{split}.parquet")
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["text", "entities_json"]):
            hits = collisions(gold, ((split, row["text"], json.loads(row["entities_json"]))
                                     for row in batch.to_pylist()))
            if hits:
                raise ValueError(f"targeted contacts overlap synthetic {split}: {sorted(hits)}")
    return gold


def main():
    rows = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_targeted_contact_train_candidate_v1",
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in EVALUATION},
        "active_silver_sha256": prior_hashes(),
        "cases": len(rows), "entity_counts": dict(sorted(counts.items())),
        "sha256": digest(data), "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "training_candidate_after_input_integration_gate",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("targeted contact candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("targeted contact candidate changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("targeted contact candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} targeted contacts pass strict split checks")


if __name__ == "__main__":
    main()
