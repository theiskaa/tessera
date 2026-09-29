"""Gate additional reviewed contacts against frozen evaluation and data diversity."""

import collections
import json
from pathlib import Path

import pyarrow.parquet as parquet

from active_sources import BASE_SILVER
from build_2025_year_contact_expansion_gold import MANIFEST as GOLD_MANIFEST
from build_2025_year_contact_expansion_gold import OUT as GOLD
from build_full_notice_gold import digest, read_rows
from build_silver import family
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import (
    EVALUATION, contact_surfaces, street_pairs,
)
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
PROCESSED = ROOT / "data/processed/detector-us-v1"
SILVER_NAME = "us-2025-year-contact-expansion-v1"
OUT = SILVER / "r24/us-2025-year-contact-expansion-train-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-2025-contact-2025-02501-1": "organization and address already repeated across active silver",
    "us-2025-contact-2025-07102-1": "USADF occurs in synthetic detector test",
    "us-2025-contact-2025-09366-1": "Internal Revenue Service occurs in synthetic detector validation",
    "us-2025-contact-2025-11924-1": "BOEM agency names occur in synthetic detector test",
    "us-2025-contact-2025-14255-1": "source omits Highway from the agency name paired with NHTSA",
    "us-2025-contact-2025-14264-1": "same source family and core contact as retained 2025-14263",
    "us-2025-contact-2025-21524-4": "near duplicate of retained 2025-21523 contact structure",
}


def candidate_rows():
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    gold = read_rows(GOLD)
    if (gold_manifest["sha256"] != digest(GOLD.read_bytes()) or
            gold_manifest["training_eligible"] or
            len({row["name"] for row in gold}) != len(gold) or
            not set(EXCLUDED) <= {row["name"] for row in gold}):
        raise ValueError("reviewed contact expansion or exclusions changed")
    rows = [row for row in gold if row["name"] not in EXCLUDED]
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    eval_sources = {row.get("source_id") for row in evaluation}
    eval_families = {family(row["source_url"]) for row in evaluation
                     if row.get("source_url")}
    if (any(row["source_id"] in eval_sources or row["source_group"] in eval_families
            for row in rows) or
            collisions(evaluation, ((row["name"], row["input"], row["expected"])
                                    for row in rows)) or
            street_pairs(rows) & street_pairs(evaluation)):
        raise ValueError("additional contacts overlap frozen evaluation")
    for kind in ("email", "phone"):
        shared = contact_surfaces(rows, kind) & contact_surfaces(evaluation, kind)
        if shared - ({"711"} if kind == "phone" else set()):
            raise ValueError(f"additional contact {kind} overlaps evaluation")
    active = [row for name, _ in BASE_SILVER if name != SILVER_NAME
              for row in read_rows(SILVER / f"{name}.jsonl")]
    active_ids = {row.get("source_id") for row in active}
    active_families = {row.get("source_group") for row in active}
    if any(row["source_id"] in active_ids or row["source_group"] in active_families
           for row in rows):
        raise ValueError("additional contact reuses an active source family")
    near = NearTextIndex()
    for row in active:
        near.add(row["id"], row["text"])
    for row in rows:
        if near.prior(row["input"]):
            raise ValueError(f"additional contact is repeated: {row['name']}")
        near.add(row["name"], row["input"])
    for split in ("valid", "test"):
        file = parquet.ParquetFile(PROCESSED / f"{split}.parquet")
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["text", "entities_json"]):
            hits = collisions(rows, ((split, row["text"], json.loads(row["entities_json"]))
                                     for row in batch.to_pylist()))
            if hits:
                raise ValueError(f"additional contacts overlap synthetic {split}: {sorted(hits)}")
    return rows


def main():
    rows = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_year_contact_expansion_train_candidate_v1",
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in EVALUATION},
        "active_silver_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                                 for name, _ in BASE_SILVER if name != SILVER_NAME},
        "excluded": EXCLUDED,
        "cases": len(rows),
        "entity_counts": dict(sorted(counts.items())),
        "sha256": digest(data),
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "training_candidate_after_input_integration_gate",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen contact expansion candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("contact expansion candidate changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("contact expansion candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} additional reviewed contacts pass strict split checks")


if __name__ == "__main__":
    main()
