"""Separate reviewed 2025 contacts by overlap with current US training inputs."""

import json
from pathlib import Path

from build_2025_contact_gold import MANIFEST as GOLD_MANIFEST, OUT as GOLD
from build_full_notice_gold import digest, read_rows
from check_holdout_overlap import SILVER, SILVER_SOURCES, SYNTHETIC, collisions, training_sources


ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "data/interim/review/us-2025-contact-holdout-v2.jsonl"
TRAIN_CANDIDATE = ROOT / "data/interim/silver/r24/us-2025-contact-train-candidate-v2.jsonl"
MANIFEST = ROOT / "data/interim/review/us-2025-contact-split-v2.manifest.json"


def prepared_hashes():
    paths = [SILVER / f"{name}.jsonl" for name in SILVER_SOURCES]
    paths += [SYNTHETIC / f"{split}.parquet" for split in ("train", "valid")]
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}


def freeze(path, rows):
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"frozen contact split changed: {path.name}")
    if not path.exists():
        path.write_bytes(data)
    return digest(data)


def main():
    gold_bytes = GOLD.read_bytes()
    gold_manifest = json.loads(GOLD_MANIFEST.read_text())
    if digest(gold_bytes) != gold_manifest["sha256"]:
        raise ValueError("2025 contact gold changed")
    rows = read_rows(GOLD)
    hashes = prepared_hashes()
    hits = collisions(rows, training_sources())
    holdout = [row for row in rows if row["name"] not in hits]
    train_candidate = [row for row in rows if row["name"] in hits]
    if (len(holdout) != 16 or len(train_candidate) != 11 or
            len({row["source_id"] for row in rows}) != len(rows)):
        raise ValueError("2025 contact split changed; review overlap before refreezing")
    holdout_hash = freeze(HOLDOUT, holdout)
    candidate_hash = freeze(TRAIN_CANDIDATE, train_candidate)
    manifest = {
        "kind": "us_2025_contact_split_v2",
        "source_group_key": "source_id",
        "gold_sha256": digest(gold_bytes),
        "prepared_sha256": hashes,
        "holdout_cases": len(holdout),
        "holdout_sha256": holdout_hash,
        "holdout_status": "diagnostic_name_disjoint_from_current_prepared_inputs",
        "train_candidate_cases": len(train_candidate),
        "train_candidate_sha256": candidate_hash,
        "train_candidate_status": "reviewed_but_not_training_eligible",
        "overlap_kinds": {name: sorted(kinds) for name, kinds in sorted(hits.items())},
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2025 contact split inputs changed")
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(holdout)} clean diagnostic contacts, "
          f"{len(train_candidate)} reviewed training candidates")


if __name__ == "__main__":
    main()
