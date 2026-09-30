"""Promote reviewed historical organization-acronym prose."""

import collections
import json
from pathlib import Path

from active_sources import V4_BASE_SILVER
from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2024_2025_acronym_prose_train_candidates_v2 import OUT as FIRST
from freeze_2024_2025_acronym_prose_train_candidates_v2 import candidate_rows as first_rows
from freeze_2024_2025_acronym_third_review_v1 import OUT as THIRD
from freeze_2024_2025_acronym_third_review_v1 import candidate_rows as third_rows
from screen_2026_contact_expansion_v3 import shingles
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/silver/us-v4-reviewed-historical-acronyms-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def first_candidates():
    """Use the agreed rows without their review-disposition counts."""
    return first_rows()[0]


PACKETS = ((FIRST, first_candidates, 14), (THIRD, third_rows, 1))
BASE_PATHS = tuple(ROOT / f"data/interim/silver/{name}.jsonl"
                   for name, _ in V4_BASE_SILVER) + (
    ROOT / "data/interim/silver/us-v4-reviewed-additions-v2.jsonl",
    ROOT / "data/interim/silver/us-v4-reviewed-long-additions-v1.jsonl",
    ROOT / "data/interim/silver/us-v4-reviewed-historical-long-v1.jsonl",
)


def addition_rows():
    """Recheck source and entity isolation before training eligibility."""
    evaluation, _ = exclusion_rows_v4()
    base = [row for path in BASE_PATHS for row in lines(path)]
    eval_keys = {document_key(row) for row in evaluation}
    seen = {document_key(row) for row in base}
    if len(base) != 1655 or len(seen) != 626 or None in seen or seen & eval_keys:
        raise ValueError("US V4 acronym base source split changed")
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
    near = NearTextIndex()
    for row in base:
        near.add(row["id"], row["text"])
    rows = []
    for path, build, expected in PACKETS:
        candidates = build()
        if (len(candidates) != expected or candidates != lines(path) or
                json.loads(path.with_suffix(".manifest.json").read_text())[
                    "training_eligible"] is not False):
            raise ValueError(f"historical acronym candidate changed: {path.name}")
        for candidate in candidates:
            key = document_key(candidate)
            if key is None or key in seen or key in eval_keys:
                raise ValueError(f"historical acronym source crossed a split: {candidate['name']}")
            if shingles(candidate["input"], 12) & eval_shingles:
                raise ValueError(f"historical acronym repeats evaluation: {candidate['name']}")
            if near.prior(candidate["input"]):
                raise ValueError(f"historical acronym repeats training: {candidate['name']}")
            seen.add(key)
            near.add(candidate["name"], candidate["input"])
            rows.append({
                "id": candidate["name"], "country": "US",
                "source": "federal-register", "source_id": candidate["source_id"],
                "source_url": candidate["source_url"],
                "source_group": candidate["source_group"],
                "source_document_key": key,
                "raw_sha256": candidate["raw_sha256"],
                "text": candidate["input"],
                "entities": [{"kind": span["kind"], "start": span["start"],
                              "end": span["end"]} for span in candidate["expected"]],
            })
    if len(rows) != 15 or len(seen) != 641:
        raise ValueError("historical acronym additions lost source diversity")
    samples = [(row["id"], row["text"], row["entities"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("historical acronym additions share held-out entities")
    return rows


def main():
    """Freeze eligible historical acronym prose without training."""
    rows = addition_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_v4_reviewed_historical_acronyms_v1",
        "training_eligible": True,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["entities"]).items())),
        "candidate_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                             for path, _, _ in PACKETS},
        "evaluation_sha256": digest(EVALUATION_PATH.read_bytes()),
        "base_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                        for path in BASE_PATHS},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical acronym additions or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical acronym additions changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical acronym addition inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} historical acronym passages promoted as V4 training input")


if __name__ == "__main__":
    main()
