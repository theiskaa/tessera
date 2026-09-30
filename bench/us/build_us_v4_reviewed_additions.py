"""Promote three independently reviewed US packets after current split checks."""

import collections
import json
import re
from pathlib import Path

from active_sources import V4_BASE_SILVER
from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2026_contact_expansion_train_candidates_v3 import OUT as EXPANSION
from freeze_2026_contact_expansion_train_candidates_v3 import candidate_rows as expansion_rows
from freeze_2026_acronym_prose_train_candidates_v1 import OUT as ACRONYM
from freeze_2026_acronym_prose_train_candidates_v1 import candidate_rows as acronym_rows
from freeze_2026_failure_target_train_candidates_v2 import OUT as CONTACTS
from freeze_2026_failure_target_train_candidates_v2 import candidate_rows as contact_rows
from freeze_2026_nonorganization_train_candidates_v1 import OUT as NONORGANIZATION
from freeze_2026_nonorganization_train_candidates_v1 import candidate_rows as nonorganization_rows
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/silver/us-v4-reviewed-additions-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PACKETS = (
    (CONTACTS, contact_rows, 24),
    (ACRONYM, acronym_rows, 31),
    (NONORGANIZATION, nonorganization_rows, 22),
    (EXPANSION, expansion_rows, 36),
)


def shingles(text, size):
    """Find long shared evaluation passages even when whole texts differ."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[index:index + size])
            for index in range(len(words) - size + 1)}


def addition_rows():
    """Revalidate source, review, and strict evaluation isolation before promotion."""
    evaluation, _ = exclusion_rows_v4()
    base_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                  for name, _ in V4_BASE_SILVER]
    base = [row for path in base_paths for row in lines(path)]
    eval_keys = {document_key(row) for row in evaluation}
    base_keys = {document_key(row) for row in base}
    if None in base_keys or len(base) != 1511 or base_keys & eval_keys:
        raise ValueError("US v4 base training shares an evaluation source")
    evaluation_shingles = set().union(*(shingles(row["input"], 12)
                                        for row in evaluation))
    near = NearTextIndex()
    for row in base:
        near.add(row["id"], row["text"])
    seen = set(base_keys)
    rows = []
    for path, build, expected in PACKETS:
        candidates, _, _ = build()
        if (len(candidates) != expected or candidates != lines(path) or
                json.loads(path.with_suffix(".manifest.json").read_text())[
                    "training_eligible"] is not False):
            raise ValueError(f"US reviewed candidate changed: {path.name}")
        for candidate in candidates:
            key = document_key(candidate)
            if key is None or key in seen or key in eval_keys:
                raise ValueError(f"US reviewed candidate source crossed a split: {candidate['name']}")
            if shingles(candidate["input"], 12) & evaluation_shingles:
                raise ValueError(f"US reviewed candidate repeats evaluation prose: {candidate['name']}")
            if near.prior(candidate["input"]):
                raise ValueError(f"US reviewed candidate repeats training prose: {candidate['name']}")
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
    if len(rows) != 113 or len(seen) != len(base_keys) + len(rows):
        raise ValueError("US reviewed additions lost source diversity")
    samples = [(row["id"], row["text"], row["entities"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("US reviewed additions share held-out labeled entities")
    return rows, base_paths


def main():
    """Freeze eligible V4 silver additions without starting a training run."""
    rows, base_paths = addition_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_v4_reviewed_additions_v2",
        "training_eligible": True,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["entities"]).items())),
        "candidate_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path, _, _ in PACKETS
        },
        "evaluation_sha256": digest(EVALUATION_PATH.read_bytes()),
        "base_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                        for path in base_paths},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US v4 reviewed additions or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US v4 reviewed additions changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US v4 reviewed addition inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed US documents promoted as V4 training input")


if __name__ == "__main__":
    main()
