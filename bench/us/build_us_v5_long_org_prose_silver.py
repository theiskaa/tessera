"""Promote independently reviewed long organization passages without activating training."""

import collections
import json
import re
from pathlib import Path

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as V4_EVAL
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2024_2025_long_org_eval_gold_v3 import OUT as LONG_EVAL
from freeze_2024_2025_long_org_eval_gold_v3 import main as verify_long_eval
from freeze_2024_2025_long_org_train_candidates_v1 import OUT as CANDIDATE
from freeze_2024_2025_long_org_train_candidates_v1 import candidate_rows
from screen_2026_contact_expansion_v3 import shingles
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/detector-shared-v4.toml"
OUT = ROOT / "data/interim/silver/us-v5-reviewed-long-org-prose-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def silver_rows():
    """Recheck source and text isolation against every active V4 source and held-out case."""
    match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", CONFIG.read_text())
    if not match:
        raise ValueError("V4 detector silver list is missing")
    base_paths = [ROOT / path for path in json.loads(match.group(1))]
    base = [row for path in base_paths for row in lines(path)]
    if len(base) != 1670:
        raise ValueError("active V4 silver membership changed")
    base_keys = {document_key(row) for row in base}
    if len(base_keys) != 641 or None in base_keys:
        raise ValueError("active V4 source identity changed")
    old_eval, _ = exclusion_rows_v4()
    new_eval = verify_long_eval()
    eval_rows = old_eval + new_eval
    if any(document_key(row) is None for row in new_eval):
        raise ValueError("new evaluation source identity is missing")
    eval_keys = {document_key(row) for row in eval_rows} - {None}
    if base_keys & eval_keys:
        raise ValueError("training and evaluation source families overlap")
    eval_shingles = set().union(*(shingles(row["input"], 12) for row in eval_rows))
    near = NearTextIndex()
    for row in base:
        near.add(row["id"], row["text"])
    candidates, _, _ = candidate_rows()
    if (len(candidates) != 35 or candidates != lines(CANDIDATE) or
            json.loads(CANDIDATE.with_suffix(".manifest.json").read_text())[
                "training_eligible"] is not False):
        raise ValueError("long organization candidate changed")
    rows = []
    for candidate in candidates:
        key = document_key(candidate)
        if key is None or key in base_keys or key in eval_keys:
            raise ValueError(f"source crossed a split: {candidate['name']}")
        if shingles(candidate["input"], 12) & eval_shingles:
            raise ValueError(f"evaluation prose repeated: {candidate['name']}")
        if near.prior(candidate["input"]):
            raise ValueError(f"training prose repeated: {candidate['name']}")
        base_keys.add(key)
        near.add(candidate["name"], candidate["input"])
        rows.append({
            "id": candidate["name"],
            "country": "US",
            "source": "federal-register",
            "source_id": candidate["source_id"],
            "source_url": candidate["source_url"],
            "source_group": candidate["source_group"],
            "source_document_key": key,
            "raw_sha256": candidate["raw_sha256"],
            "text": candidate["input"],
            "entities": [{"kind": span["kind"], "start": span["start"],
                          "end": span["end"]} for span in candidate["expected"]],
        })
    if len(base_keys) != 676:
        raise ValueError("long organization source diversity changed")
    samples = [(row["id"], row["text"], row["entities"]) for row in rows]
    if collisions(eval_rows, samples) or contact_collisions(eval_rows, samples):
        raise ValueError("new training labels overlap held-out entities")
    return rows, base_paths


def main():
    """Freeze training-eligible silver with source and evaluation hashes."""
    rows, base_paths = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_v5_reviewed_long_org_prose_v1",
        "training_eligible": True,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["entities"]).items())),
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "evaluation_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (V4_EVAL, LONG_EVAL)
        },
        "base_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                        for path in base_paths},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long organization silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long organization silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long organization silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed long US organization passages are training eligible")


if __name__ == "__main__":
    main()
