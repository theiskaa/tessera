"""Exclude semantically uncertain labels from the long US evaluation gold."""

import collections
import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_long_org_eval_gold_v1 import MANIFEST as FIRST_MANIFEST
from freeze_2024_2025_long_org_eval_gold_v1 import OUT as FIRST_GOLD
from freeze_2024_2025_long_org_eval_gold_v1 import ROOT
from freeze_2024_2025_long_org_eval_gold_v1 import main as verify_first_gold
from source_identity import document_key


OUT = ROOT / "data/interim/review/us-long-org-eval-gold-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "2024-27514": "University of California overlaps several campus names in training.",
    "2024-29467": "The proposed grant's Consultative Meeting Board may be a role, not an established body.",
    "2025-02547": "CAS modifies registry number; matching training prose was excluded on this policy.",
    "2025-09693": "CAS modifies registry number; matching training prose was excluded on this policy.",
}
UNCERTAINTY_PATHS = (
    ROOT / "data/interim/review/us-long-org-eval-review-a-uncertain-v1.json",
    ROOT / "data/interim/review/us-long-org-eval-review-b-uncertain-v1.json",
)


def gold_rows():
    """Return only independently agreed rows with consistent entity policy."""
    rows = verify_first_gold()
    removed = {row["source_id"] for row in rows if row["source_id"] in EXCLUDED}
    kept = [row for row in rows if row["source_id"] not in EXCLUDED]
    if (len(rows) != 22 or len(kept) != 18 or removed != set(EXCLUDED) or
            len({document_key(row) for row in kept}) != len(kept)):
        raise ValueError("long US evaluation policy exclusions changed")
    return kept


def main():
    """Pin the policy-consistent held-out long-document gold set."""
    rows = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (FIRST_GOLD, FIRST_MANIFEST, *UNCERTAINTY_PATHS)
    manifest = {
        "kind": "us_long_org_eval_gold_v2",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "excluded_label_ambiguity": EXCLUDED,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long evaluation gold v2 or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long evaluation gold v2 changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long evaluation gold v2 inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} policy-consistent long US evaluation cases frozen")
    return rows


if __name__ == "__main__":
    main()
