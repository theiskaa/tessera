"""Remove a registry-number acronym whose organization label conflicts with policy."""

import collections
import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_long_org_eval_gold_v2 import MANIFEST as PREVIOUS_MANIFEST
from freeze_2024_2025_long_org_eval_gold_v2 import OUT as PREVIOUS_GOLD
from freeze_2024_2025_long_org_eval_gold_v2 import ROOT
from freeze_2024_2025_long_org_eval_gold_v2 import main as verify_previous_gold
from source_identity import document_key


OUT = ROOT / "data/interim/review/us-long-org-eval-gold-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "2025-14213": "CAS describes chemical registry numbers here; the matching training example was excluded on the same policy."
}


def main():
    """Pin a policy-consistent, source-backed long-document evaluation set."""
    rows = verify_previous_gold()
    removed = {row["source_id"] for row in rows if row["source_id"] in EXCLUDED}
    kept = [row for row in rows if row["source_id"] not in EXCLUDED]
    if (len(rows) != 18 or len(kept) != 17 or removed != set(EXCLUDED) or
            len({document_key(row) for row in kept}) != len(kept)):
        raise ValueError("long US evaluation policy exclusions changed")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in kept).encode()
    manifest = {
        "kind": "us_long_org_eval_gold_v3",
        "training_eligible": False,
        "cases": len(kept),
        "source_documents": len({document_key(row) for row in kept}),
        "excluded_label_ambiguity": EXCLUDED,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in kept for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in (PREVIOUS_GOLD, PREVIOUS_MANIFEST)},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long evaluation gold v3 or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long evaluation gold v3 changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long evaluation gold v3 inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} policy-consistent long US evaluation cases frozen")
    return kept


if __name__ == "__main__":
    main()
