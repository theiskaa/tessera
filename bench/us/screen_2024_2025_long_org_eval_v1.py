"""Reserve source-distinct long US notice prose for a blind evaluation."""

import json

from build_us_full_eval_exclusions import digest
from screen_2024_2025_long_org_prose_v1 import OUT as TRAIN_PACKET
from screen_2024_2025_long_org_prose_v1 import ROOT, selected_rows
from source_identity import document_key


OUT = ROOT / "data/raw/candidates/federal-register-us-long-org-eval-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    """Freeze an unlabeled, source-separated long-form evaluation packet."""
    rows, categories, months, inputs, inventory_hash = selected_rows((TRAIN_PACKET,))
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_long_org_eval_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "intended_use": "heldout_evaluation",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "primary_categories": dict(sorted(categories.items())),
        "months": dict(sorted(months.items())),
        "raw_inventory_sha256": inventory_hash,
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long evaluation packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long evaluation packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long evaluation packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} long evaluation passages reserved across {len(months)} months")
    return rows


if __name__ == "__main__":
    main()
