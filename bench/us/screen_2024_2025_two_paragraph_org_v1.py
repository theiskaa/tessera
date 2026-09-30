"""Reserve contiguous two-paragraph Federal Register prose for blind review."""

import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_long_org_eval_candidates_v2 import OUT as EVAL_PACKET
from screen_2024_2025_extra_long_org_v1 import OUT as EXTRA_PACKET
from screen_2024_2025_long_org_prose_v1 import OUT as FIRST_PACKET
from screen_2024_2025_long_org_prose_v1 import ROOT, selected_rows
from source_identity import document_key


OUT = ROOT / "data/raw/candidates/federal-register-us-two-paragraph-org-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    """Pin distinct source windows without labels or training activation."""
    rows, categories, months, inputs, inventory_hash = selected_rows(
        extra_prior=(FIRST_PACKET, EVAL_PACKET, EXTRA_PACKET),
        min_chars=1800,
        max_chars=4500,
        ideal_chars=3000,
        category_quotas=(("acronym_context", 19), ("org_context", 1)),
        paragraphs_per_sample=2,
    )
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_two_paragraph_org_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "primary_categories": dict(sorted(categories.items())),
        "months": dict(sorted(months.items())),
        "paragraphs_per_sample": 2,
        "length_range_characters": [1800, 4500],
        "raw_inventory_sha256": inventory_hash,
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("two-paragraph packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("two-paragraph packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("two-paragraph packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} two-paragraph US passages reserved across {len(months)} months")
    return rows


if __name__ == "__main__":
    main()
