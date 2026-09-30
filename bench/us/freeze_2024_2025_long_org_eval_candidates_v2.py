"""Reserve long notice evaluation candidates with strict source-family separation."""

import json

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from screen_2024_2025_long_org_eval_v1 import MANIFEST as FIRST_MANIFEST
from screen_2024_2025_long_org_eval_v1 import OUT as FIRST_PACKET
from screen_2024_2025_long_org_eval_v1 import main as verify_first_packet
from screen_2024_2025_long_org_prose_v1 import ACTIVE_AT_CAPTURE, ROOT
from source_identity import document_key

from screen_2024_2025_long_org_prose_v1 import source_family


OUT = ROOT / "data/raw/candidates/federal-register-us-long-org-eval-v1/blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    """Drop template families already present in active silver before labeling."""
    verify_first_packet()
    first = lines(FIRST_PACKET)
    paths = [ROOT / f"data/interim/silver/{name}.jsonl"
             for name, _ in ACTIVE_AT_CAPTURE]
    active = [row for path in paths for row in lines(path)]
    groups = {source_family(row["source_url"]) for row in active
              if row.get("source_url", "").startswith(
                  "https://www.federalregister.gov/documents/")}
    kept = [row for row in first if row["source_group"] not in groups]
    excluded = [row["source_id"] for row in first if row["source_group"] in groups]
    if (len(first) != 48 or len(kept) != 43 or len(excluded) != 5 or
            len({document_key(row) for row in kept}) != len(kept) or
            {document_key(row) for row in kept} &
            {document_key(row) for row in active}):
        raise ValueError("long evaluation source-family filter changed")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in kept).encode()
    manifest = {
        "kind": "federal_register_us_long_org_eval_blind_v2",
        "training_eligible": False,
        "label_status": "unlabeled",
        "intended_use": "heldout_evaluation",
        "cases": len(kept),
        "source_documents": len(kept),
        "source_groups": len({row["source_group"] for row in kept}),
        "excluded_existing_train_families": sorted(excluded),
        "first_packet_sha256": digest(FIRST_PACKET.read_bytes()),
        "first_manifest_sha256": digest(FIRST_MANIFEST.read_bytes()),
        "active_input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                for path in paths},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("strict long evaluation packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("strict long evaluation packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("strict long evaluation packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} long evaluation candidates reserved; {len(excluded)} "
          "existing training families excluded")
    return kept


if __name__ == "__main__":
    main()
