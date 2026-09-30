"""Freeze remaining DOL offices outside the reserved states for blind review."""

import json
import re
from pathlib import Path

from active_sources import PRIOR_DOL_REMAINING_SILVER
from build_contact_snippets import lines
from build_full_notice_gold import digest
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_dol_whd_office_packet import (
    PHONE, POSTCODE, ROLE, SOURCE, SOURCE_MANIFEST, source_blocks,
)
from freeze_us_dol_whd_office_train_candidates import EXCLUDED_STATES, HELDOUT_STATES
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/candidates/us-dol-whd-remaining-offices-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def packet_rows():
    """Select complete office blocks while preserving the state-disjoint reserve."""
    blocks, raw_hash = source_blocks()
    rows = []
    for block in blocks:
        state = block["state"]
        if (block["office_index"] <= 1 or state in HELDOUT_STATES or
                state in EXCLUDED_STATES or state in {"Guam", "Puerto Rico"}):
            continue
        paragraphs = block["paragraphs"]
        addresses = [part for part in paragraphs if POSTCODE.search(part)]
        phones = [part for part in paragraphs if PHONE.fullmatch(part)]
        roles = [part for part in paragraphs if ROLE.search(part)]
        if len(addresses) != 1 or len(phones) != 1 or not roles:
            raise ValueError(f"DOL remaining office has unclear contact block: {state}")
        text = "\n".join([block["office"], addresses[0], phones[0], *roles[:2]])
        if len(text) > 400:
            raise ValueError(f"DOL remaining office is too long: {state}")
        slug = re.sub(r"[^a-z0-9]+", "-", state.casefold()).strip("-")
        rows.append({
            "name": f"us-dol-whd-remaining-{slug}-{block['office_index']}",
            "country": "US", "input": text, "source_url": SOURCE,
            "source_state": state,
            "source_block": {"state": state, "office_index": block["office_index"]},
            "raw_sha256": raw_hash,
        })
    if len(rows) != 20 or len({row["input"] for row in rows}) != 20:
        raise ValueError("DOL remaining office membership changed")
    return rows


def main():
    """Pin unlabeled office snippets and current evaluation membership."""
    rows = packet_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_DOL_REMAINING_SILVER]
    manifest = {
        "kind": "us_dol_whd_remaining_offices_blind_v1", "cases": len(rows),
        "source_manifest_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in evaluation_paths},
        "active_silver_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                 for path in active_paths},
        "excluded_states": sorted(HELDOUT_STATES | set(EXCLUDED_STATES)),
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("DOL remaining packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("DOL remaining packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("DOL remaining packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} remaining DOL offices frozen for blind review")


if __name__ == "__main__":
    main()
