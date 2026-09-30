"""Build source-linked office diagnostics from reserved DOL states and USDA pages."""

import json
from pathlib import Path

from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from freeze_us_dol_whd_office_packet import OUT as DOL_PACKET
from freeze_us_dol_whd_office_train_candidates import (
    HELDOUT_STATES, REVIEW_A as DOL_A, REVIEW_B as DOL_B,
    candidate_rows as dol_candidates,
)
from freeze_us_nrcs_field_office_packet import OUT as NRCS_PACKET
from freeze_us_nrcs_field_office_train_candidates import (
    HELDOUT_PAGES, REVIEW_A_V2 as NRCS_A, REVIEW_B as NRCS_B,
    candidate_rows as nrcs_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/review/us-reserved-office-diagnostic-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def diagnostic_rows():
    """Require frozen training exclusions and two agreeing, certain labels."""
    _, dol_held, _, _, _ = dol_candidates()
    _, nrcs_held, _, _ = nrcs_candidates()
    rows = []
    excluded = []
    for packet_path, left_path, right_path, held, source_group in (
            (DOL_PACKET, DOL_A, DOL_B, set(dol_held), "us-dol-whd-offices-v1"),
            (NRCS_PACKET, NRCS_A, NRCS_B, set(nrcs_held), "us-nrcs-pa-directory-2026-v1")):
        packet, left, right = (lines(path) for path in (packet_path, left_path, right_path))
        if not (len(packet) == len(left) == len(right)):
            raise ValueError("reserved office reviews do not cover packet")
        for source, a, b in zip(packet, left, right):
            name = source["name"]
            if name != a["name"] or name != b["name"]:
                raise ValueError("reserved office review order changed")
            if name not in held:
                continue
            a_spans = validate_spans(name, source["input"], a["entities"])
            b_spans = validate_spans(name, source["input"], b["entities"])
            if a["uncertain"] or b["uncertain"] or a_spans != b_spans:
                excluded.append(name)
                continue
            rows.append({
                "name": name, "country": "US", "doc_type": "agency_mailing_address",
                "input": source["input"],
                "expected": a_spans, "source_url": source["source_url"],
                "source_group": source_group,
            })
    if len(rows) != 20 or len(excluded) != 2:
        raise ValueError("reserved office diagnostic membership changed")
    return rows, excluded


def main():
    """Pin diagnostic gold; these source blocks remain outside training."""
    rows, excluded = diagnostic_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (DOL_PACKET, DOL_A, DOL_B, NRCS_PACKET, NRCS_A, NRCS_B)
    manifest = {
        "kind": "us_reserved_office_diagnostic_v1", "cases": len(rows),
        "excluded_uncertain": excluded,
        "heldout_states": sorted(HELDOUT_STATES),
        "heldout_pages": sorted(HELDOUT_PAGES),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                          for path in inputs},
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("reserved office diagnostic or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("reserved office diagnostic changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("reserved office diagnostic inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed reserved office cases; {len(excluded)} uncertain cases excluded")


if __name__ == "__main__":
    main()
