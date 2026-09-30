"""Freeze unlabeled US agency contacts from independent official pages."""

import collections
import hashlib
import json
import re
from pathlib import Path

from freeze_2025_ready_contact_train_candidates import EVALUATION
from official_contact_html import blocks
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/candidates/us-official-contacts-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ORG_TERM = re.compile(r"\b(?:office|division|center|commission|administration|branch)\b", re.I)
PHONE = re.compile(r"(?:\(\d{3}\)\s*|\b\d{3}[- .])\d{3}[- .]\d{4}\b")
ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packet_rows():
    """Pick diverse, source-indexed table and paragraph passages."""
    nara, nara_source = blocks("nara-organization-telephone-list")
    epa, epa_source = blocks("epa-oil-spill-regional-contacts")
    nara_candidates = []
    for index, cells in enumerate(nara.rows):
        if (len(cells) != 6 or not ORG_TERM.search(cells[1]) or
                cells[2] not in {"Archives I", "Archives II"} or
                not PHONE.search(cells[4]) or not cells[5] or cells[5] == "Vacant"):
            continue
        nara_candidates.append((index, cells))
    selected = []
    names = set()
    for index, cells in sorted(nara_candidates, key=lambda item:
                               digest(f"nara-v1:{item[0]}".encode())):
        if len(selected) >= 30:
            break
        person = cells[5].removesuffix(" (A)")
        if person in names:
            continue
        names.add(person)
        selected.append({
            "name": f"us-official-nara-telephone-row-{index:03d}",
            "country": "US", "source_page": "nara-organization-telephone-list",
            "source_block": {"kind": "table_row", "index": index},
            "input": " | ".join(cells), "category": "org_person_table",
            **nara_source,
        })
    epa_people = []
    epa_addresses = []
    for index, text in enumerate(epa.paragraphs):
        if (re.match(r"^[A-Z][A-Za-z .\-]+ \([^\n]+@epa\.gov\)", text) and
                PHONE.search(text)):
            epa_people.append((index, text))
        if (re.search(r"(?:U\.S\. EPA|EPA Region)", text) and ZIP.search(text)):
            epa_addresses.append((index, text))
    for category, candidates, limit in (
            ("role_near_person", epa_people, 12),
            ("org_address", epa_addresses, 6)):
        for index, text in sorted(candidates, key=lambda item:
                                  digest(f"epa-v1:{item[0]}".encode()))[:limit]:
            selected.append({
                "name": f"us-official-epa-paragraph-{index:03d}",
                "country": "US", "source_page": "epa-oil-spill-regional-contacts",
                "source_block": {"kind": "paragraph", "index": index},
                "input": text, "category": category, **epa_source,
            })
    if (len(selected) != 48 or len({row["name"] for row in selected}) != 48 or
            len({digest(row["input"].encode()) for row in selected}) != 48):
        raise ValueError(f"official contact packet selection changed: "
                         f"{len(selected)}, "
                         f"{dict(collections.Counter(row['category'] for row in selected))}")
    selected.sort(key=lambda row: row["name"])
    return selected


def main():
    """Pin review text without assigning model labels."""
    rows = packet_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_official_contacts_blind_v1",
        "cases": len(rows),
        "categories": dict(sorted(collections.Counter(row["category"] for row in rows).items())),
        "source_capture_sha256": digest((ROOT / "data/raw/us-official-contact-pages-v1/manifest.json").read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False,
        "label_status": "unlabeled",
        "extraction": "HTML table cells joined by ' | '; paragraph line breaks kept",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("official contact packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("official contact packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("official contact packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} official US contact passages frozen for blind review")


if __name__ == "__main__":
    main()
