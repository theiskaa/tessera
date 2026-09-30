"""Freeze USDA Maine field-office tables for blind entity review."""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from freeze_2025_ready_contact_train_candidates import EVALUATION
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-nrcs-maine-directory-v1/page.html"
SOURCE = "https://www.nrcs.usda.gov/state-offices/maine/nrcs-maine-offices-and-employee-directory"
SOURCE_MANIFEST = RAW.with_name("manifest.json")
OUT = ROOT / "data/raw/candidates/us-nrcs-maine-offices-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
OFFICE = re.compile(r"^(.+ Field Office) \((.+)\) --- (.+)$")


class OfficeTables(HTMLParser):
    """Project visible captions and staff cells from the official HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.tag = None
        self.parts = []
        self.caption = ""
        self.row = []
        self.rows = []
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.inside:
                raise ValueError("nested Maine office table")
            self.inside = True
            self.caption = ""
            self.rows = []
        if self.inside and tag in {"caption", "td"}:
            self.tag = tag
            self.parts = []

    def handle_data(self, data):
        if self.tag:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.tag == tag:
            value = " ".join("".join(self.parts).split())
            if tag == "caption":
                self.caption = value
            else:
                self.row.append(value)
            self.tag = None
        if self.inside and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = []
        if self.inside and tag == "table":
            self.tables.append((self.caption, self.rows))
            self.inside = False


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_tables():
    """Verify the captured page and return its rendered office table data."""
    data = RAW.read_bytes()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (len(data) < 200_000 or
            f'rel="canonical" href="{SOURCE}"'.encode() not in data or
            manifest["kind"] != "us_nrcs_maine_directory_v1" or
            manifest["source_url"] != SOURCE or
            manifest["raw_sha256"] != digest(data) or
            manifest["bytes"] != len(data) or
            manifest["training_eligible"]):
        raise ValueError("USDA Maine source capture changed")
    parser = OfficeTables()
    parser.feed(data.decode("utf-8"))
    offices = [(caption, rows) for caption, rows in parser.tables if OFFICE.fullmatch(caption)]
    if len(parser.tables) != 15 or len(offices) != 14:
        raise ValueError("USDA Maine office table structure changed")
    return offices, digest(data)


def packet_rows():
    """Select each named office and its first two staff rows without labels."""
    offices, raw_hash = source_tables()
    rows = []
    for index, (caption, staff) in enumerate(offices, start=1):
        match = OFFICE.fullmatch(caption)
        if match is None or not staff or any(len(row) != 4 for row in staff):
            raise ValueError(f"USDA Maine office table is incomplete: {index}")
        office_name = match.group(1)
        slug = re.sub(r"[^a-z0-9]+", "-", office_name.casefold()).strip("-")
        selected = staff[:2]
        text = "\n".join([caption, *(" | ".join(row) for row in selected)])
        if len(text) > 450:
            raise ValueError(f"USDA Maine office packet row is too long: {index}")
        rows.append({
            "name": f"us-nrcs-maine-office-{slug}", "country": "US",
            "input": text, "source_url": SOURCE,
            "source_block": {"table": index, "staff_rows": len(selected)},
            "raw_sha256": raw_hash,
        })
    if len(rows) != 14 or len({row["name"] for row in rows}) != 14:
        raise ValueError("USDA Maine office membership changed")
    return rows


def main():
    """Pin the official page and an unlabeled office packet."""
    data = RAW.read_bytes()
    source_manifest = {
        "kind": "us_nrcs_maine_directory_v1", "source_url": SOURCE,
        "captured_utc": (json.loads(SOURCE_MANIFEST.read_text())["captured_utc"]
                         if SOURCE_MANIFEST.exists() else datetime.now(timezone.utc).isoformat()),
        "raw_sha256": digest(data), "bytes": len(data),
        "training_eligible": False,
    }
    if SOURCE_MANIFEST.exists() and json.loads(SOURCE_MANIFEST.read_text()) != source_manifest:
        raise ValueError("USDA Maine source manifest changed")
    if not SOURCE_MANIFEST.exists():
        SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    rows = packet_rows()
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in rows).encode()
    manifest = {
        "kind": "us_nrcs_maine_offices_blind_v1", "cases": len(rows),
        "source_manifest_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(packet),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("USDA Maine packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != packet:
        raise ValueError("USDA Maine packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("USDA Maine packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(packet)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} USDA Maine offices frozen for blind review")


if __name__ == "__main__":
    main()
