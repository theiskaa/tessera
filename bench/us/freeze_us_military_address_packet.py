"""Freeze unlabeled military and federal mailing layouts from a NARA page."""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from freeze_2025_ready_contact_train_candidates import EVALUATION
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-official-military-addresses-v1/page.html"
SOURCE = "https://www.archives.gov/personnel-records-center/address-list"
OUT = ROOT / "data/raw/candidates/us-military-addresses-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ZIP = re.compile(r"\b[A-Z]{2} \d{5}(?:-\d{4})?\b")
LIST_ITEM = re.compile(r"<li\b[^>]*>(.*?)</li>", re.I | re.S)


class ListText(HTMLParser):
    """Keep line breaks printed by the source list items."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packet_rows():
    """Extract each short, source-indexed list item with a US postal address."""
    raw = RAW.read_bytes()
    if len(raw) < 10000 or b"Agencies Address List" not in raw:
        raise ValueError("NARA address page capture is incomplete")
    rows = []
    for index, match in enumerate(LIST_ITEM.finditer(raw.decode("utf-8"))):
        fragment = match.group(1)
        if len(fragment) > 1000 or "<ul" in fragment:
            continue
        parser = ListText()
        parser.feed(fragment)
        text = parser.text()
        if not ZIP.search(text) or len(text.splitlines()) < 3:
            continue
        rows.append({
            "name": f"us-military-address-{index:03d}",
            "country": "US", "input": text,
            "source_url": SOURCE,
            "raw_sha256": digest(raw),
            "source_block": {"kind": "list_item", "index": index},
        })
    if len(rows) != 14 or len({row["input"] for row in rows}) != 14:
        raise ValueError(f"NARA address packet membership changed: {len(rows)}")
    return rows


def main():
    """Pin source text for two independent label passes."""
    rows = packet_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    captured_utc = (json.loads(MANIFEST.read_text())["captured_utc"] if MANIFEST.exists()
                    else datetime.fromtimestamp(RAW.stat().st_mtime, timezone.utc).isoformat())
    manifest = {
        "kind": "us_military_address_blind_v1",
        "cases": len(rows), "source_url": SOURCE,
        "raw_sha256": digest(RAW.read_bytes()),
        "captured_utc": captured_utc,
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("military address packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("military address packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("military address packet input changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} NARA address passages frozen for blind review")


if __name__ == "__main__":
    main()
