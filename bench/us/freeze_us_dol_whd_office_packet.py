"""Freeze unlabeled DOL office, address, and staff-role blocks across states."""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from freeze_2025_ready_contact_train_candidates import EVALUATION
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-dol-whd-offices-v1/page.html"
SOURCE = "https://www.dol.gov/agencies/whd/contact/local-offices"
SOURCE_MANIFEST = RAW.with_name("manifest.json")
OUT = ROOT / "data/raw/candidates/us-dol-whd-offices-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
POSTCODE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
PHONE = re.compile(r"^[+()0-9][0-9()+.\s-]{8,}$")
ROLE = re.compile(r"^(?:Acting |Asst\. )?(?:District )?Directors?:|^Events, Education, and Outreach:")
EXCLUDED_STATES = {"Guam", "Puerto Rico"}


class OfficeBlocks(HTMLParser):
    """Extract visible heading and paragraph text from the offices section."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.depth = 0
        self.tag = None
        self.parts = []
        self.state = None
        self.current = None
        self.blocks = []
        self.office_counts = {}

    def handle_starttag(self, tag, attrs):
        if tag == "div" and dict(attrs).get("id") == "states":
            self.inside = True
            self.depth = 1
            return
        if self.inside and tag == "div":
            self.depth += 1
        if self.inside and tag in {"h2", "h3", "p"}:
            self.tag = tag
            self.parts = []
        if self.inside and tag == "br" and self.tag:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.inside and self.tag == tag:
            value = " ".join("".join(self.parts).split())
            if tag == "h2":
                self.state = value
                self.current = None
            elif tag == "h3":
                self.office_counts[self.state] = self.office_counts.get(self.state, 0) + 1
                self.current = {"state": self.state, "office": value,
                                "office_index": self.office_counts[self.state],
                                "paragraphs": []}
                self.blocks.append(self.current)
            elif tag == "p" and self.current is not None:
                self.current["paragraphs"].append(value)
            self.tag = None
            self.parts = []
        if self.inside and tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.inside = False

    def handle_data(self, data):
        if self.tag:
            self.parts.append(data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_blocks():
    """Verify the captured official page before extracting office blocks."""
    data = RAW.read_bytes()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (len(data) < 50_000 or b"September 4, 2026" not in data or
            manifest["kind"] != "us_dol_whd_offices_v1" or
            manifest["source_url"] != SOURCE or
            manifest["raw_sha256"] != digest(data) or
            manifest["bytes"] != len(data) or
            manifest["training_eligible"]):
        raise ValueError("DOL office source capture changed")
    parser = OfficeBlocks()
    parser.feed(data.decode("utf-8"))
    if len(parser.blocks) != 91:
        raise ValueError("DOL office page structure changed")
    return parser.blocks, digest(data)


def packet_rows():
    """Select one source-linked office per state before assigning labels."""
    blocks, raw_hash = source_blocks()
    rows = []
    states = set()
    for block in blocks:
        state = block["state"]
        if state in EXCLUDED_STATES or state in states:
            continue
        paragraphs = block["paragraphs"]
        addresses = [part for part in paragraphs if POSTCODE.search(part)]
        if len(addresses) != 1:
            raise ValueError(f"DOL office has unclear address: {state}")
        phones = [part for part in paragraphs if PHONE.fullmatch(part)]
        roles = [part for part in paragraphs if ROLE.search(part)]
        selected = [block["office"], addresses[0], *phones[:1], *roles[:2]]
        text = "\n".join(selected)
        if len(selected) < 4 or len(text) > 400:
            raise ValueError(f"DOL office has unclear contact block: {state}")
        slug = re.sub(r"[^a-z0-9]+", "-", state.casefold()).strip("-")
        rows.append({
            "name": f"us-dol-whd-office-{slug}", "country": "US",
            "input": text, "source_url": SOURCE,
            "source_state": state,
            "source_block": {"state": state, "office_index": block["office_index"]},
            "raw_sha256": raw_hash,
        })
        states.add(state)
    if len(rows) != 42 or len({row["input"] for row in rows}) != 42:
        raise ValueError(f"DOL blind office membership changed: {len(rows)}")
    return rows


def main():
    """Pin the current page and its blind office sample."""
    data = RAW.read_bytes()
    source_manifest = {
        "kind": "us_dol_whd_offices_v1", "source_url": SOURCE,
        "captured_utc": (json.loads(SOURCE_MANIFEST.read_text())["captured_utc"]
                         if SOURCE_MANIFEST.exists() else datetime.now(timezone.utc).isoformat()),
        "raw_sha256": digest(data), "bytes": len(data),
        "training_eligible": False,
    }
    if SOURCE_MANIFEST.exists() and json.loads(SOURCE_MANIFEST.read_text()) != source_manifest:
        raise ValueError("DOL office source manifest changed")
    if not SOURCE_MANIFEST.exists():
        SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    rows = packet_rows()
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in rows).encode()
    manifest = {
        "kind": "us_dol_whd_offices_blind_v1", "cases": len(rows),
        "source_manifest_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(packet),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("DOL office packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != packet:
        raise ValueError("DOL office packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("DOL office packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(packet)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} DOL offices frozen for blind review")


if __name__ == "__main__":
    main()
