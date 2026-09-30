"""Extract reviewable table rows and paragraphs from pinned contact HTML."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

from freeze_official_contact_pages import OUT, PAGES


class ContactBlocks(HTMLParser):
    """Keep visible cell and paragraph boundaries for source-linked review."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row = None
        self.cell = None
        self.paragraph = None
        self.rows = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"} and self.row is not None:
            self.cell = []
        if tag == "p":
            self.paragraph = []
        if tag == "br":
            for block in (self.cell, self.paragraph):
                if block is not None:
                    block.append("\n")

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if tag == "p" and self.paragraph is not None:
            lines = (" ".join(line.split()) for line in
                     "".join(self.paragraph).splitlines())
            self.paragraphs.append("\n".join(line for line in lines if line))
            self.paragraph = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self.paragraph is not None:
            self.paragraph.append(data)


def blocks(name):
    """Verify the captured page and return its extracted blocks and provenance."""
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["kind"] != "us_official_contact_pages_v1" or manifest["training_eligible"]:
        raise ValueError("official contact capture state changed")
    if name not in PAGES or set(manifest["pages"]) != set(PAGES):
        raise ValueError("official contact source membership changed")
    item = manifest["pages"][name]
    path = OUT / f"{name}.html"
    data = path.read_bytes()
    raw_hash = hashlib.sha256(data).hexdigest()
    if (item["source_url"] != PAGES[name] or item["sha256"] != raw_hash or
            item["bytes"] != len(data)):
        raise ValueError(f"official contact page changed: {name}")
    parser = ContactBlocks()
    parser.feed(data.decode("utf-8"))
    return parser, {"source_url": item["source_url"], "raw_sha256": raw_hash,
                    "source_file": str(path.relative_to(Path(__file__).resolve().parents[2]))}
