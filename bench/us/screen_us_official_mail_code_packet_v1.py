"""Freeze official US mail code examples for independent blind labeling."""

import hashlib
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION, rows
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/candidates/us-official-mail-code-v1"
OUT = BASE / "blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
SOURCES = (
    ("jpl-michael-shao", "https://science.jpl.nasa.gov/people/shao/",
     "fb8369bd9f8ab7a3e1c8a3a28a59316d21c614a002cea073f3667d378e7166a6",
     r"Michael Shao\nAddress:.*?Fax:.*?818\.393\.0068", "m_s"),
    ("nasa-goddard-abshire", "https://espo.nasa.gov/atmosphere_2026/person/James_Abshire",
     "753afb4e3794801aa85f4787adf5f17b667f832badd9e72ba15d0ac6f4386549",
     r"James Abshire\nOrganization.*?United States", "mail_code"),
    ("nasa-marshall-goodman", "https://espo.nasa.gov/firesense/person/Michael_Goodman",
     "702a923b28ec96a2d7f03847ba30fa4acaf468ed7e3bdfc86ffb9ebae539fff4",
     r"Michael Goodman\nOrganization.*?United States", "mail_code"),
    ("nimh-neuropsychology", "https://www.nimh.nih.gov/research/research-conducted-at-nimh/research-areas/clinics-and-labs/ln/shn/contact",
     "d3ee46546e0a969c2185d27d7b7ca2f878390a5937dde7c12ae4f534063e807e",
     r"Andrew Mitz\nLaboratory.*?20892-4401", "mail_stop_code"),
)
BLOCKS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "dt", "dd", "tr", "td"}


def digest(data):
    """Hash captured bytes or a deterministic extracted window."""
    return hashlib.sha256(data).hexdigest()


class PageText(HTMLParser):
    """Preserve visible block boundaries from captured official HTML."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skipped = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skipped += 1
        elif not self.skipped and tag in BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skipped -= 1
        elif not self.skipped and tag in BLOCKS:
            self.parts.append("\n")

    def handle_data(self, value):
        if not self.skipped:
            self.parts.append(value)


def visible_text(raw):
    """Extract stable line-based page text without adding new words."""
    parser = PageText()
    parser.feed(raw.decode("utf-8"))
    return "\n".join(" ".join(line.split()) for line in
                     unescape("".join(parser.parts)).splitlines() if line.strip())


def packet_rows():
    """Select one exact visible-text window from each pinned official page."""
    result = []
    inputs = {}
    for source_id, url, expected_hash, pattern, category in SOURCES:
        path = BASE / "sources" / f"{source_id}.html"
        raw = path.read_bytes()
        if digest(raw) != expected_hash:
            raise ValueError(f"official page capture changed: {source_id}")
        inputs[str(path.relative_to(ROOT))] = expected_hash
        text = visible_text(raw)
        matches = list(re.finditer(pattern, text, re.S))
        if len(matches) != 1:
            raise ValueError(f"official page window is not unique: {source_id}")
        match = matches[0]
        excerpt = match.group(0)
        start = len(text[:match.start()].encode())
        end = len(text[:match.end()].encode())
        if ("\ufffd" in excerpt or "\n" not in excerpt or
                text.encode()[start:end].decode() != excerpt):
            raise ValueError(f"official page excerpt changed: {source_id}")
        result.append({
            "name": f"us-official-mail-code-{source_id}",
            "country": "US", "input": excerpt,
            "source_id": source_id, "source_url": url,
            "source_document_key": document_key({"source_url": url}),
            "source_group": f"official-page:{source_id}",
            "raw_path": str(path.relative_to(ROOT)),
            "raw_sha256": expected_hash,
            "source_text_sha256": digest(text.encode()),
            "source_start_byte": start, "source_end_byte": end,
            "category": category,
        })
    if len({row["source_document_key"] for row in result}) != 4:
        raise ValueError("mail code packet repeats an official page")
    return result, inputs


def main():
    """Keep captured examples blind and fail on any heldout overlap."""
    packet, inputs = packet_rows()
    gold = [row for path in EVALUATION for row in rows(path)]
    heldout = heldout_reasons(packet, gold)
    if heldout:
        raise ValueError(f"official mail code packet overlaps evaluation: {heldout}")
    heldout_fragments = set().union(*(shingles(row["input"], 12) for row in gold))
    if any(shingles(row["input"], 12) & heldout_fragments for row in packet):
        raise ValueError("mail code packet repeats evaluation prose")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in packet).encode()
    inputs.update({str(path.relative_to(ROOT)): digest(path.read_bytes())
                   for path in (*EVALUATION,
                                ROOT / "bench/us/check_us_blind_packet_holdout.py",
                                ROOT / "bench/us/org_aliases.py")})
    manifest = {
        "kind": "us_official_mail_code_blind_v1",
        "training_eligible": False,
        "evaluation_eligible": False,
        "label_status": "unlabeled",
        "cases": 4,
        "source_documents": 4,
        "categories": {row["source_id"]: row["category"] for row in packet},
        "input_sha256": dict(sorted(inputs.items())),
        "sha256": digest(data),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("mail code packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen mail code packet changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("mail code screening inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded_manifest)
    print("4 blind official mail code passages frozen from 3 government sites")


if __name__ == "__main__":
    main()
