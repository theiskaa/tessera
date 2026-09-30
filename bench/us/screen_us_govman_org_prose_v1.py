"""Freeze source-pinned Government Manual prose for blind US organization review."""

import collections
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from active_sources import V4_SILVER


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/us-govman-2025-v1/source.xml"
SOURCE_MANIFEST = SOURCE.with_name("manifest.json")
OUT = ROOT / "data/raw/candidates/us-govman-org-prose-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EVALUATION = (
    ROOT / "data/interim/review/us-development-gold-v3.jsonl",
    ROOT / "data/interim/review/us-long-org-eval-gold-v3.jsonl",
    ROOT / "data/interim/review/us-doe-org-overviews-strict-gold-v1.jsonl",
)
SILVER = tuple(ROOT / f"data/interim/silver/{name}.jsonl"
               for name, _ in V4_SILVER) + (
    ROOT / "data/interim/silver/us-v5-reviewed-long-org-prose-v1.jsonl",
    ROOT / "data/interim/silver/us-reviewed-extra-long-org-prose-v2.jsonl",
    ROOT / "data/interim/silver/us-reviewed-two-paragraph-org-prose-v3.jsonl",
)
ORG_CUE = re.compile(
    r"\b(?:Department|Agency|Office|Board|Commission|Bureau|Committee|Service|"
    r"Administration|University|Division|Corporation|Institute|Center)\b"
)
WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def digest(data):
    """Return a SHA-256 digest of frozen bytes."""
    return hashlib.sha256(data).hexdigest()


def shingles(text):
    """Return normalized twelve-word fragments for source overlap checks."""
    words = WORD.findall(text.lower())
    return {tuple(words[i:i + 12]) for i in range(len(words) - 11)}


def existing_text(path):
    """Read training or evaluation text from a frozen JSONL file."""
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            yield row.get("input", row.get("text", ""))


def selected_rows():
    """Choose one reviewable paragraph from each of thirty distinct entities."""
    source_data = SOURCE.read_bytes()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (digest(source_data) != source_manifest["xml_sha256"] or
            source_manifest["training_eligible"] is not False or
            source_manifest["evaluation_eligible"] is not False):
        raise ValueError("Government Manual capture changed")
    root = ET.fromstring(source_data)
    if root.tag != "GovernmentManual":
        raise ValueError("unexpected Government Manual XML root")
    blocked = set().union(*(shingles(text) for path in EVALUATION + SILVER
                            for text in existing_text(path)))
    by_entity = []
    inventory = collections.Counter()
    for entity in root.findall("Entity"):
        entity_id = entity.get("EntityId")
        name = " ".join((entity.findtext("AgencyName") or "").split())
        category = " ".join((entity.findtext("Category") or "").split())
        if not entity_id or not name or name == "Department of Energy":
            continue
        options = []
        for number, paragraph in enumerate(
                entity.findall("./ProgramAndActivities//Paragraph"), start=1):
            source_text = "".join(paragraph.itertext())
            text = " ".join(source_text.split())
            cue_count = len(ORG_CUE.findall(text))
            if (not 500 <= len(text) <= 1600 or cue_count < 2 or
                    "\ufffd" in text or not text.endswith((".", "!", "?")) or
                    shingles(text) & blocked):
                continue
            inventory[category] += 1
            options.append({
                "name": f"us-govman-2025-entity-{entity_id}-paragraph-{number}",
                "country": "US", "input": text,
                "source": "us-government-manual-2025",
                "source_url": source_manifest["source_url"],
                "source_document_key": f"govinfo.gov:GOVMAN-2025-12-31:entity-{entity_id}",
                "source_group": f"govinfo.gov:GOVMAN-2025-12-31:entity-{entity_id}",
                "source_entity_id": entity_id,
                "source_entity_name": name,
                "source_category": category,
                "source_paragraph_number": number,
                "source_paragraph_sha256": digest(source_text.encode()),
                "source_xml_sha256": source_manifest["xml_sha256"],
                "org_cue_count": cue_count,
            })
        if options:
            options.sort(key=lambda row: (
                abs(len(row["input"]) - 1000), -row["org_cue_count"],
                digest(row["name"].encode())))
            by_entity.append(options[0])
    by_entity.sort(key=lambda row: (
        -row["org_cue_count"], abs(len(row["input"]) - 1000),
        digest(row["name"].encode())))
    selected = []
    selected_shingles = set()
    for row in by_entity:
        parts = shingles(row["input"])
        if parts & selected_shingles:
            continue
        selected.append(row)
        selected_shingles.update(parts)
        if len(selected) == 30:
            break
    if len(selected) != 30 or len({row["source_entity_id"] for row in selected}) != 30:
        raise ValueError(f"too few distinct Government Manual passages: {len(selected)}")
    selected.sort(key=lambda row: row["name"])
    return selected, inventory


def main():
    """Freeze an unlabeled packet while leaving training and evaluation disabled."""
    rows, inventory = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (SOURCE, SOURCE_MANIFEST, *EVALUATION, *SILVER)}
    manifest = {
        "kind": "us_govman_org_prose_blind_v1",
        "training_eligible": False,
        "evaluation_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_entities": len({row["source_entity_id"] for row in rows}),
        "categories": dict(sorted(collections.Counter(
            row["source_category"] for row in rows).items())),
        "candidate_inventory": dict(sorted(inventory.items())),
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("Government Manual packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("Government Manual packet changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("Government Manual packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(rows)} Government Manual passages from distinct entities frozen for review")


if __name__ == "__main__":
    main()
