"""Freeze source-backed US office chains for independent relation review."""

import collections
import json
import re
from pathlib import Path

from active_sources import ACTIVE_SILVER
from build_full_notice_gold import digest, validate_spans


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
RAW = ROOT / "data/raw/silver/federal-register"
OUT = ROOT / "data/raw/candidates/us-org-hierarchies-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
UNIT = re.compile(r"\b(?:office|branch|division|section|directorate|bureau)\b", re.I)
SOURCE_ID = re.compile(r"20\d{2}-\d{5}")
MAX_CASES = 48
YEAR_CAP = {"2024": 32, "2025": 16}


def candidates():
    """Read labeled real text and verify it against the captured notice."""
    inputs = {}
    result = []
    for name, _ in ACTIVE_SILVER:
        path = SILVER / f"{name}.jsonl"
        inputs[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        for line in path.read_text().splitlines():
            row = json.loads(line)
            url = row.get("source_url", "")
            if (not url.startswith("https://www.federalregister.gov/documents/") or
                    len(row["text"]) > 1500):
                continue
            source_bytes = row["text"].encode()
            org = [{**span, "text": source_bytes[span["start"]:span["end"]].decode()}
                   for span in row["entities"] if span["kind"] == "org"]
            if len(org) < 2:
                continue
            spans = validate_spans(row["id"], row["text"], org)
            if not any(UNIT.search(span["text"]) for span in spans):
                continue
            match = SOURCE_ID.search(url)
            if not match:
                raise ValueError(f"Federal Register URL lacks document number: {url}")
            source_id = row.get("source_id", match.group())
            raw_path = RAW / f"{source_id}.json"
            raw_bytes = raw_path.read_bytes()
            raw = json.loads(raw_bytes)
            if (raw["id"] != source_id or raw["url"] != url or
                    raw["text"].count(row["text"]) != 1):
                raise ValueError(f"silver office chain differs from raw source: {row['id']}")
            result.append({
                "name": f"us-org-hierarchy-{row['id']}",
                "source_id": source_id, "source_url": url,
                "raw_sha256": digest(raw_bytes),
                "silver_file": str(path.relative_to(ROOT)),
                "silver_id": row["id"],
                "input": row["text"], "org_entities": spans,
            })
    return result, inputs


def selected_rows():
    """Keep diverse source and organization-name combinations."""
    rows, inputs = candidates()
    ranked = sorted(rows, key=lambda row: (
        -len({span["text"].casefold() for span in row["org_entities"]}),
        digest(f"tessera-us-hierarchy-v1:{row['name']}".encode()),
    ))
    selected = []
    sources = set()
    signatures = set()
    years = collections.Counter()
    for row in ranked:
        if len(selected) >= MAX_CASES:
            break
        year = row["source_id"][:4]
        signature = frozenset(span["text"].casefold() for span in row["org_entities"])
        if (row["source_id"] in sources or signature in signatures or
                years[year] >= YEAR_CAP.get(year, 0)):
            continue
        selected.append(row)
        sources.add(row["source_id"])
        signatures.add(signature)
        years[year] += 1
    if (len(selected) < 35 or years["2024"] < 20 or years["2025"] < 10):
        raise ValueError(f"too few diverse real organization chains: {dict(years)}")
    selected.sort(key=lambda row: row["name"])
    return selected, inputs, years


def main():
    """Write a pinned, non-training packet for two independent passes."""
    rows, inputs, years = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_org_hierarchy_blind_v1",
        "cases": len(rows),
        "cases_by_year": dict(sorted(years.items())),
        "silver_input_sha256": dict(sorted(inputs.items())),
        "selected_raw_sha256": {row["source_id"]: row["raw_sha256"] for row in rows},
        "training_eligible": False,
        "intended_use": "independent_unit_parent_relation_review",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("organization hierarchy packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("organization hierarchy packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("organization hierarchy inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-backed organization-chain passages frozen: {dict(years)}")


if __name__ == "__main__":
    main()
