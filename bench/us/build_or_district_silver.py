"""Build source-backed Oregon district names without held contact overlap."""

import hashlib
import json
from pathlib import Path

from check_ready import normalized, person_key


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "bench/us/fixtures/or-district-directory-v1.jsonl"
EVAL = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUT = ROOT / "data/interim/silver/us-or-districts-v1.jsonl"
MANIFEST = ROOT / "data/interim/silver/us-or-districts-v1.manifest.json"
SOURCE_SHA256 = "90c600f4ebd9bddd813e24c63b7dbb00421a35e3e4e156d59b1a10ed26203d21"
SOURCE_URL = "https://www.oregon.gov/ode/students-and-family/SpecialEducation/GeneralSupervision/Documents/districtsupportcontacts.pdf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = SOURCE.read_bytes()
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Oregon directory extraction changed")
    source_rows = [json.loads(line) for line in source.splitlines()]
    if len(source_rows) != 197 or len({row["district"] for row in source_rows}) != 197:
        raise ValueError("Oregon directory row count changed")
    eval_rows = [json.loads(line) for line in EVAL.read_text().splitlines()]
    held_orgs = {normalized(span["text"]) for row in eval_rows
                 for span in row["expected"] if span["kind"] == "org"}
    held_people = {person_key(span["text"]) for row in eval_rows
                   for span in row["expected"] if span["kind"] == "person"}
    rows = []
    for index, source_row in enumerate(source_rows):
        district = source_row["district"]
        if normalized(district) in held_orgs or person_key(source_row["specialist"]) in held_people:
            continue
        text = f"District: {district}"
        start = len("District: ".encode())
        rows.append({"id": f"or-district-{index:03}", "source": "oregon-ode-district-contacts-2026",
                     "source_url": SOURCE_URL,
                     "source_line": source_row["source_line"],
                     "country": "US", "text": text,
                     "entities": [{"kind": "org", "start": start,
                                   "end": start + len(district.encode())}]})
    if len(rows) != 66:
        raise ValueError(f"expected 66 disjoint Oregon districts, got {len(rows)}")
    output = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    OUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "source_checked_us_oregon_district_names",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "evaluation_gold_sha256": digest(EVAL.read_bytes()),
        "documents": len(rows), "labels": {"org": len(rows)},
        "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"built {len(rows)} Oregon district name examples")


if __name__ == "__main__":
    main()
