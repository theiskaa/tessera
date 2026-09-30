"""Add independently adjudicated FDA unit labels to the frozen US gold."""

import json
from pathlib import Path

from build_corrected_dev import OUT as BASE, digest
from build_full_notice_gold import validate_spans


ROOT = Path(__file__).resolve().parents[2]
ERRATA = ROOT / "bench/us/fixtures/us-development-gold-errata-v3.json"
OUT = ROOT / "data/interim/review/us-development-gold-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def corrected_rows_v3():
    """Apply reviewed additions after validating every original gold span."""
    errata = json.loads(ERRATA.read_text())
    if digest(BASE.read_bytes()) != errata["base_sha256"]:
        raise ValueError("US development v2 gold changed")
    rows = [json.loads(line) for line in BASE.read_text().splitlines()]
    by_name = {row["name"]: row for row in rows}
    if len(rows) != 196 or len(by_name) != len(rows):
        raise ValueError("US development case membership changed")
    corrections = errata["corrections"]
    if len(corrections) != 1 or corrections[0]["name"] != "2026-02972":
        raise ValueError("US development adjudication changed")
    for correction in corrections:
        row = by_name[correction["name"]]
        additions = validate_spans(row["name"], row["input"], correction["add"])
        if (len(additions) != 5 or
                any(span["text"] != "Dockets Management Staff" for span in additions)):
            raise ValueError("FDA unit adjudication changed")
        row["expected"] = validate_spans(
            row["name"], row["input"], row["expected"] + additions)
    for row in rows:
        validate_spans(row["name"], row["input"], row["expected"])
    return rows, errata


def main():
    """Freeze the corrected gold while retaining both earlier gold versions."""
    rows, errata = corrected_rows_v3()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    manifest = {
        "kind": "us_development_gold_corrected_v3",
        "cases": len(rows),
        "added_org_spans": len(errata["corrections"][0]["add"]),
        "base_sha256": digest(BASE.read_bytes()),
        "errata_sha256": digest(ERRATA.read_bytes()),
        "sha256": digest(data),
        "training_eligible": False,
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US development v3 gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US development v3 gold changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US development v3 inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} US development cases; five FDA unit labels corrected")


if __name__ == "__main__":
    main()
