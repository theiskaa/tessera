"""Apply reviewed errata to the US development gold without changing its original."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
SOURCES = (
    REVIEW / "us-dev-v1.jsonl",
    REVIEW / "us-office-eval-v1.jsonl",
    REVIEW / "us-staff-challenge-v1.jsonl",
    REVIEW / "us-park-address-challenge-v1.jsonl",
)
ERRATA = ROOT / "bench/us/fixtures/us-development-gold-errata-v2.json"
OUT = REVIEW / "us-development-gold-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def corrected_rows():
    base = b"".join(path.read_bytes() for path in SOURCES)
    errata = json.loads(ERRATA.read_text())
    if digest(base) != errata["base_sha256"]:
        raise ValueError("development gold differs from the reviewed errata base")
    rows = [json.loads(line) for line in base.splitlines()]
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("duplicate development gold case")
    revised = set()
    for correction in errata["corrections"]:
        name = correction["name"]
        if name in revised or name not in by_name:
            raise ValueError(f"duplicate or unknown errata case: {name}")
        revised.add(name)
        row = by_name[name]
        source = row["input"].encode()
        labels = row["expected"]
        for span in correction.get("remove", []):
            if span not in labels:
                raise ValueError(f"errata removal is absent: {name} {span}")
            labels.remove(span)
        for span in correction.get("add", []):
            if source[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"errata addition differs from source: {name} {span}")
            labels.append(span)
        labels.sort(key=lambda span: (span["start"], span["end"]))
    for row in rows:
        source = row["input"].encode()
        labels = row["expected"]
        for span in labels:
            if (span["start"] >= span["end"] or
                    source[span["start"]:span["end"]].decode() != span["text"]):
                raise ValueError(f"invalid corrected span: {row['name']} {span}")
        for left, right in zip(labels, labels[1:]):
            if left["end"] > right["start"]:
                raise ValueError(f"overlapping corrected labels: {row['name']}")
    return rows, base, errata


def main():
    rows, base, errata = corrected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("corrected development gold changed after freezing")
    manifest = {
        "kind": "us_development_gold_corrected",
        "cases": len(rows),
        "corrections": len(errata["corrections"]),
        "base_sha256": digest(base),
        "errata_sha256": digest(ERRATA.read_bytes()),
        "sources_sha256": {path.name: digest(path.read_bytes()) for path in SOURCES},
        "sha256": digest(data),
        "use": "development only; the original gold and earlier scores remain intact",
    }
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} corrected development cases: {digest(data)}")


if __name__ == "__main__":
    main()
