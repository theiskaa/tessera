"""Freeze distinct US development cases from the existing reviewed gold."""

import collections
import glob
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/interim/review/gold-all-with-hosts.jsonl"
OUTPUT = ROOT / "data/interim/review/us-dev-v1.jsonl"
MANIFEST = ROOT / "data/interim/review/us-dev-v1.manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def raw_records():
    records = {}
    for path in (ROOT / "data/raw/review").rglob("*.json"):
        row = json.loads(path.read_text())
        name = row["id"]
        if name in records:
            raise ValueError(f"duplicate raw review id: {name}")
        records[name] = row
    return records


def training_keys():
    names = set()
    texts = set()
    paths = sorted(glob.glob(str(ROOT / "data/interim/silver/r*/train.jsonl")))
    paths.append(str(ROOT / "data/interim/silver/federal-register.jsonl"))
    for path in paths:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("country", "US") != "US":
                    continue
                names.add(row["id"])
                texts.add(digest(row["text"].encode()))
    return names, texts


def main():
    raw = raw_records()
    train_names, train_texts = training_keys()
    seen_names = set()
    seen_texts = {}
    retained = []
    hosts = collections.Counter()
    labels = collections.Counter()
    duplicates = []
    for line in SOURCE.read_text().splitlines():
        row = json.loads(line)
        if row["country"] != "US":
            continue
        name = row["name"]
        text = row["input"]
        if name in seen_names or name not in raw:
            raise ValueError(f"duplicate or missing US raw record: {name}")
        seen_names.add(name)
        if raw[name]["country"] != "US" or raw[name]["text"] != text:
            raise ValueError(f"US gold differs from raw source: {name}")
        text_hash = digest(text.encode())
        if name in train_names or text_hash in train_texts:
            raise ValueError(f"US gold overlaps existing silver: {name}")
        for span in row["expected"]:
            start, end = span["start"], span["end"]
            if start >= end or text.encode()[start:end].decode() != span["text"]:
                raise ValueError(f"invalid gold span: {name} {start}..{end}")
        previous = seen_texts.get(text_hash)
        if previous:
            if previous["expected"] != row["expected"]:
                raise ValueError(f"conflicting gold labels: {name}")
            duplicates.append(name)
            continue
        seen_texts[text_hash] = row
        retained.append(row)
        hosts[row["source_host"]] += 1
        labels.update(span["kind"] for span in row["expected"])
    if not retained:
        raise ValueError("no US development cases")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained).encode()
    manifest = {
        "kind": "us_development_gold",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE.read_bytes()),
        "gold_sha256": digest(data),
        "cases": len(retained),
        "source_hosts": dict(hosts),
        "labels": dict(labels),
        "duplicate_text_cases_removed": duplicates,
        "existing_silver_id_or_text_overlap": 0,
        "use": "development evaluation only; not training or independent release test",
    }
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"frozen {len(retained)} US development cases at {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
