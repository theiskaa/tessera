"""Copy synthetic detector shards, omitting documents with gold entity-name overlap.

The gold labels are used only to exclude training examples. Their text is never
added to a training shard. Run this after regenerating the synthetic corpus.
"""

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/processed/detector"
OUTPUT = ROOT / "data/processed/detector-no-gold-overlap"
GOLD = [
    ROOT / "data/interim/review/gold.jsonl",
    *(ROOT / f"data/interim/review/gold-r{round_no}.jsonl" for round_no in (2, 4, 6, 8)),
    ROOT / "data/interim/review/qa-sa-bounded-20260926/gold-v1.jsonl",
    ROOT / "data/interim/review/qa-ge-sda-confirmation-20260927/gold-v1.jsonl",
]
PINNED_GOLD = {
    "qa-sa-bounded-20260926": "2e1b7b80d68495c3efbc1db33b619fca4ada18eef91421dd75b331879d25f76c",
    "qa-ge-sda-confirmation-20260927": "35f76daa56fb2b6a061e5979d665291fc310677016a804086fde4ddbf39b47a4",
}
KINDS = {"person", "org", "address"}


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def gold_names():
    names = defaultdict(set)
    digest = hashlib.sha256()
    for path in GOLD:
        contents = path.read_bytes()
        expected = PINNED_GOLD.get(path.parent.name)
        if expected and hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"sealed gold changed: {path}")
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(contents)
        for line in contents.splitlines():
            row = json.loads(line)
            for entity in row["expected"]:
                if entity["kind"] in KINDS:
                    names[entity["kind"]].add(normalized(entity["text"]))
    return names, digest.hexdigest()


def filter_shard(path, names):
    table = pq.read_table(path)
    keep = []
    removed = Counter()
    for index, row in enumerate(table.select(["text", "entities_json"]).to_pylist()):
        encoded = row["text"].encode("utf-8")
        matching_kinds = set()
        for entity in json.loads(row["entities_json"]):
            kind = entity["kind"]
            if kind not in KINDS:
                continue
            value = encoded[entity["start"] : entity["end"]].decode("utf-8")
            if normalized(value) in names[kind]:
                matching_kinds.add(kind)
        if matching_kinds:
            removed.update(matching_kinds)
            removed["any"] += 1
        else:
            keep.append(index)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        pa.field(field.name, pa.string() if pa.types.is_string_view(field.type) else field.type)
        for field in table.schema
    )
    pq.write_table(table.cast(schema).take(pa.array(keep, type=pa.int64())), OUTPUT / path.name)
    return {"source_rows": table.num_rows, "kept_rows": len(keep), "removed": dict(removed)}


def main():
    names, gold_sha256 = gold_names()
    report = {"gold_sha256": gold_sha256, "name_counts": {k: len(v) for k, v in names.items()}}
    for path in sorted(SOURCE.glob("*.parquet")):
        report[path.name] = filter_shard(path, names)
    (OUTPUT / "overlap-filter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
