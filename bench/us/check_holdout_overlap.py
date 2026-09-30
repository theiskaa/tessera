"""Find labeled entity leakage from prepared US data into held-out gold."""

import argparse
import collections
import json
import re
from pathlib import Path

from address_keys import address_keys
from active_sources import ACTIVE_SILVER
from check_ready import normalized, person_key
from freeze_full_notice_packet import SILVER_SOURCES
from org_aliases import active_aliases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = ROOT / "data/interim/review/us-full-notice-gold-v1.jsonl"
SILVER = ROOT / "data/interim/silver"
SYNTHETIC = ROOT / "data/processed/detector-us-v1"
CURRENT_SYNTHETIC = ROOT / "data/processed/detector-us-v2"
KINDS = {"person", "org", "address"}


def index_gold(rows, *, strict_aliases=False):
    surfaces = collections.defaultdict(lambda: collections.defaultdict(set))
    people = collections.defaultdict(set)
    addresses = collections.defaultdict(set)
    aliases = collections.defaultdict(set)
    for row in rows:
        for alias in active_aliases([row], strict=strict_aliases):
            aliases[alias].add(row["name"])
        for span in row["expected"]:
            kind = span["kind"]
            if kind not in KINDS:
                continue
            value = span["text"]
            surfaces[kind][normalized(value)].add(row["name"])
            if kind == "person" and person_key(value) is not None:
                people[person_key(value)].add(row["name"])
            if kind == "address":
                for key in address_keys(value):
                    addresses[key].add(row["name"])
    return surfaces, people, addresses, aliases


def collisions(rows, sources, *, strict_aliases=False):
    surfaces, people, addresses, aliases = index_gold(rows, strict_aliases=strict_aliases)
    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    for origin, text, spans in sources:
        source = text.encode()
        for span in spans:
            kind = span["kind"]
            if kind not in KINDS:
                continue
            value = source[span["start"]:span["end"]].decode()
            key = normalized(value)
            matched = set(surfaces[kind].get(key, ()))
            if kind == "person" and person_key(value) is not None:
                matched.update(people.get(person_key(value), ()))
            if kind == "address":
                for address in address_keys(value):
                    matched.update(addresses.get(address, ()))
            if kind == "org":
                matched.update(aliases.get(key, ()))
            for name in matched:
                hits[name][kind].add((origin, value))
    return hits


def contact_collisions(rows, sources):
    """Find exact email and phone overlaps, including short service numbers."""
    index = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in rows:
        for span in row["expected"]:
            kind = span["kind"]
            if kind in {"email", "phone"}:
                value = span["text"].casefold() if kind == "email" else re.sub(
                    r"\D", "", span["text"])
                index[kind][value].add(row["name"])
    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    for origin, text, spans in sources:
        source = text.encode()
        for span in spans:
            kind = span["kind"]
            if kind not in {"email", "phone"}:
                continue
            value = source[span["start"]:span["end"]].decode()
            key = value.casefold() if kind == "email" else re.sub(r"\D", "", value)
            for name in index[kind].get(key, ()):
                hits[name][kind].add((origin, value))
    return hits


def training_sources():
    for name in SILVER_SOURCES:
        path = SILVER / f"{name}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            yield name, row["text"], row["entities"]
    yield from synthetic_sources()


def current_training_sources(processed=SYNTHETIC):
    """Read the real silver files used by the current detector configuration."""
    for name, _ in ACTIVE_SILVER:
        path = SILVER / f"{name}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            yield name, row["text"], row["entities"]
    yield from synthetic_sources(processed)


def synthetic_sources(processed=SYNTHETIC):
    """Read prepared synthetic training and validation labels."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("pyarrow is required to inspect synthetic training") from error
    for split in ("train", "valid"):
        file = parquet.ParquetFile(processed / f"{split}.parquet")
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["text", "entities_json"]):
            for row in batch.to_pylist():
                yield f"synthetic_{split}", row["text"], json.loads(row["entities_json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.gold.read_text().splitlines()]
    hits = collisions(rows, current_training_sources(CURRENT_SYNTHETIC))
    contacts = contact_collisions(rows, current_training_sources(CURRENT_SYNTHETIC))
    for name, kinds in contacts.items():
        for kind, values in kinds.items():
            hits[name][kind].update(values)
    print(f"{len(hits)}/{len(rows)} held-out cases share labeled entities "
          "with prepared training or validation")
    for name, kinds in sorted(hits.items()):
        values = {kind: sorted({value for _, value in matches})
                  for kind, matches in kinds.items()}
        print(f"{name}: {json.dumps(values, ensure_ascii=False, sort_keys=True)}")
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
