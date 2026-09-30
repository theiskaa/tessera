"""Match entity names and contact details across training and evaluation."""

import collections
import json
import re
import unicodedata

from address_keys import address_keys
from org_aliases import active_aliases


KINDS = {"person", "org", "address"}


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())

def person_key(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    if "," in value:
        surname, given = value.split(",", 1)
        given_parts = re.findall(r"[a-z]+", given)
        surname_parts = re.findall(r"[a-z]+", surname)
        return (given_parts[0], surname_parts[-1]) if given_parts and surname_parts else None
    parts = re.findall(r"[a-z]+", value)
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        parts.pop()
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


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

def synthetic_sources(processed):
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
