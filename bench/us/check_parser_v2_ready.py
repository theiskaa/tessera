"""Verify the isolated US parser data before a future parser training run."""

import collections
import json
import re
from pathlib import Path

from address_keys import address_keys


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed/parser-us-v2"
SOURCE = ROOT / "data/manifests/libpostal-osm-addresses.json"
EXPECTED = {"train": 30000, "valid": 3000, "test": 3000}
DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south",
              "east", "west", "wst", "northeast", "northwest", "southeast",
              "southwest"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def street_city_key(row):
    """Match a street number and first name word where the source omits ZIP codes."""
    text = row["text"]
    parts = {label: text[start:end] for label, start, end in zip(
        row["span_label"], row["span_start"], row["span_end"])}
    if not all(label in parts for label in (0, 1, 5)):
        return None
    house = re.sub(r"[^a-z0-9-]", "", parts[0].casefold())
    road = [word for word in re.findall(r"[a-z0-9]+", parts[1].casefold())
            if word not in DIRECTIONS]
    city = " ".join(re.findall(r"[a-z0-9]+", parts[5].casefold()))
    if not house or not road or not city:
        return None
    return house, road[0], city


def main():
    """Check source identity, row counts, and source-location separation."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("check_parser_v2_ready needs pyarrow") from error

    manifest = json.loads((PROCESSED / "sample.json").read_text())
    source = json.loads(SOURCE.read_text())
    require(manifest["source"] == "libpostal-osm-addresses" and
            manifest["sha256"] == source["sha256"] and
            manifest["split_seed"] == 42 and
            manifest["filters"]["countries"] == ["US"] and
            manifest["checks"]["passed"] and
            all(value == 0 for key, value in manifest["checks"]["counts"].items()
                if key != "postcode_and_number_shared"),
            "US parser v2 source or preparation checks changed")
    prior_groups = set()
    prior_places = set()
    prior_street_cities = set()
    counts = collections.Counter()
    for split, expected in EXPECTED.items():
        file = parquet.ParquetFile(PROCESSED / f"{split}.parquet")
        groups = set()
        places = set()
        street_cities = set()
        ids = set()
        originals = 0
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["id", "group_id", "country", "text",
                                                "span_label", "span_start", "span_end",
                                                "augmented"]):
            for row in batch.to_pylist():
                require(row["country"] == "US" and row["id"] not in ids,
                        f"US parser {split} has a foreign or duplicate row")
                ids.add(row["id"])
                if row["augmented"]:
                    require(split == "train", "parser augmentation left training")
                    continue
                originals += 1
                group = row["group_id"]
                require(group not in prior_groups,
                        f"parser {split} source group entered an earlier split")
                groups.add(group)
                physical = address_keys(row["text"])
                require(not physical & prior_places,
                        f"parser {split} physical address entered an earlier split: {row['text']}")
                places.update(physical)
                street_city = street_city_key(row)
                require(street_city not in prior_street_cities,
                        f"parser {split} street and city entered an earlier split: {row['text']}")
                if street_city is not None:
                    street_cities.add(street_city)
        require(originals == expected and file.metadata.num_rows == len(ids),
                f"US parser {split} row count changed")
        counts[split] = originals
        prior_groups.update(groups)
        prior_places.update(places)
        prior_street_cities.update(street_cities)
    require(manifest["counts"]["US"]["train"] == counts["train"] and
            manifest["counts"]["US"]["valid"] == counts["valid"] and
            manifest["counts"]["US"]["test"] == counts["test"] and
            manifest["counts"]["US"]["augmented"] + counts["train"] ==
            parquet.ParquetFile(PROCESSED / "train.parquet").metadata.num_rows,
            "US parser v2 manifest count changed")
    print(f"US parser v2 data gate passed: {dict(counts)}, "
          f"{len(prior_places)} ZIP address keys, "
          f"{len(prior_street_cities)} street and city keys")


if __name__ == "__main__":
    main()
