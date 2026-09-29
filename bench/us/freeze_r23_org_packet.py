"""Make a blind, evaluation-disjoint packet from singly reviewed US notices."""

import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import (GOLD, RAW, SOURCE_ARTIFACT, SOURCE_NOISE,
                                    excluded_surface_pattern, lines, paragraphs)
from build_silver import normalize_surface


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/interim/silver/r23/train.jsonl"
OUT = ROOT / "data/interim/silver/r23/us-org-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
TARGET = 60
FROZEN_SHA256 = "707f1e1fa4c9ff49b43649fcf695a31770a3e8dfc65b263fcf603b2de9620c43"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selected_cases():
    gold = lines(GOLD)
    gold_texts = [row["input"] for row in gold]
    forbidden = excluded_surface_pattern(gold)
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    candidates = []
    for row in lines(SOURCE):
        if row["country"] != "US":
            continue
        raw = json.loads((RAW / f"{row['id']}.json").read_text())
        if any(text in raw["text"] for text in gold_texts):
            continue
        source = row["text"].encode()
        for number, (start, end) in enumerate(paragraphs(source), 1):
            piece = source[start:end]
            text = piece.decode()
            labels = [span for span in row["entities"]
                      if start <= span["start"] and span["end"] <= end]
            orgs = sum(span["kind"] == "org" for span in labels)
            if (not orgs or not 80 <= len(piece) <= 800
                    or not text.lstrip()[:1].isupper()
                    or not text.rstrip().endswith((".", "?", "!"))
                    or text not in raw["text"] or not raw.get("url")
                    or SOURCE_ARTIFACT.search(text) or SOURCE_NOISE.search(text)
                    or forbidden.search(normalize_surface(text))
                    or address_keys(text) & addresses
                    or any(span["start"] < end and start < span["end"] and span not in labels
                           for span in row["entities"])):
                continue
            candidates.append((orgs, digest(piece), row["id"], number, text, raw["url"]))
    candidates.sort(key=lambda item: (-min(item[0], 3), item[1]))
    selected = []
    source_ids = set()
    texts = set()
    for _, text_hash, source_id, number, text, source_url in candidates:
        if source_id in source_ids or text_hash in texts:
            continue
        source_ids.add(source_id)
        texts.add(text_hash)
        selected.append({"name": f"r23-us-org-{source_id}-{number}",
                         "country": "US", "input": text, "source_url": source_url})
    return selected


def main():
    selected = selected_cases()[:TARGET]
    if len(selected) != TARGET:
        raise ValueError(f"only {len(selected)} blind paragraphs available")
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError(f"blind US organization packet differs: {digest(data)}")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("blind US organization packet changed")
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_r23_blind_org_packet",
                                    "source_sha256": digest(SOURCE.read_bytes()),
                                    "evaluation_sha256": digest(GOLD.read_bytes()),
                                    "cases": len(selected), "sha256": digest(data)},
                                   indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind US notice paragraphs frozen: {digest(data)}")


if __name__ == "__main__":
    main()
