"""Recover reviewed Federal Register paragraphs with unseen organization labels."""

import collections
import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import (CANDIDATE, CANDIDATE_MANIFEST, GOLD, RAW,
                                    SOURCE_ARTIFACT, SOURCE_NOISE,
                                    excluded_surface_pattern, lines, paragraphs)
from build_silver import REVIEW_EXCLUDED, legacy_cutoff, normalize_surface
from check_ready import person_key
from silver_dedupe import NearDuplicateIndex


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
OUT = SILVER / "us-reviewed-org-paragraphs-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    candidate_bytes = CANDIDATE.read_bytes()
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    if digest(candidate_bytes) != candidate_manifest["sha256"]:
        raise ValueError("reviewed candidate changed")
    gold_bytes = GOLD.read_bytes()
    gold = lines(GOLD)
    forbidden = excluded_surface_pattern(gold)
    gold_people = {person_key(span["text"]) for row in gold
                   for span in row["expected"] if span["kind"] == "person"}
    gold_addresses = set().union(*(address_keys(span["text"]) for row in gold
                                    for span in row["expected"] if span["kind"] == "address"))
    previous = {digest(row["text"].encode()) for filename in
                ("us-reviewed-strict-v1.jsonl", "us-reviewed-contact-snippets-v1.jsonl")
                for row in lines(SILVER / filename)}
    near = NearDuplicateIndex()
    for filename in ("us-reviewed-strict-v1.jsonl", "us-reviewed-contact-snippets-v1.jsonl"):
        for row in lines(SILVER / filename):
            near.add(row)
    seen = set(previous)
    kept = []
    skipped = collections.Counter()
    for row in lines(CANDIDATE):
        if row["id"] in REVIEW_EXCLUDED:
            continue
        raw = json.loads((RAW / f"{row['id']}.json").read_text())
        source = row["text"].encode()
        for number, (start, end) in enumerate(paragraphs(source), 1):
            piece = source[start:end]
            text = piece.decode()
            spans = [span for span in row["entities"]
                     if start <= span["start"] and span["end"] <= end]
            if not any(span["kind"] == "org" for span in spans):
                continue
            if legacy_cutoff(raw) and end == len(source):
                skipped["legacy_cutoff_tail"] += 1
                continue
            if len(piece) < 80 or len(piece) > 2500 or not text.rstrip().endswith((".", "!", "?")):
                skipped["paragraph_shape"] += 1
                continue
            if any(span["start"] < end and start < span["end"] and span not in spans
                   for span in row["entities"]):
                skipped["crosses_paragraph"] += 1
                continue
            if text not in raw["text"] or not raw.get("url"):
                skipped["source_mismatch"] += 1
                continue
            if SOURCE_ARTIFACT.search(text) or SOURCE_NOISE.search(text):
                skipped["source_artifact"] += 1
                continue
            if forbidden.search(normalize_surface(text)):
                skipped["evaluation_surface"] += 1
                continue
            if any(person_key(source[span["start"]:span["end"]].decode()) in gold_people
                   for span in spans if span["kind"] == "person"):
                skipped["evaluation_person_alias"] += 1
                continue
            if any(address_keys(source[span["start"]:span["end"]].decode()) & gold_addresses
                   for span in spans if span["kind"] == "address"):
                skipped["evaluation_address"] += 1
                continue
            key = digest(piece)
            if key in seen:
                skipped["duplicate_text"] += 1
                continue
            adjusted = [{"kind": span["kind"], "start": span["start"] - start,
                         "end": span["end"] - start} for span in spans]
            if any(not piece[span["start"]:span["end"]] for span in adjusted):
                raise ValueError(f"empty label: {row['id']}")
            result = {"id": f"{row['id']}-org-paragraph-{number}",
                      "source": "federal-register", "source_url": raw["url"],
                      "country": "US", "text": text, "entities": adjusted}
            if near.prior(result):
                skipped["near_duplicate"] += 1
                continue
            seen.add(key)
            near.add(result)
            kept.append(result)
    counts = collections.Counter(span["kind"] for row in kept for span in row["entities"])
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept).encode()
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_reviewed_org_paragraphs",
        "source_candidate_sha256": digest(candidate_bytes),
        "evaluation_gold_sha256": digest(gold_bytes),
        "documents": len(kept), "labels": dict(counts),
        "skipped": dict(skipped), "sha256": digest(data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"built {len(kept)} reviewed org paragraphs with {dict(counts)} labels")


if __name__ == "__main__":
    main()
