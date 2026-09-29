"""Accept only source-backed Federal Register paragraphs with two exact blind reviews."""

import collections
import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import GOLD, RAW, excluded_surface_pattern, lines
from build_r23_reviewed_org import checked_labels
from build_silver import normalize_surface
from check_ready import person_key
from freeze_federal_org_packet import FROZEN_SHA256
from org_aliases import active_aliases
from silver_dedupe import NearDuplicateIndex


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
PACKET = SILVER / "r23/us-raw-org-blind-v1.jsonl"
REVIEWS = (SILVER / "r23/us-raw-org-review-a-v1.jsonl",
           SILVER / "r23/us-raw-org-review-b-v1.jsonl")
REVIEW_SHA256 = (
    "9df071a26f13357c06d604c9707820b42bdf5610bbacfc1be94b02d8e77a07fc",
    "f8ad7ed390424af053eab4a3c38ec5fe980b7630212927d6c796ce0b9e4e8546",
)
PREVIOUS_OUT = SILVER / "us-federal-org-reviewed-v1.jsonl"
OUT = SILVER / "us-federal-org-reviewed-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXISTING = (
    "us-reviewed-strict-v1", "us-house-staff-v1", "us-house-district-offices-v1",
    "us-reviewed-contact-snippets-v1", "us-reviewed-org-paragraphs-v1",
    "us-r23-reviewed-org-v1", "us-park-contacts-v1", "us-or-districts-v1",
)
UNSAFE_NEGATIVES = {
    "raw-us-org-2024-31395-5",
    "raw-us-org-2024-28719-4",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def by_name(rows):
    result = {row["name"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate blind case")
    return result


def main():
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha:
        raise ValueError("previous federal organization silver changed")
    if digest(PACKET.read_bytes()) != FROZEN_SHA256:
        raise ValueError("raw blind packet changed")
    packet = by_name(lines(PACKET))
    for path, expected in zip(REVIEWS, REVIEW_SHA256):
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"blind review changed: {path.name}")
    passes = [by_name(lines(path)) for path in REVIEWS]
    if any(set(review) != set(packet) for review in passes):
        raise ValueError("blind review omitted or added a paragraph")
    gold_bytes = GOLD.read_bytes()
    gold = lines(GOLD)
    forbidden = excluded_surface_pattern(gold)
    org_aliases = active_aliases(gold)
    people = {key for row in gold for span in row["expected"]
              if span["kind"] == "person" and (key := person_key(span["text"]))}
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    prior_rows = [row for name in EXISTING
                  for row in lines(SILVER / f"{name}.jsonl")]
    seen = {digest(row["text"].encode()) for row in prior_rows}
    near = NearDuplicateIndex()
    for row in prior_rows:
        near.add(row)
    kept = []
    dropped = collections.Counter()
    for name, case in packet.items():
        text = case["input"]
        reviews = [review[name] for review in passes]
        labels = [checked_labels(text, review) for review in reviews]
        if any(review.get("uncertain") for review in reviews):
            dropped["uncertain"] += 1
            continue
        if labels[0] != labels[1]:
            dropped["disagreement"] += 1
            continue
        if name in UNSAFE_NEGATIVES:
            dropped["unsafe_negative"] += 1
            continue
        source_id = name.removeprefix("raw-us-org-").rsplit("-", 1)[0]
        raw = json.loads((RAW / f"{source_id}.json").read_text())
        if case["source_url"] != raw["url"] or text not in raw["text"]:
            raise ValueError(f"reviewed paragraph lost source: {name}")
        if forbidden.search(normalize_surface(text)) or address_keys(text) & addresses:
            raise ValueError(f"reviewed paragraph overlaps evaluation: {name}")
        source = text.encode()
        if any(normalize_surface(source[start:end].decode()) in org_aliases
               for kind, start, end in labels[0] if kind == "org"):
            dropped["org_alias"] += 1
            continue
        if any(person_key(source[start:end].decode()) in people
               for kind, start, end in labels[0] if kind == "person"):
            dropped["person_alias"] += 1
            continue
        text_hash = digest(source)
        if text_hash in seen:
            dropped["duplicate_text"] += 1
            continue
        result = {"id": name, "source": "federal-register",
                  "source_url": case["source_url"], "source_group": case["group"],
                  "country": "US", "text": text,
                  "entities": [{"kind": kind, "start": start, "end": end}
                               for kind, start, end in labels[0]]}
        if near.prior(result):
            dropped["near_duplicate"] += 1
            continue
        seen.add(text_hash)
        near.add(result)
        kept.append(result)
    counts = collections.Counter(span["kind"] for row in kept for span in row["entities"])
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept).encode()
    manifest = {"kind": "us_federal_org_reviewed_v2",
                                    "supersedes_sha256": previous_sha,
                                    "packet_sha256": FROZEN_SHA256,
                                    "review_sha256": {path.name: digest(path.read_bytes())
                                                      for path in REVIEWS},
                                    "evaluation_gold_sha256": digest(gold_bytes),
                                    "documents": len(kept), "labels": dict(counts),
                                    "dropped": dict(dropped), "sha256": digest(data)}
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("federal organization silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("federal organization silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("federal organization silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} agreed US paragraphs with {dict(counts)} labels; {dict(dropped)} dropped")


if __name__ == "__main__":
    main()
