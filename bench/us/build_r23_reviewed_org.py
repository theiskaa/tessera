"""Accept only fully agreed, source-backed annotations from two blind reviews."""

import collections
import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import GOLD, RAW, excluded_surface_pattern, lines
from build_silver import normalize_surface
from check_ready import person_key
from freeze_r23_org_packet import FROZEN_SHA256 as FIRST_PACKET_SHA256
from freeze_r23_org_packet_remaining import FROZEN_SHA256 as SECOND_PACKET_SHA256
from org_aliases import active_aliases
from silver_dedupe import NearDuplicateIndex


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
PACKETS = (
    (SILVER / "r23/us-org-blind-v1.jsonl", FIRST_PACKET_SHA256,
     (SILVER / "r23/us-org-review-a-v1.jsonl",
      SILVER / "r23/us-org-review-b-v1.jsonl"),
     ("ae9cf805c773509b299f7472df24dedb9351ae90f2fefd9ee2013b2c4ac41aee",
      "b65f181a1f9d06efe7a7c579d2cf91a8e3faaeb02416e113f1ad1126e4cc503e")),
    (SILVER / "r23/us-org-blind-v2.jsonl", SECOND_PACKET_SHA256,
     (SILVER / "r23/us-org-review-a-v2.jsonl",
      SILVER / "r23/us-org-review-b-v2.jsonl"),
     ("914247678929c8a42c9f7ef146671c691896b302e33873e4090c79e5740d716e",
      "b0ddbe0fdd732a370caa04ff49285f30ed592fb9ae4fca351e1b65e6f8631d8b")),
)
OUT = SILVER / "us-r23-reviewed-org-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXISTING = (
    "us-reviewed-strict-v1", "us-house-staff-v1", "us-house-district-offices-v1",
    "us-reviewed-contact-snippets-v1", "us-reviewed-org-paragraphs-v1",
    "us-park-contacts-v1", "us-or-districts-v1",
)
KINDS = {"person", "org", "address"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def by_name(rows):
    result = {}
    for row in rows:
        name = row["name"]
        if name in result:
            raise ValueError(f"duplicate review name: {name}")
        result[name] = row
    return result


def checked_labels(text, review):
    source = text.encode()
    labels = []
    for span in review["entities"]:
        start, end = span["start"], span["end"]
        if (span["kind"] not in KINDS or not isinstance(start, int) or
                not isinstance(end, int) or not 0 <= start < end <= len(source)):
            raise ValueError(f"invalid blind span: {review['name']}")
        value = source[start:end].decode()
        if value != span["text"]:
            raise ValueError(f"blind span text differs: {review['name']}")
        labels.append((span["kind"], start, end))
    labels.sort(key=lambda item: (item[1], item[2], item[0]))
    if any(left[2] > right[1] for left, right in zip(labels, labels[1:])):
        raise ValueError(f"overlapping blind spans: {review['name']}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate blind span: {review['name']}")
    return labels


def main():
    gold_bytes = GOLD.read_bytes()
    gold = lines(GOLD)
    gold_texts = [row["input"] for row in gold]
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
    packet_hashes = {}
    review_hashes = {}
    for packet_path, expected, review_paths, expected_reviews in PACKETS:
        packet_bytes = packet_path.read_bytes()
        if digest(packet_bytes) != expected:
            raise ValueError(f"blind packet changed: {packet_path.name}")
        packet_hashes[packet_path.name] = expected
        packet = by_name(lines(packet_path))
        for path, expected_review in zip(review_paths, expected_reviews):
            review_hash = digest(path.read_bytes())
            if review_hash != expected_review:
                raise ValueError(f"blind review changed: {path.name}")
            review_hashes[path.name] = review_hash
        passes = [by_name(lines(path)) for path in review_paths]
        if any(set(review) != set(packet) for review in passes):
            raise ValueError("blind review omitted or added a paragraph")
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
            if not any(kind == "org" for kind, _, _ in labels[0]):
                dropped["no_org"] += 1
                continue
            source_id = name.removeprefix("r23-us-org-").rsplit("-", 1)[0]
            raw = json.loads((RAW / f"{source_id}.json").read_text())
            if case["source_url"] != raw["url"] or text not in raw["text"]:
                raise ValueError(f"reviewed paragraph lost source: {name}")
            if any(gold_text in raw["text"] for gold_text in gold_texts):
                raise ValueError(f"reviewed page overlaps evaluation: {name}")
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
                      "source_url": case["source_url"], "country": "US",
                      "text": text,
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
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_r23_reviewed_org",
                                    "packet_sha256": packet_hashes,
                                    "review_sha256": review_hashes,
                                    "evaluation_gold_sha256": digest(gold_bytes),
                                    "documents": len(kept), "labels": dict(counts),
                                    "dropped": dict(dropped), "sha256": digest(data)},
                                   indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} agreed US paragraphs with {dict(counts)} labels; {dict(dropped)} dropped")


if __name__ == "__main__":
    main()
