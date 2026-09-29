"""Freeze complete 2025 notice excerpts for blind US training-data review."""

import collections
import hashlib
import json
import re
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import SOURCE_ARTIFACT, family, normalize_surface
from freeze_full_notice_packet import (SILVER_SOURCES, folded_words,
                                       has_person_alias, no_key_address_phrases,
                                       person_alias_patterns)
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/silver/federal-register"
SILVER = ROOT / "data/interim/silver"
REVIEW = ROOT / "data/interim/review"
PACKET_DIR = SILVER / "r24"
SNAPSHOT = PACKET_DIR / "us-2025-source-snapshot-v1.json"
OUT = PACKET_DIR / "us-2025-context-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
GOLD = (REVIEW / "us-eval-exclusions-v1.jsonl",
        REVIEW / "us-development-gold-v2.jsonl")
MAX_WORDS = 600
MIN_WORDS = 150


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_snapshot():
    if SNAPSHOT.exists():
        snapshot = json.loads(SNAPSHOT.read_text())
    else:
        files = sorted(RAW.glob("2025-*.json"))
        snapshot = {path.stem: digest(path.read_bytes()) for path in files}
        if not snapshot:
            raise ValueError("no 2025 Federal Register source capture")
        PACKET_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    for source_id, expected in snapshot.items():
        path = RAW / f"{source_id}.json"
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"2025 source changed: {source_id}")
    return snapshot


def chunks(text):
    current = []
    words = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        count = len(paragraph.split())
        if words + count > MAX_WORDS and current:
            yield "\n\n".join(current)
            current, words = [], 0
        if count > MAX_WORDS:
            continue
        current.append(paragraph)
        words += count
    if current:
        yield "\n\n".join(current)


def selected_cases():
    snapshot = source_snapshot()
    gold = [row for path in GOLD for row in lines(path)]
    forbidden = excluded_surface_pattern(gold)
    aliases = active_aliases(gold)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True))) if aliases else None)
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    address_phrases = no_key_address_phrases(gold)
    people = person_alias_patterns(gold)
    train_near = NearTextIndex()
    eval_near = NearTextIndex()
    packet_near = NearTextIndex()
    for name in SILVER_SOURCES:
        for row in lines(SILVER / f"{name}.jsonl"):
            train_near.add(row["id"], row["text"])
    for row in gold:
        eval_near.add(row["name"], row["input"])
    counts = collections.Counter()
    selected = []
    families = collections.Counter()
    for source_id in sorted(snapshot):
        raw = json.loads((RAW / f"{source_id}.json").read_text())
        if (raw["id"] != source_id or raw["source"] != "federal-register" or
                not raw["date"].startswith("2025-") or
                not raw["url"].startswith("https://www.federalregister.gov/documents/") or
                f"/{source_id}/" not in raw["url"] or
                raw["text_url"] !=
                f"https://www.govinfo.gov/content/pkg/FR-{raw['date']}/html/{source_id}.htm"):
            raise ValueError(f"2025 source metadata differs: {source_id}")
        if SOURCE_ARTIFACT.search(raw["text"]) or "\ufffd" in raw["text"]:
            counts["source_artifact"] += 1
            continue
        group = family(raw["url"])
        if families[group] >= 2:
            counts["source_family_cap"] += 1
            continue
        for number, text in enumerate(chunks(raw["text"]), 1):
            counts["chunks"] += 1
            if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
                counts["length"] += 1
                continue
            normalized = normalize_surface(text)
            if forbidden.search(normalized):
                counts["evaluation_surface"] += 1
                continue
            if alias_pattern is not None and alias_pattern.search(normalized):
                counts["evaluation_alias"] += 1
                continue
            if address_keys(text) & addresses:
                counts["evaluation_address"] += 1
                continue
            if has_person_alias(text, people):
                counts["evaluation_person_alias"] += 1
                continue
            words = folded_words(text)
            if any(tuple(words[index:index + 4]) in address_phrases
                   for index in range(len(words) - 3)):
                counts["evaluation_partial_address"] += 1
                continue
            if eval_near.prior(text):
                counts["evaluation_near_text"] += 1
                continue
            if train_near.prior(text):
                counts["training_near_text"] += 1
                continue
            if packet_near.prior(text):
                counts["packet_near_text"] += 1
                continue
            packet_near.add(source_id, text)
            families[group] += 1
            selected.append({
                "name": f"us-2025-context-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": raw["url"], "raw_sha256": snapshot[source_id],
                "input": text,
            })
            break
    return selected, counts, snapshot


def main():
    selected, counts, snapshot = selected_cases()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2025 context packet changed")
    manifest = {
        "kind": "us_2025_context_blind_packet",
        "intended_use": "training_candidate_after_two_blind_reviews",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in GOLD},
        "source_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                          for name in SILVER_SOURCES},
        "capture_count": len(snapshot),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2025 packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind 2025 notice excerpts frozen: {digest(data)}")


if __name__ == "__main__":
    main()
