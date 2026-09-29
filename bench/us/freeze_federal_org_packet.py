"""Freeze official Federal Register paragraphs for independent blind labeling."""

import hashlib
import json
import re
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import (GOLD, RAW, SOURCE_ARTIFACT, SOURCE_NOISE,
                                    excluded_surface_pattern, lines, paragraphs)
from build_silver import normalize_surface


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
OUT = SILVER / "r23/us-raw-org-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
FROZEN_SHA256 = "c14d3acc3baff64a01248df3d0dc4dce90efe7b43f37df70e76d735d9fcfd2e4"
EXCLUDED_TEXT = ROOT / "bench/us/fixtures/federal-org-existing-text-v1.json"
EXCLUDED_TEXT_SHA256 = "bceec0b4c04feb9e2a380947e05994a08be55391a90a9a5f48bc4ea3c4d55d53"
REVIEWED_SOURCE_SHA256 = {
    "train.jsonl": "e1b53f7474b84c31b11216f8dfb52966397547179577f727f2dad238ecdffd47",
    "us-reviewed-candidate-v1.jsonl": "e8fb797b707016c6eab74c81ba82f2d4a99d4396aa7d4473e9b07cfa97004fcc",
}
GROUP_COUNTS = {"acronym": 90, "contact": 50, "narrative": 15}
ORG_CUE = re.compile(
    r"\b(?:(?:(?!The\b|This\b)[A-Z][A-Za-z.&'-]*\s+){1,6}(?:Department|Agency|"
    r"Administration|Commission|University|Bureau|Office|Council|Board|Service|"
    r"Institute|Authority|Association|Corporation|Division|Center|Laboratory)|"
    r"(?:Department|Agency|Administration|Commission|University|Bureau|Office|"
    r"Council|Board|Service|Institute|Authority|Association|Corporation|Division|"
    r"Center|Laboratory)\s+of\s+(?:the\s+)?[A-Z])"
)
ACRONYM = re.compile(r"\([A-Z][A-Z0-9/&-]{2,12}\)|\b[A-Z]{3,8}\b")
HEADING = re.compile(r"^(?:FOR FURTHER INFORMATION CONTACT|ADDRESSES|SUPPLEMENTARY INFORMATION):\s*")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selected_cases():
    gold = lines(GOLD)
    forbidden = excluded_surface_pattern(gold)
    gold_texts = [row["input"] for row in gold]
    gold_addresses = set().union(*(address_keys(span["text"]) for row in gold
                                   for span in row["expected"] if span["kind"] == "address"))
    reviewed_files = (SILVER / "r23/train.jsonl", SILVER / "us-reviewed-candidate-v1.jsonl")
    for path in reviewed_files:
        if digest(path.read_bytes()) != REVIEWED_SOURCE_SHA256[path.name]:
            raise ValueError(f"reviewed source changed: {path.name}")
    reviewed_ids = {row["id"] for path in reviewed_files for row in lines(path)}
    if digest(EXCLUDED_TEXT.read_bytes()) != EXCLUDED_TEXT_SHA256:
        raise ValueError("frozen source text exclusions changed")
    seen_text = set(json.loads(EXCLUDED_TEXT.read_text()))
    groups = {"acronym": [], "contact": [], "narrative": []}
    for path in sorted(RAW.glob("*.json")):
        if path.stem in reviewed_ids:
            continue
        raw = json.loads(path.read_text())
        source = raw["text"].encode()
        if not raw.get("url") or any(text in raw["text"] for text in gold_texts):
            continue
        for number, (start, end) in enumerate(paragraphs(source), 1):
            piece = source[start:end]
            text = piece.decode()
            if (not 100 <= len(piece) <= 800 or not text.lstrip()[:1].isupper()
                    or not text.rstrip().endswith((".", "?", "!"))
                    or not ORG_CUE.search(text) or SOURCE_ARTIFACT.search(text)
                    or SOURCE_NOISE.search(text)
                    or forbidden.search(normalize_surface(text))
                    or address_keys(text) & gold_addresses
                    or digest(piece) in seen_text):
                continue
            if ACRONYM.search(HEADING.sub("", text)):
                group = "acronym"
            elif text.startswith(("FOR FURTHER INFORMATION CONTACT:", "ADDRESSES:")):
                group = "contact"
            else:
                group = "narrative"
            groups[group].append((digest(piece), path.stem, number, text, raw["url"]))
    selected = []
    used_sources = set()
    used_prefixes = set()
    for group in ("acronym", "contact", "narrative"):
        count = 0
        for _, source_id, number, text, url in sorted(groups[group]):
            prefix = tuple(re.findall(r"[a-z]+", text.casefold())[:18])
            if source_id in used_sources or prefix in used_prefixes:
                continue
            used_sources.add(source_id)
            used_prefixes.add(prefix)
            selected.append({"name": f"raw-us-org-{source_id}-{number}",
                             "country": "US", "input": text, "source_url": url,
                             "group": group})
            count += 1
            if count == GROUP_COUNTS[group]:
                break
        if count != GROUP_COUNTS[group]:
            raise ValueError(f"only {count} source-distinct {group} paragraphs")
    return selected


def main():
    if OUT.exists():
        data = OUT.read_bytes()
        selected = lines(OUT)
    else:
        selected = selected_cases()
        data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                          for row in selected) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError(f"raw blind packet differs: {digest(data)}")
    if len({row["name"] for row in selected}) != len(selected):
        raise ValueError("raw blind packet repeats a case")
    for row in selected:
        source_id = row["name"].removeprefix("raw-us-org-").rsplit("-", 1)[0]
        raw = json.loads((RAW / f"{source_id}.json").read_text())
        if row["source_url"] != raw["url"] or raw["text"].count(row["input"]) != 1:
            raise ValueError(f"raw blind case differs from source: {row['name']}")
    manifest = {"kind": "us_raw_org_blind_packet", "cases": len(selected),
                "groups": GROUP_COUNTS, "sha256": digest(data)}
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("raw blind packet manifest changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} raw blind US paragraphs frozen: {digest(data)}")


if __name__ == "__main__":
    main()
