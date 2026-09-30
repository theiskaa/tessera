"""Freeze blind US contact passages with source-backed organization hierarchies."""

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from active_sources import ACTIVE_SILVER
from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import ROOT, normalize_surface
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE, digest
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex, shingles


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import historical_sources, source_family


RAW = ROOT / "data/raw/silver/federal-register"
OUT = ROOT / "data/raw/candidates/federal-register-org-contacts-v1"
PACKET = OUT / "blind-v1.jsonl"
MANIFEST = OUT / "blind-v1.manifest.json"
STRICT = ROOT / "data/interim/review/us-2026-contact-strict-holdout-v2.jsonl"
ORG_TERM = re.compile(
    r"\b(?:Branch|Division|Office|Bureau|Agency|Department|Administration|Service|"
    r"Commission|Directorate|Center|District|University)\b", re.I
)
MAX_CASES = 24
MAX_PER_YEAR = 16


def inventory():
    """Pin the raw 2024–25 capture set used for this screen."""
    paths = sorted(path for path in RAW.glob("20*.json")
                   if path.stem.startswith(("2024-", "2025-")))
    state = hashlib.sha256()
    for path in paths:
        state.update(path.name.encode())
        state.update(b"\0")
        state.update(bytes.fromhex(digest(path.read_bytes())))
    return paths, state.hexdigest()


def selected_rows():
    """Select varied contact layouts without using entity labels for candidates."""
    frozen = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    reserved_paths = frozen["reserved_input_sha256"] if frozen else None
    reserved_ids, reserved_families, reserved_hashes = historical_sources(
        frozen_paths=reserved_paths)
    if frozen and reserved_hashes != frozen["reserved_input_sha256"]:
        raise ValueError("source-use inputs changed since org contact screening")
    raw_paths, inventory_sha = inventory()
    if frozen and (frozen["raw_inventory_sha256"] != inventory_sha or
                   frozen["raw_inventory_files"] != len(raw_paths)):
        raise ValueError("raw Federal Register inventory changed after screening")
    evaluation = [row for path in EVALUATION for row in lines(path)] + lines(STRICT)
    if len(evaluation) != 339:
        raise ValueError("US evaluation and strict holdout membership changed")
    eval_pattern = excluded_surface_pattern(evaluation)
    aliases = active_aliases(evaluation, include_roster=False)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True))) if aliases else None
    people = person_alias_patterns(evaluation)
    addresses = set().union(*(address_keys(span["text"])
                              for row in evaluation for span in row["expected"]
                              if span["kind"] == "address"))
    streets = {key[1:] for key in addresses}
    eval_emails = {match.casefold() for row in evaluation
                   for match in EMAIL.findall(row["input"])}
    eval_phones = {re.sub(r"\D", "", match) for row in evaluation
                   for match in PHONE.findall(row["input"])}
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_SILVER]
    near = NearTextIndex()
    for row in evaluation:
        near.add(row["name"], row["input"])
    for path in active_paths:
        for row in lines(path):
            near.add(row["id"], row["text"])
    counts = collections.Counter()
    candidates = []
    for path in raw_paths:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        source_id = path.stem
        if (raw.get("id") != source_id or raw.get("source") != "federal-register" or
                not raw.get("date", "").startswith(source_id[:4] + "-") or
                not raw.get("url", "").startswith(
                    "https://www.federalregister.gov/documents/") or
                not isinstance(raw.get("text"), str)):
            raise ValueError(f"invalid Federal Register source: {path}")
        source_group = source_family(raw["url"])
        if source_id in reserved_ids or source_group in reserved_families:
            counts["reserved_source"] += 1
            continue
        for number, text in enumerate(raw["text"].split("\n\n"), 1):
            if not text.startswith(("FOR FURTHER INFORMATION CONTACT:", "ADDRESSES:")):
                continue
            counts["contact_paragraphs"] += 1
            if (not 15 <= len(text.split()) <= 120 or
                    not text.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(text) or raw["text"].count(text) != 1):
                counts["boundary_or_artifact"] += 1
                continue
            terms = {match.group().casefold() for match in ORG_TERM.finditer(text)}
            if len(terms) < 2:
                counts["not_org_rich"] += 1
                continue
            normalized = normalize_surface(text)
            keys = address_keys_in_text(text)
            emails = {match.casefold() for match in EMAIL.findall(text)}
            phones = {re.sub(r"\D", "", match) for match in PHONE.findall(text)}
            if eval_pattern.search(normalized):
                counts["evaluation_surface"] += 1
                continue
            if alias_pattern and alias_pattern.search(normalized):
                counts["evaluation_alias"] += 1
                continue
            if has_person_alias(text, people):
                counts["evaluation_person"] += 1
                continue
            if keys & addresses or {key[1:] for key in keys} & streets:
                counts["evaluation_street"] += 1
                continue
            if emails & eval_emails or phones & eval_phones:
                counts["evaluation_contact"] += 1
                continue
            if near.prior(text):
                counts["prior_near_text"] += 1
                continue
            score = (3 * len(terms) +
                     3 * bool(terms & {"branch", "division"}) +
                     2 * bool(terms & {"office", "district"}) +
                     bool(emails) + bool(phones) + bool(keys) +
                     bool(text.startswith("FOR FURTHER INFORMATION CONTACT:")))
            candidates.append({
                "name": f"us-org-contact-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": raw["url"], "source_group": source_group,
                "raw_sha256": digest(raw_bytes), "input": text,
                "year": source_id[:4], "paragraph": number,
                "layout_score": score,
            })
    ranked = sorted(candidates, key=lambda row: (
        -row["layout_score"],
        digest(f"tessera-us-org-contact-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    years = collections.Counter()
    selected_shingles = []
    for row in ranked:
        if len(selected) >= MAX_CASES:
            break
        if row["source_group"] in groups or years[row["year"]] >= MAX_PER_YEAR:
            counts["diversity_cap"] += 1
            continue
        sample = shingles(row["input"])
        if any(len(sample & prior) / min(len(sample), len(prior)) >= 0.55
               for prior in selected_shingles if sample and prior):
            counts["packet_near_text"] += 1
            continue
        selected.append(row)
        selected_shingles.append(sample)
        groups.add(row["source_group"])
        years[row["year"]] += 1
    if (len(selected) < 16 or set(years) != {"2024", "2025"} or
            min(years.values()) < 4):
        raise ValueError(f"too few diverse organization contacts: {len(selected)}, {dict(years)}")
    selected.sort(key=lambda row: row["name"])
    paths = [*EVALUATION, STRICT, *active_paths]
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}
    return selected, counts, inventory_sha, len(raw_paths), reserved_hashes, inputs


def main():
    """Write a pinned unlabeled packet for two independent label reviews."""
    rows, counts, inventory_sha, raw_count, reserved, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_org_rich_contact_blind_v1",
        "cases": len(rows),
        "cases_by_year": dict(sorted(collections.Counter(
            row["year"] for row in rows).items())),
        "raw_inventory_files": raw_count,
        "raw_inventory_sha256": inventory_sha,
        "reserved_input_sha256": reserved,
        "evaluation_and_active_sha256": dict(sorted(inputs.items())),
        "selected_source_sha256": {row["source_id"]: row["raw_sha256"] for row in rows},
        "selection": dict(sorted(counts.items())),
        "source_group_key": "source_group",
        "label_status": "unlabeled",
        "training_eligible": False,
        "intended_use": "two_blind_reviews_then_split_audit",
        "sha256": digest(data),
    }
    if PACKET.exists() != MANIFEST.exists():
        raise ValueError("org contact packet or manifest is missing")
    if PACKET.exists() and PACKET.read_bytes() != data:
        raise ValueError("org contact packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("org contact screen inputs changed after freezing")
    OUT.mkdir(parents=True, exist_ok=True)
    if not PACKET.exists():
        PACKET.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-distinct organization contact candidates frozen: "
          f"{manifest['cases_by_year']}")


if __name__ == "__main__":
    main()
