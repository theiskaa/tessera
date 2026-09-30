"""Freeze unused US contacts that resemble measured detection failures."""

import collections
import json
import re
import sys
import hashlib

from active_sources import ACTIVE_SILVER
from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import ROOT, normalize_surface
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE, digest
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from screen_2026_contact_candidates import previous_texts
from screen_org_rich_contact_candidates import ORG_TERM, STRICT
from silver_dedupe import NearTextIndex, shingles


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import OUT as CAPTURE
from fetch_federal_register_2026_candidates import (historical_sources,
                                                     source_family,
                                                     verify_manifest)


RAW = ROOT / "data/raw/silver/federal-register"
ORG_PACKET = ROOT / "data/raw/candidates/federal-register-org-contacts-v1/blind-v1.jsonl"
ORG_CANDIDATES = ROOT / "data/interim/silver/r25/us-org-rich-contacts-candidate-v1.jsonl"
OUT = ROOT / "data/raw/candidates/federal-register-us-error-target-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ROLE = re.compile(r"\b(?:Specialist|Director|Officer|Coordinator|Manager|Secretary|Analyst)\b", re.I)
VENUE = re.compile(r"\b(?:meeting|hearing|conference)\b.{0,130}\b(?:Hotel|Convention Center|Conference Center|Meeting Center|Trade Center|Auditorium)\b", re.I | re.S)
LONG_ADDRESS = re.compile(r"\b(?:Suite|Building|Joint Base|Room)\b|\b\d{5}-\d{4}\b", re.I)
MALFORMED_MARKUP = re.compile(r'">|<[^>]+>')
MAX_CASES = 40
MAX_PER_MONTH = 6
MAX_PER_CATEGORY = 14
CATEGORIES = ("org_chain", "role_near_name", "venue", "address_boundary")


def inventory(capture):
    """Pin the 2024–26 captured notice files used by the screen."""
    paths = sorted(path for path in RAW.glob("20*.json")
                   if path.stem.startswith(("2024-", "2025-")))
    paths.extend(CAPTURE / "sources" / f"{source['source_id']}.json"
                 for source in capture["sources"])
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("captured US notice IDs repeat across inventories")
    state = hashlib.sha256()
    for path in paths:
        state.update(path.name.encode())
        state.update(b"\0")
        state.update(bytes.fromhex(digest(path.read_bytes())))
    return paths, state.hexdigest()


def categories(text):
    """Group contacts by observable layout, before assigning any entity label."""
    found = set()
    terms = {match.group().casefold() for match in ORG_TERM.finditer(text)}
    if len(terms) >= 2 and terms & {"branch", "division", "office", "center"}:
        found.add("org_chain")
    if ROLE.search(text) and (EMAIL.search(text) or PHONE.search(text)):
        found.add("role_near_name")
    if VENUE.search(text):
        found.add("venue")
    if LONG_ADDRESS.search(text) and address_keys_in_text(text):
        found.add("address_boundary")
    return found


def selected_rows():
    """Screen source-distinct passages against held-out and prepared US inputs."""
    capture = verify_manifest()
    if (capture["training_eligible"] or len(capture["sources"]) != 2000 or
            capture["months"] != [f"2026-{month:02d}" for month in range(1, 9)]):
        raise ValueError("2026 source capture is incomplete")
    frozen = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    paths, inventory_sha = inventory(capture)
    if len(paths) < 9000:
        raise ValueError("US captured notice inventory is incomplete")
    if frozen and (frozen["raw_inventory_sha256"] != inventory_sha or
                   frozen["raw_inventory_files"] != len(paths)):
        raise ValueError("US notice inventory changed after screening")
    if frozen:
        reserved_paths = frozen["reserved_input_sha256"]
    else:
        _, _, prior_hashes = historical_sources()
        reserved_paths = [*prior_hashes,
                          str(ORG_PACKET.relative_to(ROOT)),
                          str(ORG_CANDIDATES.relative_to(ROOT))]
    reserved_ids, reserved_families, reserved_hashes = historical_sources(
        frozen_paths=reserved_paths)
    if frozen and frozen["reserved_input_sha256"] != reserved_hashes:
        raise ValueError("2026 source-use inputs changed after screening")
    evaluation = [row for path in EVALUATION for row in lines(path)] + lines(STRICT)
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    active_gold = [{
        "name": row["id"], "input": row["text"],
        "expected": [{"kind": span["kind"],
                      "text": row["text"].encode()[span["start"]:span["end"]].decode()}
                     for span in row["entities"]],
    } for row in active]
    gold = evaluation + active_gold
    surface_pattern = excluded_surface_pattern(gold)
    aliases = active_aliases(gold, include_roster=True)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(aliases, key=len, reverse=True)), re.I) if aliases else None
    people = person_alias_patterns(gold)
    addresses = set().union(*(address_keys(span["text"])
                              for row in gold for span in row["expected"]
                              if span["kind"] == "address"))
    streets = {key[1:] for key in addresses}
    prior = previous_texts(reserved_hashes)
    prior.extend((row["id"], row["text"]) for row in active)
    near = NearTextIndex()
    for name, text in prior:
        near.add(name, text)
    prior_emails = {match.casefold() for _, text in prior for match in EMAIL.findall(text)}
    prior_phones = {re.sub(r"\D", "", match) for _, text in prior
                    for match in PHONE.findall(text)}
    counts = collections.Counter()
    candidates = []
    capture_sources = {source["source_id"]: source for source in capture["sources"]}
    for path in paths:
        source_id = path.stem
        source = capture_sources.get(source_id)
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        source_group = source_family(raw["url"])
        month = raw["date"][:7]
        if (raw.get("id") != source_id or
                raw.get("source") != "federal-register" or
                (not source and not raw.get("date", "").startswith(
                    source_id[:4] + "-")) or
                not raw.get("url", "").startswith(
                    "https://www.federalregister.gov/documents/") or
                not isinstance(raw.get("text"), str) or
                (source and (digest(raw_bytes) != source["raw_sha256"] or
                             raw["url"] != source["source_url"] or
                             month != source["month"] or
                             source_group != source["source_group"]))):
            raise ValueError(f"US notice source changed: {source_id}")
        if source_id in reserved_ids or source_group in reserved_families:
            counts["reserved_source"] += 1
            continue
        for number, text in enumerate(raw["text"].split("\n\n"), 1):
            if not text.startswith(("FOR FURTHER INFORMATION CONTACT:", "ADDRESSES:")):
                continue
            counts["contact_paragraph"] += 1
            if (not 15 <= len(text.split()) <= 120 or
                    not text.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(text) or MALFORMED_MARKUP.search(text) or
                    raw["text"].count(text) != 1):
                counts["boundary_or_artifact"] += 1
                continue
            found = categories(text)
            if not found:
                counts["outside_target_patterns"] += 1
                continue
            normalized = normalize_surface(text)
            keys = address_keys_in_text(text)
            emails = {match.casefold() for match in EMAIL.findall(text)}
            phones = {re.sub(r"\D", "", match) for match in PHONE.findall(text)}
            if surface_pattern.search(normalized):
                counts["labeled_surface"] += 1
                continue
            if alias_pattern and alias_pattern.search(normalized):
                counts["body_alias"] += 1
                continue
            if has_person_alias(text, people):
                counts["person_alias"] += 1
                continue
            if keys & addresses or {key[1:] for key in keys} & streets:
                counts["street"] += 1
                continue
            if emails & prior_emails or phones & prior_phones:
                counts["contact_surface"] += 1
                continue
            if near.prior(text):
                counts["prior_near_text"] += 1
                continue
            candidates.append({
                "name": f"us-error-target-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": raw["url"],
                "source_group": source_group,
                "raw_sha256": digest(raw_bytes), "input": text,
                "month": month,
                "paragraph": number, "categories": sorted(found),
            })
    selected = []
    groups = set()
    months = collections.Counter()
    category_counts = collections.Counter()
    selected_shingles = []
    ranked = sorted(candidates, key=lambda row: (
        len(row["categories"]) != 1,
        digest(f"tessera-us-error-target-v1:{row['name']}".encode()),
    ))
    for category in CATEGORIES:
        for row in ranked:
            if len(selected) >= MAX_CASES or category_counts[category] >= MAX_PER_CATEGORY:
                break
            if category not in row["categories"] or row["source_group"] in groups:
                continue
            if months[row["month"]] >= MAX_PER_MONTH:
                continue
            sample = shingles(row["input"])
            if any(len(sample & prior) / min(len(sample), len(prior)) >= 0.55
                   for prior in selected_shingles if sample and prior):
                continue
            selected.append(row)
            groups.add(row["source_group"])
            months[row["month"]] += 1
            category_counts[category] += 1
            selected_shingles.append(sample)
    if (len(selected) < 16 or len(months) < 5 or
            category_counts["org_chain"] < 1 or
            category_counts["role_near_name"] < 4 or
            category_counts["address_boundary"] < 4):
        raise ValueError(f"too few diverse error-target contacts: "
                         f"{len(selected)}, {dict(category_counts)}, {dict(months)}, "
                         f"screen counts {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [*EVALUATION, STRICT, *active_paths]}
    return selected, counts, reserved_hashes, inputs, category_counts, inventory_sha, len(paths)


def main():
    """Pin unlabeled passages for two independent reviews."""
    rows, counts, reserved, inputs, category_counts, inventory_sha, path_count = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_error_target_blind_v1",
        "cases": len(rows),
        "raw_inventory_files": path_count,
        "raw_inventory_sha256": inventory_sha,
        "cases_by_month": dict(sorted(collections.Counter(
            row["month"] for row in rows).items())),
        "cases_by_primary_category": dict(sorted(category_counts.items())),
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "reserved_input_sha256": reserved,
        "evaluation_and_active_sha256": dict(sorted(inputs.items())),
        "selection": dict(sorted(counts.items())),
        "source_group_key": "source_group",
        "training_eligible": False,
        "label_status": "unlabeled",
        "intended_use": "two_independent_reviews_then_split_audit",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("error-target packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("error-target packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("error-target screening inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} US error-target contacts frozen: {dict(category_counts)}")


if __name__ == "__main__":
    main()
