"""Freeze diverse blind contacts from the isolated 2026 source capture."""

import collections
import json
import re
import sys

from active_sources import ACTIVE_SILVER, BASE_SILVER
from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import ROOT, family, normalize_surface
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE, SILVER, digest
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_2025_targeted_contact_packet import layout_tags
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import OUT as CAPTURE
from fetch_federal_register_2026_candidates import (historical_sources,
                                                     source_family,
                                                     verify_manifest)


SOURCE_MANIFEST = CAPTURE / "manifest.json"
OUT = CAPTURE / "screened-contact-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MAX_CASES = 40
MAX_PER_MONTH = 6
MAX_TEMPLATES = 10
MAX_TEMPLATES_PER_MONTH = 2
MIN_CASES = 12
MIN_MONTHS = 4
MIN_NON_TEMPLATES = 8
NAME_WITH_PHONE = re.compile(
    r"^FOR FURTHER INFORMATION CONTACT:\s*[A-Z][A-Za-z'’.-]+\s+"
    r"(?:[A-Z]\.\s+)?[A-Z][A-Za-z'’.-]+\b", re.I
)


def tags(text):
    """Describe a contact layout without assigning entity labels."""
    found = layout_tags(text)
    if NAME_WITH_PHONE.search(text) and PHONE.search(text):
        found.add("name_with_phone")
    return found


def previous_texts(frozen_paths):
    """Read prior review text for exact contact and near-text exclusion."""
    paths = [ROOT / relative for relative in frozen_paths]
    texts = []
    for path in paths:
        if path.parent == ROOT / "data/raw/review/federal-register":
            row = json.loads(path.read_text())
            texts.append((row["id"], row["text"]))
            continue
        for row in lines(path):
            if not isinstance(row, dict) or row.get("country", "US") != "US":
                continue
            text = row.get("input") or row.get("text")
            if isinstance(text, str):
                texts.append((row.get("name") or row.get("id") or str(path), text))
    return texts


def selected_rows():
    """Apply frozen split, contact-surface, and text-diversity screens."""
    capture = verify_manifest()
    if (capture["training_eligible"] or len(capture["sources"]) != 2000 or
            capture["months"] != [f"2026-{month:02d}" for month in range(1, 9)]):
        raise ValueError("2026 candidate source capture is incomplete")
    reserved_ids, reserved_families, reserved_hashes = historical_sources(
        frozen_paths=capture["reserved_input_sha256"])
    if capture["reserved_input_sha256"] != reserved_hashes:
        raise ValueError("source-use inputs changed since 2026 capture")
    evaluation = [row for path in EVALUATION for row in lines(path)]
    if len(evaluation) != 311:
        raise ValueError("frozen evaluation case set changed")
    active_paths = [SILVER / f"{name}.jsonl" for name, _ in ACTIVE_SILVER]
    base_paths = [SILVER / f"{name}.jsonl" for name, _ in BASE_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    active_gold = [
        {"name": row["id"], "input": row["text"],
         "expected": [{"kind": span["kind"],
                       "text": row["text"].encode()[span["start"]:span["end"]].decode()}
                      for span in row["entities"]]}
        for row in active
    ]
    gold = evaluation + active_gold
    forbidden = excluded_surface_pattern(gold)
    aliases = active_aliases(gold, include_roster=True)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True))) if aliases else None)
    people = person_alias_patterns(gold)
    addresses = set().union(*(address_keys(span["text"])
                              for row in gold for span in row["expected"]
                              if span["kind"] == "address"))
    street_pairs = {key[1:] for key in addresses}
    prior = previous_texts(capture["reserved_input_sha256"])
    prior.extend((row["id"], row["text"]) for row in active)
    prior.extend((row["id"], row["text"])
                 for path in base_paths for row in lines(path))
    near = NearTextIndex()
    for name, text in prior:
        near.add(name, text)
    prior_emails = {match.casefold() for _, text in prior for match in EMAIL.findall(text)}
    prior_phones = {re.sub(r"\D", "", match) for _, text in prior
                    for match in PHONE.findall(text)}
    counts = collections.Counter()
    candidates = []
    for source in capture["sources"]:
        source_id = source["source_id"]
        path = CAPTURE / "sources" / f"{source_id}.json"
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source_id or raw["url"] != source["source_url"] or
                not raw["date"].startswith(source["month"]) or
                source_id in reserved_ids or
                source_family(raw["url"]) in reserved_families):
            raise ValueError(f"2026 candidate source changed or reused: {source_id}")
        for number, text in enumerate(raw["text"].split("\n\n"), 1):
            if not text.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
                continue
            counts["contact_paragraphs"] += 1
            if not 15 <= len(text.split()) <= 120 or not text.rstrip().endswith((".", "!", "?")):
                counts["length_or_boundary"] += 1
                continue
            if ARTIFACT.search(text) or raw["text"].count(text) != 1:
                counts["source_artifact_or_repeat"] += 1
                continue
            layout = tags(text)
            if not layout:
                counts["outside_target_layouts"] += 1
                continue
            normalized = normalize_surface(text)
            keys = address_keys_in_text(text)
            emails = {match.casefold() for match in EMAIL.findall(text)}
            phones = {re.sub(r"\D", "", match) for match in PHONE.findall(text)}
            if forbidden.search(normalized):
                counts["evaluation_or_training_surface"] += 1
                continue
            if alias_pattern and alias_pattern.search(normalized):
                counts["evaluation_or_training_alias"] += 1
                continue
            if has_person_alias(text, people):
                counts["evaluation_or_training_person"] += 1
                continue
            if keys & addresses or {key[1:] for key in keys} & street_pairs:
                counts["evaluation_or_training_street"] += 1
                continue
            if emails & prior_emails or phones & prior_phones:
                counts["prior_contact_surface"] += 1
                continue
            if near.prior(text):
                counts["prior_near_text"] += 1
                continue
            candidates.append({
                "name": f"us-2026-contact-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": raw["url"], "raw_sha256": source["raw_sha256"],
                "input": text, "source_group": source["source_group"],
                "month": source["month"], "agency": source["agency"],
                "template": source["template"],
            })
    weights = {"routed_contact": 5, "institution": 3,
               "nested_org_units": 4, "role_near_entity": 3,
               "phone_near_address": 3, "alternate_contact_verb": 2,
               "name_with_phone": 2}
    ranked = sorted(candidates, key=lambda row: (
        -sum(weights[tag] for tag in tags(row["input"])), row["name"]
    ))
    selected = []
    used_ids = set()
    used_families = set()
    month_counts = collections.Counter()
    template_months = collections.Counter()
    for row in sorted(ranked, key=lambda row: (row["template"],
                                                -sum(weights[tag] for tag in tags(row["input"])),
                                                row["name"])):
        if len(selected) >= MAX_CASES:
            break
        if row["source_id"] in used_ids or row["source_group"] in used_families:
            counts["packet_source_repeat"] += 1
            continue
        if near.prior(row["input"]):
            counts["packet_near_text"] += 1
            continue
        if month_counts[row["month"]] >= MAX_PER_MONTH:
            counts["packet_month_cap"] += 1
            continue
        if row["template"] and (sum(item["template"] for item in selected) >= MAX_TEMPLATES or
                                template_months[row["month"]] >= MAX_TEMPLATES_PER_MONTH or
                                sum(item["template"] for item in selected) * 3 >=
                                sum(not item["template"] for item in selected)):
            counts["packet_template_cap"] += 1
            continue
        selected.append(row)
        used_ids.add(row["source_id"])
        used_families.add(row["source_group"])
        month_counts[row["month"]] += 1
        template_months[row["month"]] += row["template"]
        near.add(row["name"], row["input"])
    selected.sort(key=lambda row: row["name"])
    if (len(selected) < MIN_CASES or len(month_counts) < MIN_MONTHS or
            sum(not row["template"] for row in selected) < MIN_NON_TEMPLATES):
        raise ValueError(f"2026 clean contact pool is too small: {len(selected)} selected, "
                         f"{len(month_counts)} months, {dict(counts)}")
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (*EVALUATION, *active_paths, *base_paths)}
    return selected, counts, hashes


def main():
    rows, counts, hashes = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_2026_contact_blind_candidates_v1",
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_capture_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "evaluation_and_training_sha256": dict(sorted(hashes.items())),
        "selection": dict(sorted(counts.items())),
        "layout_cases": dict(sorted(collections.Counter(
            tag for row in rows for tag in tags(row["input"])).items())),
        "cases_by_month": dict(sorted(collections.Counter(row["month"] for row in rows).items())),
        "template_cases": sum(row["template"] for row in rows),
        "cases": len(rows),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("2026 blind packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2026 blind packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2026 blind packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-disjoint 2026 contact cases frozen for blind review")


if __name__ == "__main__":
    main()
