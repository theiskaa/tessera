"""Freeze 2025 contact layouts that challenge US entity boundaries."""

import collections
import json
import re

from active_sources import BASE_SILVER
from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import family, normalize_surface
from freeze_2025_contact_packet import (EMAIL, GOLD, HOLDOUT, HOLDOUT_REVIEWS,
                                        PHONE, RAW, SILVER, digest, selected_cases)
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_2025_year_contact_packet import PACKET_DIR, SNAPSHOT, source_snapshot
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases


EXPANSION = PACKET_DIR / "us-2025-year-contact-expansion-blind-v2.jsonl"
OUT = PACKET_DIR / "us-2025-targeted-contact-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MAX_CASES = 20
MAX_TEMPLATE_CASES = 12
MAX_TEMPLATE_PER_MONTH = 6
SOURCE_EXCLUSIONS = {
    "2025-07423": "source prints an extra-digit phone number",
    "2025-07435": "source fuses an organization and street number",
    "2025-23034": "source prints an ambiguous No before the street number",
    "2025-23343": "source address lacks a state abbreviation",
}
ROLE = re.compile(
    r"\b(?:chief curator|deputy director|regional manager|project manager|"
    r"community planner|committee management officer|assistant counsel|"
    r"financial analyst|program analyst|coordinator|associate director|"
    r"chief accountant|director|curator|manager|officer|analyst)\b", re.I
)
UNIT = re.compile(r"\b(?:office|department|division|bureau|program|"
                  r"laborator(?:y|ies)|center|agency|commission|museum|"
                  r"universit(?:y|ies)|college|academy|institute)\b", re.I)
ROUTED = re.compile(r"\b(?:should contact|directed to|direct requests)\b|"
                    r"\b(?:attention|attn\.?)\s*:", re.I)
INSTITUTION = re.compile(r"\b(?:academy|museum|universit(?:y|ies)|college)\b", re.I)


def surfaces(rows):
    emails = {match.casefold() for row in rows for match in EMAIL.findall(row["input"])}
    phones = {re.sub(r"\D", "", match) for row in rows
              for match in PHONE.findall(row["input"])}
    return emails, phones


def layout_tags(text):
    tags = set()
    if ROUTED.search(text):
        tags.add("routed_contact")
    if INSTITUTION.search(text):
        tags.add("institution")
    if len(UNIT.findall(text)) >= 2:
        tags.add("nested_org_units")
    if ROLE.search(text):
        tags.add("role_near_entity")
    if PHONE.search(text) and address_keys_in_text(text):
        tags.add("phone_near_address")
    if re.search(r"\b(?:obtained from|contacting|by emailing)\b", text, re.I):
        tags.add("alternate_contact_verb")
    return tags


def is_template(row):
    slug = row["source_url"].rsplit("/", 1)[-1]
    return slug.startswith(("notice-of-inventory-completion",
                            "notice-of-intended-repatriation"))


def selected_rows():
    snapshot = source_snapshot(persist=False)
    evaluation = [row for path in EVALUATION for row in lines(path)]
    if len(evaluation) != 311 or len({row["name"] for row in evaluation}) != 311:
        raise ValueError("the frozen evaluation case set changed")
    expansion_manifest = json.loads(EXPANSION.with_suffix(".manifest.json").read_text())
    expansion = lines(EXPANSION)
    if (expansion_manifest["sha256"] != digest(EXPANSION.read_bytes()) or
            expansion_manifest["training_eligible"] or
            len(expansion) != expansion_manifest["cases"]):
        raise ValueError("reviewed contact expansion packet changed")
    active_paths = [SILVER / f"{name}.jsonl" for name, _ in BASE_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    excluded = active + expansion
    excluded_sources = {row["source_id"] for row in excluded if row.get("source_id")}
    excluded_families = {row.get("source_group") or family(row["source_url"])
                         for row in excluded if row.get("source_url")}
    candidates, baseline_counts, _ = selected_cases(
        snapshot,
        extra_gold=[row for path in EVALUATION[1:] for row in lines(path)],
        excluded_sources=excluded_sources,
        excluded_families=excluded_families,
        extra_train=([{"name": row["id"], "input": row["text"]} for row in active] +
                     expansion),
        month_cap=None,
        address_screen=address_keys_in_text,
        address_pair_screen=True,
    )
    active_gold = [
        {"name": row["id"], "input": row["text"],
         "expected": [{"kind": span["kind"],
                       "text": row["text"].encode()[span["start"]:span["end"]].decode()}
                      for span in row["entities"]]}
        for row in active
    ]
    forbidden = excluded_surface_pattern(active_gold)
    aliases = active_aliases(active_gold)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True))) if aliases else None)
    people = person_alias_patterns(active_gold)
    active_addresses = set().union(*(address_keys(span["text"])
                                     for row in active_gold for span in row["expected"]
                                     if span["kind"] == "address"))
    active_streets = {key[1:] for key in active_addresses}
    prior_emails, prior_phones = surfaces(evaluation + expansion + active_gold)
    screened = []
    counts = collections.Counter()
    for row in candidates:
        text = row["input"]
        normalized = normalize_surface(text)
        keys = address_keys_in_text(text)
        email, phone = surfaces([row])
        reason = None
        if row["source_id"] in SOURCE_EXCLUSIONS:
            reason = "source_quality"
        elif forbidden.search(normalized):
            reason = "active_label_surface"
        elif alias_pattern and alias_pattern.search(normalized):
            reason = "active_org_alias"
        elif has_person_alias(text, people):
            reason = "active_person_alias"
        elif keys & active_addresses or {key[1:] for key in keys} & active_streets:
            reason = "active_numbered_street"
        elif email & prior_emails or phone & prior_phones:
            reason = "reserved_contact_surface"
        elif not layout_tags(text):
            reason = "outside_target_layouts"
        if reason:
            counts[reason] += 1
        else:
            screened.append(row)
    weights = {"routed_contact": 5, "institution": 3,
               "nested_org_units": 4, "role_near_entity": 3,
               "phone_near_address": 2, "alternate_contact_verb": 3}
    ranked = sorted(screened, key=lambda row: (
        -sum(weights[tag] for tag in layout_tags(row["input"])), row["source_id"]
    ))
    selected = []
    template_months = collections.Counter()
    templates = 0
    for row in ranked:
        template = is_template(row)
        month = json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
        if template and (templates >= MAX_TEMPLATE_CASES or
                         template_months[month] >= MAX_TEMPLATE_PER_MONTH):
            counts["template_cap"] += 1
            continue
        if len(selected) >= MAX_CASES:
            counts["packet_cap"] += 1
            continue
        selected.append(row)
        if template:
            templates += 1
            template_months[month] += 1
    selected.sort(key=lambda row: row["name"])
    if (not selected or len({row["source_id"] for row in selected}) != len(selected) or
            len({family(row["source_url"]) for row in selected}) != len(selected) or
            source_snapshot(persist=False) != snapshot):
        raise ValueError("targeted contact source selection changed")
    for row in selected:
        source = (RAW / f"{row['source_id']}.json").read_bytes()
        raw = json.loads(source)
        if (digest(source) != row["raw_sha256"] or
                raw["id"] != row["source_id"] or
                raw["url"] != row["source_url"] or
                raw["text"].count(row["input"]) != 1):
            raise ValueError(f"targeted contact differs from source: {row['name']}")
    inputs = (*EVALUATION, *GOLD, HOLDOUT, *HOLDOUT_REVIEWS,
              EXPANSION, SNAPSHOT, *active_paths)
    hashes = {str(path.relative_to(PACKET_DIR.parents[2])): digest(path.read_bytes())
              for path in inputs}
    return selected, counts, baseline_counts, hashes, len(snapshot)


def main():
    rows, counts, baseline, hashes, capture_count = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    tags = collections.Counter(tag for row in rows for tag in layout_tags(row["input"]))
    months = collections.Counter(json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                                 for row in rows)
    manifest = {
        "kind": "us_2025_targeted_contact_blind_v1",
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_capture_count": capture_count,
        "evaluation_cases": 311,
        "input_sha256": dict(sorted(hashes.items())),
        "source_exclusions": SOURCE_EXCLUSIONS,
        "selection": dict(sorted(counts.items())),
        "baseline_selection": dict(sorted(baseline.items())),
        "layout_cases": dict(sorted(tags.items())),
        "cases_by_month": dict(sorted(months.items())),
        "template_cases": sum(is_template(row) for row in rows),
        "cases": len(rows),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("targeted contact packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen targeted contact packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen targeted contact inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} targeted US contact cases frozen for blind review")


if __name__ == "__main__":
    main()
