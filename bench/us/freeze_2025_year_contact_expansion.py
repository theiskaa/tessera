"""Freeze additional source-disjoint 2025 contacts for independent review."""

import collections
import json
import re

from active_sources import BASE_SILVER
from address_keys import address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import family, normalize_surface
from freeze_2025_contact_packet import (EMAIL, GOLD, HOLDOUT, HOLDOUT_REVIEWS,
                                        PHONE,
                                        RAW, SILVER, SILVER_SOURCES, digest,
                                        selected_cases)
from freeze_2025_year_contact_packet import (EXTRA_TRAIN, MANIFEST as YEAR_MANIFEST,
                                             OUT as YEAR_PACKET, PACKET_DIR,
                                             PRIOR_2025_PACKETS, REVIEWED_2025,
                                             SNAPSHOT, source_snapshot)


YEAR_GOLD = PACKET_DIR.parent.parent / "review/us-2025-year-contact-gold-v1.jsonl"
YEAR_HOLDOUT = PACKET_DIR.parent.parent / "review/us-2025-year-contact-holdout-v1.jsonl"
ACRONYM_PACKET = PACKET_DIR / "us-2025-year-acronym-prose-blind-v4.jsonl"
ACRONYM_GOLD = PACKET_DIR.parent.parent / "review/us-2025-year-acronym-prose-gold-v2.jsonl"
PREVIOUS_OUT = PACKET_DIR / "us-2025-year-contact-expansion-blind-v1.jsonl"
OUT = PACKET_DIR / "us-2025-year-contact-expansion-blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MAX_PER_MONTH = 40
TEMPLATE_CAP_PER_MONTH = 5
PRIOR_CONTACT_HOLDOUT = PACKET_DIR.parent.parent / "review/us-2025-contact-holdout-v2.jsonl"
CURRENT_YEAR_HOLDOUT = PACKET_DIR.parent.parent / "review/us-2025-year-contact-holdout-v2.jsonl"
MALFORMED_SOURCES = {
    "2025-07435": "organization and street number fused in source",
    "2025-23034": "source prints an ambiguous No before street number",
    "2025-23152": "source address omits a city",
}


def selected_rows():
    snapshot = source_snapshot(persist=False)
    year_manifest = json.loads(YEAR_MANIFEST.read_text())
    if (year_manifest["sha256"] != digest(YEAR_PACKET.read_bytes()) or
            year_manifest["training_eligible"]):
        raise ValueError("original full-year contact packet changed")
    holdout_manifest = json.loads(YEAR_HOLDOUT.with_suffix(".manifest.json").read_text())
    if (holdout_manifest["sha256"] != digest(YEAR_HOLDOUT.read_bytes()) or
            not holdout_manifest["strict_entity_holdout"]):
        raise ValueError("strict contact holdout changed")
    excluded_paths = (YEAR_PACKET, *PRIOR_2025_PACKETS,
                      REVIEWED_2025, ACRONYM_PACKET)
    excluded = [row for path in excluded_paths for row in lines(path)]
    extra_train_paths = (YEAR_GOLD, ACRONYM_GOLD, REVIEWED_2025,
                         *PRIOR_2025_PACKETS, *EXTRA_TRAIN)
    selected, counts, _ = selected_cases(
        snapshot, extra_gold=lines(YEAR_HOLDOUT),
        excluded_sources={row["source_id"] for row in excluded},
        excluded_families={family(row["source_url"]) for row in excluded},
        extra_train=(row for path in extra_train_paths for row in lines(path)),
        month_cap=MAX_PER_MONTH, address_screen=address_keys_in_text,
        address_pair_screen=True)
    months = collections.Counter(
        json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
        for row in selected)
    if (any(value > MAX_PER_MONTH for value in months.values()) or
            len({row["source_id"] for row in selected}) != len(selected) or
            source_snapshot(persist=False) != snapshot):
        raise ValueError("contact expansion source or month selection changed")
    inputs = (*excluded_paths, *extra_train_paths, YEAR_HOLDOUT)
    input_hashes = {path.name: digest(path.read_bytes()) for path in inputs}
    return selected, counts, months, input_hashes, len(snapshot)


def successor_rows():
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous = lines(PREVIOUS_OUT)
    if (previous_manifest["sha256"] != digest(PREVIOUS_OUT.read_bytes()) or
            previous_manifest["training_eligible"] or
            len(previous) != previous_manifest["cases"]):
        raise ValueError("previous contact expansion changed")
    holdout = lines(PRIOR_CONTACT_HOLDOUT)
    forbidden = excluded_surface_pattern(holdout)
    active = [row for name, _ in BASE_SILVER
              if name != "us-2025-year-contact-expansion-v1"
              for row in lines(SILVER / f"{name}.jsonl")]
    active_families = {family(row["source_url"]) for row in active if row.get("source_url")}
    active_emails = {match.casefold() for row in active for match in EMAIL.findall(row["text"])}
    active_phones = {re.sub(r"\D", "", match) for row in active
                     for match in PHONE.findall(row["text"])}
    exclusions = {}
    templates = collections.defaultdict(list)
    selected = []
    for row in previous:
        name, source_id, text = row["name"], row["source_id"], row["input"]
        raw_bytes = (RAW / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != row["raw_sha256"] or
                raw["id"] != source_id or raw["url"] != row["source_url"] or
                raw["text"].count(text) != 1):
            raise ValueError(f"contact expansion differs from source: {name}")
        month = raw["date"][:7]
        email = {match.casefold() for match in EMAIL.findall(text)}
        phone = {re.sub(r"\D", "", match) for match in PHONE.findall(text)}
        reason = MALFORMED_SOURCES.get(source_id)
        if reason is None and forbidden.search(normalize_surface(text)):
            reason = "strict contact holdout contains a labeled surface"
        if reason is None and family(row["source_url"]) in active_families:
            reason = "source family already in active silver"
        if reason is None and (email & active_emails or phone & active_phones):
            reason = "contact surface already in active silver"
        if reason:
            exclusions[name] = reason
        elif ("notice-of-inventory-completion" in row["source_url"] or
              "notice-of-intended-repatriation" in row["source_url"]):
            templates[month].append(row)
        else:
            selected.append(row)
    for month, rows in sorted(templates.items()):
        ranked = sorted(rows, key=lambda row: digest(row["name"].encode()))
        selected.extend(ranked[:TEMPLATE_CAP_PER_MONTH])
        exclusions.update({row["name"]: "repatriation template month cap"
                           for row in ranked[TEMPLATE_CAP_PER_MONTH:]})
    selected.sort(key=lambda row: row["name"])
    if not selected or len(selected) + len(exclusions) != len(previous):
        raise ValueError("contact expansion split is incomplete")
    months = collections.Counter(json.loads((RAW / f"{row['source_id']}.json").read_text())["date"][:7]
                                 for row in selected)
    active_hashes = {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                     for name, _ in BASE_SILVER
                     if name != "us-2025-year-contact-expansion-v1"}
    return selected, exclusions, months, active_hashes


def main():
    rows, excluded, months, active_hashes = successor_rows()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in rows) + "\n").encode()
    manifest = {
        "kind": "us_2025_year_contact_expansion_blind_v2",
        "supersedes_sha256": digest(PREVIOUS_OUT.read_bytes()),
        "intended_use": "two_blind_reviews_then_split_audit",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "current_silver_sha256": active_hashes,
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in (PRIOR_CONTACT_HOLDOUT, CURRENT_YEAR_HOLDOUT)},
        "excluded": dict(sorted(excluded.items())),
        "template_cap_per_month": TEMPLATE_CAP_PER_MONTH,
        "review_cases_by_month": dict(sorted(months.items())),
        "cases": len(rows),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen contact expansion or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen contact expansion changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen contact expansion inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} additional 2025 contact cases frozen for blind review")


if __name__ == "__main__":
    main()
