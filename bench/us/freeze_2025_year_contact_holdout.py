"""Freeze a source and entity disjoint diagnostic holdout from 2025 contacts."""

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from address_keys import address_keys_in_text
from build_full_notice_gold import read_rows
from build_silver import family
from build_2025_year_contact_gold import gold_rows
from check_holdout_overlap import SILVER, SILVER_SOURCES, SYNTHETIC, training_sources
from check_ready import person_key
from org_aliases import FAMILIES


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
GOLD = REVIEW / "us-2025-year-contact-gold-v1.jsonl"
READY_MANIFEST = REVIEW / "us-2025-ready-contact-gold-v1.manifest.json"
EVALUATION = (
    REVIEW / "us-eval-exclusions-v1.jsonl",
    REVIEW / "us-2025-contact-holdout-v2.jsonl",
)
REVIEWED_TRAIN = (
    REVIEW / "us-2024-contact-gold-v3.jsonl",
    REVIEW / "us-2024-acronym-prose-gold-v2.jsonl",
)
PREVIOUS_OUT = REVIEW / "us-2025-year-contact-holdout-v1.jsonl"
OUT = REVIEW / "us-2025-year-contact-holdout-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
TARGET_CASES = 36
MIN_MONTH_CASES = 2
MAX_MONTH_CASES = 4
MIN_FIELD_LABELS = {"person": 25, "org": 20, "address": 12, "email": 15, "phone": 25}
ALIASES = FAMILIES + (
    ("Social Security Administration", "SSA"),
    ("Securities and Exchange Commission", "SEC"),
    ("State, Private and Tribal Forestry", "State, Private, and Tribal Forestry"),
    ("United States Department of Justice", "U.S. Department of Justice", "Department of Justice"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", value)
    if words[:2] == ["u", "s"] or words[:2] == ["united", "states"]:
        words = words[2:]
    return " ".join(words)


ALIAS_KEYS = {canonical(name): canonical(group[0]) for group in ALIASES for name in group}


def fingerprints(text, spans):
    source = text.encode()
    keys = set()
    for span in spans:
        value = source[span["start"]:span["end"]].decode()
        if value != span.get("text", value):
            raise ValueError("span text differs from input")
        kind = span["kind"]
        if kind == "person":
            keys.add((kind, "surface", canonical(value)))
            person = person_key(value)
            if person:
                keys.add((kind, "name", person))
        elif kind == "org":
            key = canonical(value)
            keys.add((kind, "name", ALIAS_KEYS.get(key, key)))
        elif kind == "address":
            keys.add((kind, "surface", canonical(value)))
            for postcode, number, first_word in address_keys_in_text(value):
                keys.add((kind, "street", number, first_word))
        elif kind == "email":
            keys.add((kind, value.casefold()))
        elif kind == "phone":
            number = re.sub(r"\D", "", value)
            if number != "711":
                keys.add((kind, number))
        else:
            raise ValueError(f"unknown label kind: {kind}")
    return keys


def prepared_fingerprints():
    keys = set()
    for _, text, spans in training_sources():
        keys.update(fingerprints(text, spans))
    groups = set()
    for name in SILVER_SOURCES:
        for row in read_rows(SILVER / f"{name}.jsonl"):
            if row.get("source_url"):
                groups.add(family(row["source_url"]))
    for path in REVIEWED_TRAIN:
        for row in read_rows(path):
            keys.update(fingerprints(row["input"], row["expected"]))
            if row.get("source_url"):
                groups.add(family(row["source_url"]))
    paths = [SILVER / f"{name}.jsonl" for name in SILVER_SOURCES]
    paths += [SYNTHETIC / f"{split}.parquet" for split in ("train", "valid")]
    paths += list(REVIEWED_TRAIN)
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}
    return keys, groups, hashes


def connected_groups(rows, row_keys):
    indexed = collections.defaultdict(set)
    for row in rows:
        name = row["name"]
        indexed[("source_group", row["source_group"])].add(name)
        for key in row_keys[name]:
            indexed[("entity", key)].add(name)
    adjacent = {row["name"]: set() for row in rows}
    for members in indexed.values():
        if len(members) > 1:
            for name in members:
                adjacent[name].update(members - {name})
    groups = []
    unseen = set(adjacent)
    while unseen:
        pending = [min(unseen)]
        group = set()
        while pending:
            name = pending.pop()
            if name in group:
                continue
            group.add(name)
            pending.extend(adjacent[name] - group)
        unseen.difference_update(group)
        groups.append(frozenset(group))
    return sorted(groups, key=lambda group: min(group))


def month(row):
    source_id = row["source_id"]
    raw = json.loads((ROOT / "data/raw/silver/federal-register" / f"{source_id}.json").read_text())
    if raw["id"] != source_id or raw["url"] != row["source_url"]:
        raise ValueError(f"year source changed: {source_id}")
    return raw["date"][:7]


def selected_rows():
    gold = read_rows(GOLD)
    if len(gold) != 157 or len({row["name"] for row in gold}) != len(gold):
        raise ValueError("reviewed full-year contact gold changed")
    gold_manifest = json.loads(GOLD.with_suffix(".manifest.json").read_text())
    if digest(GOLD.read_bytes()) != gold_manifest["sha256"] or gold_manifest["training_eligible"]:
        raise ValueError("reviewed full-year contact gold hash changed")
    rebuilt, _, _ = gold_rows()
    if gold != rebuilt:
        raise ValueError("reviewed full-year contacts differ from pinned sources or blind reviews")
    ready_manifest = json.loads(READY_MANIFEST.read_text())
    prior_excluded = set(ready_manifest["training_excluded"])
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    evaluation_keys = set().union(*(fingerprints(row["input"], row["expected"])
                                    for row in evaluation))
    evaluation_groups = {family(row["source_url"]) for row in evaluation
                         if row.get("source_url")}
    prepared_keys, prepared_groups, prepared_hashes = prepared_fingerprints()
    row_keys = {row["name"]: fingerprints(row["input"], row["expected"])
                for row in gold}
    direct_reasons = collections.defaultdict(list)
    eval_conflicts = set()
    for row in gold:
        name = row["name"]
        keys = row_keys[name]
        if name in prior_excluded:
            direct_reasons[name].append("prior_review_exclusion")
        if keys & evaluation_keys or row["source_group"] in evaluation_groups:
            direct_reasons[name].append("evaluation_overlap")
            eval_conflicts.add(name)
        if keys & prepared_keys:
            direct_reasons[name].append("training_entity_overlap")
        if row["source_group"] in prepared_groups:
            direct_reasons[name].append("training_source_family")
    groups = connected_groups(gold, row_keys)
    unavailable = set().union(*(group for group in groups if group & direct_reasons.keys()))
    available_groups = [group for group in groups if not group & unavailable]
    by_name = {row["name"]: row for row in gold}
    months = {name: month(row) for name, row in by_name.items()}
    labels = {name: collections.Counter(span["kind"] for span in row["expected"])
              for name, row in by_name.items()}
    chosen = set()

    def score(group):
        remaining = [name for name in group if name not in chosen]
        if not remaining or len(chosen) + len(remaining) > TARGET_CASES:
            return None
        current_months = collections.Counter(months[name] for name in chosen)
        added_months = collections.Counter(months[name] for name in remaining)
        if any(current_months[key] + added > MAX_MONTH_CASES
               for key, added in added_months.items()):
            return None
        current_labels = sum((labels[name] for name in chosen), collections.Counter())
        month_gain = sum(max(0, MIN_MONTH_CASES - current_months[months[name]]) > 0
                         for name in remaining)
        field_gain = sum(min(max(0, MIN_FIELD_LABELS[kind] - current_labels[kind]),
                             sum(labels[name][kind] for name in remaining))
                         for kind in MIN_FIELD_LABELS)
        return (month_gain, field_gain, -len(remaining), -sum(len(by_name[name]["input"])
                                                             for name in remaining))

    while len(chosen) < TARGET_CASES:
        ranked = [(score(group), group) for group in available_groups]
        ranked = [(value, group) for value, group in ranked if value is not None]
        if not ranked:
            break
        _, group = max(ranked, key=lambda item: (item[0], tuple(sorted(item[1]))))
        chosen.update(group)
    selected = [row for row in gold if row["name"] in chosen]
    selected_months = collections.Counter(months[row["name"]] for row in selected)
    selected_labels = sum((labels[row["name"]] for row in selected), collections.Counter())
    if len(selected) != TARGET_CASES or len(selected_months) != 12:
        raise ValueError("strict contact holdout lacks year coverage")
    if any(selected_months[f"2025-{number:02d}"] < MIN_MONTH_CASES
           for number in range(1, 13)):
        raise ValueError("strict contact holdout lacks monthly coverage")
    if any(selected_labels[kind] < target for kind, target in MIN_FIELD_LABELS.items()):
        raise ValueError("strict contact holdout lacks field coverage")
    return selected, {
        "source_gold_sha256": digest(GOLD.read_bytes()),
        "prior_review_manifest_sha256": digest(READY_MANIFEST.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in EVALUATION},
        "prepared_sha256": prepared_hashes,
        "direct_exclusion_reasons": {name: sorted(reasons)
                                     for name, reasons in sorted(direct_reasons.items())},
        "eval_conflict_cases": sorted(eval_conflicts),
        "unavailable_cases_after_grouping": len(unavailable),
        "available_cases_before_holdout": len(gold) - len(unavailable),
        "available_month_counts": dict(sorted(collections.Counter(
            months[name] for name in by_name if name not in unavailable).items())),
        "month_counts": dict(sorted(selected_months.items())),
        "entity_counts": dict(sorted(selected_labels.items())),
        "reserved_groups": [sorted(group) for group in groups if group & chosen],
    }


def main():
    rows, provenance = selected_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha or not previous_manifest["strict_entity_holdout"]:
        raise ValueError("previous strict contact holdout changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    manifest = {
        "kind": "us_2025_year_contact_strict_holdout_v2",
        "supersedes_sha256": previous_sha,
        "cases": len(rows),
        "source_group_key": "source_group",
        "training_eligible": False,
        "strict_entity_holdout": True,
        "intended_use": "diagnostic_holdout_never_train",
        "sha256": digest(data),
        **provenance,
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen 2025 strict holdout or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2025 strict holdout changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2025 strict holdout manifest changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source and name disjoint 2025 contact holdout cases")


if __name__ == "__main__":
    main()
