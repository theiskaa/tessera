"""Freeze source-distinct US notice excerpts for independent blind review."""

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from address_keys import address_keys
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import PAGE_MARKER, SOURCE_ARTIFACT, family, normalize_surface
from check_ready import person_key
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/silver/federal-register"
SILVER = ROOT / "data/interim/silver"
GOLD = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUT = SILVER / "r23/us-full-notice-blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ROUNDS = (1, 11, 13, 15, 17, 23)
PRIOR_PACKETS = (
    "r23/us-org-blind-v1.jsonl",
    "r23/us-org-blind-v2.jsonl",
    "r23/us-raw-org-blind-v1.jsonl",
)
SILVER_SOURCES = (
    "us-reviewed-strict-v1", "us-house-staff-v1",
    "us-house-district-offices-v1", "us-reviewed-contact-snippets-v1",
    "us-reviewed-org-paragraphs-v1", "us-r23-reviewed-org-v1",
    "us-federal-org-reviewed-v1", "us-park-contacts-v1", "us-or-districts-v1",
)
FROZEN_SHA256 = "5cd02e400d00cb9fcc5e34648703f1e96ee0823ad65aeff928df19af227c1dc9"
GOLD_SHA256 = "217c4e2e3e202662bd5ca30cf4570c6b51e67488691044bd6134842764cc0efd"
TEMPLATE_PICKS = {
    "civil_rights_advisory": {"2024-26723", "2024-30291"},
    "air_force_record_of_decision": {"2024-27538"},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ascii_fold(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def folded_words(value):
    return re.findall(r"[a-z0-9]+", ascii_fold(value))


def person_alias_patterns(gold):
    keys = {person_key(span["text"]) for row in gold for span in row["expected"]
            if span["kind"] == "person"}
    patterns = []
    for key in sorted(keys - {None}):
        first, last = map(re.escape, key)
        patterns.append(re.compile(
            rf"\b(?:{first}(?:\s+[a-z]+\.?){{0,3}}\s+{last}|"
            rf"{last},\s*{first})\b"
        ))
    return patterns


def has_person_alias(text, patterns):
    folded = ascii_fold(text)
    return any(pattern.search(folded) for pattern in patterns)


def recurring_template(row):
    slug = row["url"].rsplit("/", 1)[-1]
    if slug.startswith("notice-of-public-meeting-of-the-") and "advisory-committee" in slug:
        return "civil_rights_advisory"
    if "record-of-decision" in slug and "AFCEC/CIE" in row["text"]:
        return "air_force_record_of_decision"
    return None


def no_key_address_phrases(gold):
    phrases = set()
    for row in gold:
        for span in row["expected"]:
            if span["kind"] != "address" or address_keys(span["text"]):
                continue
            words = folded_words(span["text"])
            phrases.update(tuple(words[index:index + 4])
                           for index in range(len(words) - 3)
                           if index == 0 or any(word.isdigit()
                                                for word in words[index:index + 4]))
    return phrases


def source_ids():
    ids = set()
    snapshots = {}
    for number in ROUNDS:
        path = SILVER / f"r{number}/train.jsonl"
        snapshots[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        ids.update(row["id"] for row in lines(path) if row["country"] == "US")
    for name in PRIOR_PACKETS:
        path = SILVER / name
        snapshots[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        for row in lines(path):
            match = re.search(r"2024-\d{5}", row["name"])
            if match is None:
                raise ValueError(f"blind packet lacks source ID: {row['name']}")
            ids.add(match.group())
    return ids, snapshots


def selected_cases():
    if digest(GOLD.read_bytes()) != GOLD_SHA256:
        raise ValueError("frozen US evaluation exclusions changed")
    gold = lines(GOLD)
    forbidden = excluded_surface_pattern(gold)
    aliases = active_aliases(gold)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True)
    )) if aliases else None)
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    address_phrases = no_key_address_phrases(gold)
    people = person_alias_patterns(gold)
    used, source_hashes = source_ids()
    evaluation_near = NearTextIndex()
    training_near = NearTextIndex()
    selected_near = NearTextIndex()
    for row in gold:
        evaluation_near.add(row["name"], row["input"])
    for name in SILVER_SOURCES:
        path = SILVER / f"{name}.jsonl"
        source_hashes[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        for row in lines(path):
            training_near.add(row["id"], row["text"])
    counts = collections.Counter()
    selected = []
    families = set()
    for path in sorted(RAW.glob("2024-*.json")):
        raw_bytes = path.read_bytes()
        row = json.loads(raw_bytes)
        counts["raw"] += 1
        if row["id"] != path.stem:
            raise ValueError(f"raw source ID differs from filename: {path.name}")
        if row["id"] in used:
            counts["used_source"] += 1
            continue
        cleaned = PAGE_MARKER.sub(b"\n\n", row["text"].encode()).decode()
        words = len(cleaned.split())
        if not 150 <= words <= 500:
            counts["length"] += 1
            continue
        if (row.get("source") != "federal-register" or
                not row.get("url", "").startswith(
                    "https://www.federalregister.gov/documents/") or
                f"/{row['id']}/" not in row["url"]):
            counts["provenance"] += 1
            continue
        if (SOURCE_ARTIFACT.search(cleaned) or "\ufffd" in cleaned or
                "[[Page " in cleaned or re.search(r"_{5,}", cleaned)):
            counts["artifact"] += 1
            continue
        normalized = normalize_surface(cleaned)
        if forbidden.search(normalized):
            counts["evaluation_surface"] += 1
            continue
        if alias_pattern is not None and alias_pattern.search(normalized):
            counts["evaluation_alias"] += 1
            continue
        if address_keys(cleaned) & addresses:
            counts["evaluation_address"] += 1
            continue
        if has_person_alias(cleaned, people):
            counts["evaluation_person_alias"] += 1
            continue
        text_words = folded_words(cleaned)
        if any(tuple(text_words[index:index + 4]) in address_phrases
               for index in range(len(text_words) - 3)):
            counts["evaluation_partial_address"] += 1
            continue
        if evaluation_near.prior(cleaned):
            counts["evaluation_near_text"] += 1
            continue
        if training_near.prior(cleaned):
            counts["training_near_text"] += 1
            continue
        template = recurring_template(row)
        if template is not None and row["id"] not in TEMPLATE_PICKS[template]:
            counts["repeated_template"] += 1
            continue
        if selected_near.prior(cleaned):
            counts["packet_near_text"] += 1
            continue
        group = family(row["url"])
        if group in families:
            counts["source_family"] += 1
            continue
        families.add(group)
        selected_near.add(row["id"], cleaned)
        selected.append({
            "name": f"full-us-notice-{row['id']}", "country": "US",
            "source_id": row["id"], "source_url": row["url"],
            "raw_sha256": digest(raw_bytes), "input": cleaned,
        })
    return selected, counts, source_hashes


def main():
    selected, counts, source_hashes = selected_cases()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError(f"full US notice packet differs: {digest(data)}")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("full US notice packet changed")
    manifest = {
        "kind": "us_full_notice_blind_packet",
        "intended_use": "source_distinct_diagnostic_only",
        "training_eligible": False,
        "strict_entity_holdout": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "cases": len(selected), "selection": dict(counts),
        "evaluation_sha256": digest(GOLD.read_bytes()),
        "source_sha256": source_hashes, "sha256": digest(data),
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("full US notice packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind US full notices frozen: {digest(data)}")


if __name__ == "__main__":
    main()
