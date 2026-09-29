"""Freeze complete 2025 Federal Register contact paragraphs for blind review."""

import collections
import json
import re

from address_keys import address_keys
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import SOURCE_ARTIFACT, family, normalize_surface
from freeze_2025_context_packet import (GOLD, PACKET_DIR, RAW, REVIEW, SILVER,
                                        SILVER_SOURCES, SNAPSHOT,
                                        digest, source_snapshot)
from freeze_full_notice_packet import (folded_words, has_person_alias,
                                       no_key_address_phrases, person_alias_patterns)
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex


OUT = PACKET_DIR / "us-2025-contact-blind-v4.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
HOLDOUT = SILVER / "r23/us-full-notice-blind-v1.jsonl"
HOLDOUT_REVIEWS = (
    REVIEW / "us-full-notice-review-a-v1.jsonl",
    REVIEW / "us-full-notice-review-b-v1.jsonl",
)
HEADING = ("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")
ARTIFACT = re.compile(
    r"\[\[Page \d+\]\]|\[GRAPHIC\]|\[TIFF OMITTED\]|\[FR Doc\.|"
    r"BILLING CODE|-{12,}|_{5,}|\ufffd"
)
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE = re.compile(r"\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}")
ZIP = re.compile(
    r"\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|"
    r"ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|"
    r"RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b"
)


def candidate_score(text):
    return (3 * bool(PHONE.search(text)) + 2 * bool(EMAIL.search(text)) +
            2 * bool(ZIP.search(text)) +
            2 * text.startswith("FOR FURTHER INFORMATION CONTACT:"))


def reserved_gold():
    cases = {row["name"]: row for row in lines(HOLDOUT)}
    labels = collections.defaultdict(set)
    for path in HOLDOUT_REVIEWS:
        reviews = lines(path)
        if {row["name"] for row in reviews} != set(cases):
            raise ValueError(f"reserved review does not cover all cases: {path}")
        for review in reviews:
            text = cases[review["name"]]["input"].encode()
            for span in review["entities"]:
                actual = text[span["start"]:span["end"]].decode()
                if actual != span["text"]:
                    raise ValueError(f"reserved label differs from case: {review['name']}")
                labels[review["name"]].add((span["kind"], actual))
    return [{"name": name, "input": case["input"],
             "expected": [{"kind": kind, "text": value}
                          for kind, value in sorted(labels[name])]}
            for name, case in cases.items()]


def ordered_candidates(candidates, month_cap):
    if month_cap is not None and month_cap < 1:
        raise ValueError("month cap must be positive")
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    if month_cap is None:
        return ranked
    by_month = collections.defaultdict(collections.deque)
    for candidate in ranked:
        by_month[candidate[5]].append(candidate)
    ordered = []
    while any(by_month.values()):
        for month in sorted(by_month):
            if by_month[month]:
                ordered.append(by_month[month].popleft())
    return ordered


def selected_cases(snapshot=None, year=2025, extra_gold=(), excluded_sources=(),
                   extra_train=(), month_cap=None, excluded_families=(),
                   address_screen=address_keys, address_pair_screen=False):
    snapshot = source_snapshot() if snapshot is None else snapshot
    extra_gold = list(extra_gold)
    extra_train = list(extra_train)
    gold = [row for path in GOLD for row in lines(path)] + reserved_gold() + extra_gold
    reserved_families = {family(row["source_url"]) for row in lines(HOLDOUT)}
    reserved_families.update(family(row["source_url"]) for row in extra_gold)
    reserved_families.update(family(row["source_url"]) for row in gold
                             if row.get("source_url"))
    reserved_families.update(excluded_families)
    forbidden = excluded_surface_pattern(gold)
    aliases = active_aliases(gold)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True))) if aliases else None)
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    address_pairs = {key[1:] for key in addresses} if address_pair_screen else set()
    address_phrases = no_key_address_phrases(gold)
    people = person_alias_patterns(gold)
    train_near = NearTextIndex()
    eval_near = NearTextIndex()
    selected_near = NearTextIndex()
    for name in SILVER_SOURCES:
        for row in lines(SILVER / f"{name}.jsonl"):
            train_near.add(row["id"], row["text"])
    for row in extra_train:
        train_near.add(row["name"], row["input"])
    prior_emails = {email.casefold() for row in extra_train
                    for email in EMAIL.findall(row["input"])}
    prior_phones = {re.sub(r"\D", "", phone) for row in extra_train
                    for phone in PHONE.findall(row["input"])}
    for row in gold:
        eval_near.add(row["name"], row["input"])
    counts = collections.Counter()
    candidates = []
    for source_id in sorted(snapshot):
        if source_id in excluded_sources:
            counts["used_source"] += 1
            continue
        raw = json.loads((RAW / f"{source_id}.json").read_text())
        if (raw["id"] != source_id or raw["source"] != "federal-register" or
                not raw["date"].startswith(f"{year}-") or
                not raw["url"].startswith("https://www.federalregister.gov/documents/") or
                f"/{source_id}/" not in raw["url"] or
                raw["text_url"] !=
                f"https://www.govinfo.gov/content/pkg/FR-{raw['date']}/html/{source_id}.htm"):
            raise ValueError(f"{year} source metadata differs: {source_id}")
        if SOURCE_ARTIFACT.search(raw["text"]):
            counts["source_artifact"] += 1
            continue
        if family(raw["url"]) in reserved_families:
            counts["reserved_source_family"] += 1
            continue
        for number, text in enumerate(raw["text"].split("\n\n"), 1):
            if not text.startswith(HEADING):
                continue
            counts["contact_paragraphs"] += 1
            if not 15 <= len(text.split()) <= 120:
                counts["length"] += 1
                continue
            if text not in raw["text"]:
                raise ValueError(f"contact paragraph differs from source: {source_id}-{number}")
            if ARTIFACT.search(text):
                counts["artifact"] += 1
                continue
            if not text.rstrip().endswith((".", "!", "?")):
                counts["incomplete_boundary"] += 1
                continue
            normalized = normalize_surface(text)
            if forbidden.search(normalized):
                counts["evaluation_surface"] += 1
                continue
            if alias_pattern is not None and alias_pattern.search(normalized):
                counts["evaluation_alias"] += 1
                continue
            candidate_addresses = address_screen(text)
            if candidate_addresses & addresses:
                counts["evaluation_address"] += 1
                continue
            if (address_pair_screen and
                    {key[1:] for key in candidate_addresses} & address_pairs):
                counts["evaluation_street_number"] += 1
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
            score = candidate_score(text)
            if score == 0:
                counts["no_contact_signal"] += 1
                continue
            candidates.append((score, source_id, number, raw["url"], text,
                               raw["date"][:7]))
    selected = []
    families = collections.Counter()
    emails = set()
    phones = collections.Counter()
    street_addresses = collections.Counter()
    used_sources = set()
    months = collections.Counter()
    for score, source_id, number, url, text, month in ordered_candidates(
            candidates, month_cap):
        if month_cap is not None and months[month] >= month_cap:
            counts["month_cap"] += 1
            continue
        group = family(url)
        if source_id in used_sources:
            counts["same_source"] += 1
            continue
        if families[group] >= 2:
            counts["source_family_cap"] += 1
            continue
        present_emails = {value.casefold() for value in EMAIL.findall(text)}
        present_phones = {re.sub(r"\D", "", value) for value in PHONE.findall(text)}
        present_addresses = address_keys(text)
        if present_emails & prior_emails:
            counts["repeated_prior_email"] += 1
            continue
        if present_phones & prior_phones:
            counts["repeated_prior_phone"] += 1
            continue
        if present_emails & emails:
            counts["repeated_email"] += 1
            continue
        if any(phones[value] >= 1 for value in present_phones):
            counts["repeated_phone"] += 1
            continue
        if any(street_addresses[value] >= 2 for value in present_addresses):
            counts["repeated_street_address"] += 1
            continue
        if selected_near.prior(text):
            counts["packet_near_text"] += 1
            continue
        used_sources.add(source_id)
        months[month] += 1
        families[group] += 1
        emails.update(present_emails)
        phones.update(present_phones)
        street_addresses.update(present_addresses)
        selected_near.add(source_id, text)
        selected.append({
            "name": f"us-{year}-contact-{source_id}-{number}",
            "country": "US", "source_id": source_id, "source_url": url,
            "raw_sha256": snapshot[source_id], "input": text,
        })
    selected.sort(key=lambda row: row["name"])
    return selected, counts, snapshot


def main():
    selected, counts, snapshot = selected_cases()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2025 contact packet changed")
    manifest = {
        "kind": "us_2025_contact_blind_packet",
        "intended_use": "training_candidate_after_two_blind_reviews",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in GOLD},
        "reserved_sha256": {path.name: digest(path.read_bytes())
                            for path in (HOLDOUT, *HOLDOUT_REVIEWS)},
        "source_sha256": {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                          for name in SILVER_SOURCES},
        "capture_count": len(snapshot),
        "capture_dates": sorted({json.loads((RAW / f"{source_id}.json").read_text())["date"]
                                 for source_id in snapshot}),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2025 contact packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind 2025 contact paragraphs frozen: {digest(data)}")


if __name__ == "__main__":
    main()
