"""Freeze source-backed US acronym prose for independent blind review."""

import collections
import json
import re

from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import SOURCE_ARTIFACT, family, normalize_surface
from freeze_2025_contact_packet import (ARTIFACT, GOLD, HOLDOUT,
                                        HOLDOUT_REVIEWS, RAW, REVIEW, SILVER,
                                        SILVER_SOURCES, digest, reserved_gold)
from freeze_full_notice_packet import (PRIOR_PACKETS, ROUNDS, folded_words,
                                       has_person_alias, no_key_address_phrases,
                                       person_alias_patterns)
from org_aliases import active_aliases
from silver_dedupe import NearTextIndex


PACKET_DIR = SILVER / "r24"
SNAPSHOT = PACKET_DIR / "us-2024-acronym-source-snapshot-v1.json"
OUT = PACKET_DIR / "us-2024-acronym-prose-blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PREVIOUS_OUT = PACKET_DIR / "us-2024-acronym-prose-blind-v1.jsonl"
PREVIOUS_MANIFEST = PREVIOUS_OUT.with_suffix(".manifest.json")
ROOT = RAW.parents[3]
CONTACT_PACKETS = (
    PACKET_DIR / "us-2024-contact-blind-v1.jsonl",
    PACKET_DIR / "us-2024-contact-blind-v2.jsonl",
)
CONTACT_GOLDS = (
    REVIEW / "us-2024-contact-gold-v2.jsonl",
    REVIEW / "us-2024-contact-gold-v3.jsonl",
)
CONTACT_HOLDOUTS = (
    REVIEW / "us-2025-contact-holdout-v1.jsonl",
    REVIEW / "us-2025-contact-holdout-v2.jsonl",
)
CONTACT_GOLD = REVIEW / "us-2025-contact-gold-v5.jsonl"
ACRONYM = re.compile(r"\(([A-Z]{2,9})\)")
ORG_HEAD = re.compile(
    r"\b(?:Department|Bureau|Office|Administration|Commission|Council|Agency|"
    r"Authority|Corporation|Institute|University|Division|Association|Board|"
    r"Center|Centre|Committee|Foundation|Laboratory|Company|Inc|LLC)\b", re.I
)
STRICT_ORG_HEAD = re.compile(
    r"\b(?:Department|Bureau|Office|Administration|Commission|Council|Agency|"
    r"Authority|Corporation|Institute|University|Division|Association|Board|"
    r"Center|Centre|Committee|Foundation|Laboratory|Company|Inc|LLC|"
    r"Conference|Services|Group)\b"
)
TITLE_CONNECTORS = {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}
NUMBERED_HEADING = re.compile(r"(?:^|[.!?]\s+)\d{1,3}\.\s+[A-Z]")
TRUNCATED_CITATION_START = re.compile(r"^[A-Z]\.\s+\d")
STRUCTURED_FIELD_START = re.compile(r"^(?:Justification|Time and Date):", re.I)
PROSE_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|shall|may|must|uses?|provides?|"
    r"receives?|collects?|administers?|supports?|operates?|seeks?|publishes?|"
    r"proposes?|issues?|develops?|coordinates?|works?|conducts?|determines?|"
    r"participates?|recruits?|identifies?|awards?|plans?|notifies?|urges?|"
    r"invites?|requires?|files?|requests?|announces?|reviews?|approves?|issued|"
    r"authorizes?|petitions?|advises?|sets?|establishes?|maintains?|submits?|"
    r"agrees?|finds?|recommends?|implements?|helps?)\b", re.I
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
END_PUNCT = re.compile(r"[.!?][\"')\]]*$")
NON_BODY_ACRONYMS = {
    "PRA", "FACA", "CFR", "FRS", "FIRS", "EDIS", "CAS", "NTL", "FMS",
    "FSR", "TDD", "TTY", "NCAP", "DFO", "ICR", "NEPA", "NAGPRA", "IRB",
    "FMP", "OMB", "RPA",
}
NON_BODY_ENDINGS = re.compile(
    r"\b(?:Act|Form|Rule|Regulation|Report|Program|Programme|System|Plan|"
    r"Project|Code|Number|Portal|Website|Survey|Surveys)\s*$", re.I
)
ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "No.", "Nos.", "Inc.", "Corp.",
    "Ltd.", "St.", "Ave.", "a.m.", "p.m.", "etc.", "et.", "al.",
}


def source_snapshot():
    """Pin the complete captured 2024 source pool before candidate selection."""
    if SNAPSHOT.exists():
        snapshot = json.loads(SNAPSHOT.read_text())
    else:
        snapshot = {path.stem: digest(path.read_bytes())
                    for path in sorted(RAW.glob("2024-*.json"))}
        if not snapshot:
            raise ValueError("no captured 2024 Federal Register sources")
        PACKET_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    for source_id, expected in snapshot.items():
        if not re.fullmatch(r"2024-\d{5}", source_id):
            raise ValueError(f"invalid source ID in snapshot: {source_id}")
        if digest((RAW / f"{source_id}.json").read_bytes()) != expected:
            raise ValueError(f"captured 2024 source changed: {source_id}")
    return snapshot


def source_paragraphs(text):
    """Yield exact paragraph slices and their character offsets in captured text."""
    cursor = 0
    for paragraph in text.split("\n\n"):
        start = text.find(paragraph, cursor)
        if start < 0:
            raise ValueError("captured paragraph cannot be located")
        yield start, paragraph
        cursor = start + len(paragraph)


def sentence_slices(paragraph):
    """Yield conservative complete sentence windows with exact local offsets."""
    start = 0
    for match in SENTENCE_BREAK.finditer(paragraph):
        prior = paragraph[start:match.start()].split()
        if not prior:
            continue
        word = re.sub(r"^[^A-Za-z0-9]+", "", prior[-1])
        if (word in ABBREVIATIONS or
                re.fullmatch(r"(?:[A-Za-z]\.){1,5}", word) or
                re.fullmatch(r"\d+\.", word)):
            continue
        yield _trim_slice(paragraph, start, match.start())
        start = match.end()
    yield _trim_slice(paragraph, start, len(paragraph))


def _trim_slice(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, text[start:end]


def expanded_bodies(text):
    """Find acronym expansions with an organization headword in this source."""
    found = {}
    for match in ACRONYM.finditer(text):
        acronym = match[1]
        if acronym in NON_BODY_ACRONYMS:
            continue
        prefix = text[max(0, match.start() - 65):match.start()]
        if not ORG_HEAD.search(prefix) or NON_BODY_ENDINGS.search(prefix):
            continue
        found.setdefault(acronym, (max(0, match.start() - 65), match.end()))
    return found


def strict_expanded_bodies(text):
    """Keep acronym expansions tied to a nearby titled organization name."""
    found = {}
    for match in ACRONYM.finditer(text):
        acronym = match[1]
        if acronym in NON_BODY_ACRONYMS:
            continue
        if re.search(r"\[\[Page \d+\]\]", text[max(0, match.start() - 200):match.start()]):
            continue
        start = max(0, match.start() - 160)
        prefix = text[start:match.start()]
        boundary = prefix.rfind("\n\n")
        if boundary >= 0:
            start += boundary + 2
            prefix = prefix[boundary + 2:]
        if "[[Page " in prefix or NON_BODY_ENDINGS.search(prefix):
            continue
        for head in reversed(list(STRICT_ORG_HEAD.finditer(prefix))):
            tail = prefix[head.end():]
            if re.search(r"[.!?]\s+[A-Z]", tail):
                continue
            words = re.findall(r"[A-Za-z]+", tail.replace("'s", "").replace("’s", ""))
            if any(word.casefold() not in TITLE_CONNECTORS and not word[0].isupper()
                   for word in words):
                continue
            if re.search(r"[^\w\s,.'’&/\-]", tail):
                continue
            before = list(re.finditer(r"\S+", prefix[:head.start()]))
            evidence_start = before[-min(6, len(before))].start() if before else head.start()
            found.setdefault(acronym, (start + evidence_start, match.end()))
            break
    return found


def prose_acronyms(sentence, expansions):
    """Return expanded acronyms used with an action verb in this sentence."""
    found = []
    for acronym in sorted(expansions):
        pattern = re.compile(r"(?<![\w@./-])" + re.escape(acronym) +
                             r"(?![\w./-])(?:['’]s)?\s+")
        if any(PROSE_VERB.match(sentence, match.end()) for match in pattern.finditer(sentence)):
            found.append(acronym)
    return found


def prose_score(sentence, acronym):
    """Prefer short direct agency-action prose over titles and indirect clauses."""
    match = re.search(r"(?<!\w)" + re.escape(acronym) + r"(?!\w)", sentence)
    position = match.start() if match else len(sentence)
    return (position <= 24, ":" not in sentence, len(sentence.split()) <= 60,
            -position, -len(sentence))


def used_source_ids(year=2024, extra_paths=(), family_paths=(), ignored_paths=()):
    """Find notices from the selected year represented in earlier US data."""
    ids = set()
    families = set()
    paths = set()
    for round_number in ROUNDS:
        for split in ("train", "valid", "test"):
            path = SILVER / f"r{round_number}/{split}.jsonl"
            if path.exists():
                paths.add(path)
    paths.update(SILVER / f"{name}.jsonl" for name in SILVER_SOURCES)
    paths.update(SILVER.glob("us-*.jsonl"))
    paths.update(SILVER / name for name in PRIOR_PACKETS)
    paths.add(HOLDOUT)
    paths.update(CONTACT_PACKETS)
    paths.update(CONTACT_GOLDS)
    paths.update(extra_paths)
    paths.update(family_paths)
    paths.difference_update(ignored_paths)
    family_paths = set(family_paths)
    hashes = {}
    source_id_pattern = re.compile(rf"{year}-\d{{5}}")
    source_url_pattern = re.compile(rf"/{year}-\d{{5}}/")
    for path in sorted(paths):
        if not path.exists():
            raise ValueError(f"required source-use input missing: {path}")
        hashes[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        for row in lines(path):
            url = row.get("source_url", "")
            if isinstance(url, str) and (source_url_pattern.search(url) or
                                         path in family_paths):
                families.add(family(url))
            for key in ("id", "name", "source_id", "source_url"):
                value = row.get(key, "")
                if isinstance(value, str):
                    ids.update(source_id_pattern.findall(value))
    for source_id in ids:
        path = RAW / f"{source_id}.json"
        if path.exists():
            families.add(family(json.loads(path.read_text())["url"]))
    return ids, families, hashes


def ordered_candidates(candidates, month_cap):
    if month_cap is not None and month_cap < 1:
        raise ValueError("month cap must be positive")
    ranked = sorted(candidates,
                    key=lambda row: (tuple(-x for x in row[0]), row[1], row[5]))
    if month_cap is None:
        return ranked
    by_month = collections.defaultdict(collections.deque)
    for candidate in ranked:
        by_month[candidate[11]].append(candidate)
    ordered = []
    while any(by_month.values()):
        for month in sorted(by_month):
            if by_month[month]:
                ordered.append(by_month[month].popleft())
    return ordered


def selected_cases(snapshot=None, year=2024, extra_gold=(), extra_train=(),
                   excluded_sources=(), excluded_families=(), month_cap=None,
                   max_acronym_uses=None, used_paths=(), strict_expansions=False,
                   address_screen=address_keys, address_pair_screen=False,
                   family_paths=(), ignored_paths=()):
    """Select unlabeled prose windows after all reserved-data screens."""
    if max_acronym_uses is not None and max_acronym_uses < 1:
        raise ValueError("acronym cap must be positive")
    snapshot = source_snapshot() if snapshot is None else snapshot
    gold = [row for path in (*GOLD, *CONTACT_HOLDOUTS, CONTACT_GOLD)
            for row in lines(path)]
    gold.extend(reserved_gold())
    gold.extend(extra_gold)
    forbidden = excluded_surface_pattern(gold)
    aliases = active_aliases(gold)
    alias_pattern = (re.compile("|".join(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        for name in sorted(aliases, key=len, reverse=True)
    )) if aliases else None)
    addresses = set().union(*(address_keys(span["text"]) for row in gold
                              for span in row["expected"] if span["kind"] == "address"))
    address_pairs = {key[1:] for key in addresses} if address_pair_screen else set()
    address_phrases = no_key_address_phrases(gold)
    people = person_alias_patterns(gold)
    used, used_families, input_hashes = used_source_ids(year, used_paths,
                                                        family_paths,
                                                        ignored_paths)
    used.update(excluded_sources)
    reserved_families = {family(row["source_url"]) for row in lines(HOLDOUT)}
    reserved_families.update(used_families)
    reserved_families.update(excluded_families)
    for path in CONTACT_HOLDOUTS:
        reserved_families.update(family(row["source_url"]) for row in lines(path))
    reserved_families.update(family(row["source_url"]) for row in lines(CONTACT_GOLD))
    reserved_families.update(family(row["source_url"]) for row in gold
                             if row.get("source_url"))
    evaluation_near = NearTextIndex()
    training_near = NearTextIndex()
    selected_near = NearTextIndex()
    for row in gold:
        evaluation_near.add(row["name"], row["input"])
    for name in SILVER_SOURCES:
        for row in lines(SILVER / f"{name}.jsonl"):
            training_near.add(row["id"], row["text"])
    for row in extra_train:
        training_near.add(row["name"], row["input"])
    counts = collections.Counter()
    candidates = []
    for source_id in sorted(snapshot):
        if source_id in used:
            counts["used_source"] += 1
            continue
        raw_bytes = (RAW / f"{source_id}.json").read_bytes()
        source = json.loads(raw_bytes)
        if (source["id"] != source_id or source["source"] != "federal-register" or
                not source["date"].startswith(f"{year}-") or
                not source["url"].startswith("https://www.federalregister.gov/documents/") or
                f"/{source_id}/" not in source["url"] or
                source["text_url"] !=
                f"https://www.govinfo.gov/content/pkg/FR-{source['date']}/html/{source_id}.htm"):
            raise ValueError(f"{year} source metadata differs: {source_id}")
        if SOURCE_ARTIFACT.search(source["text"]):
            counts["source_artifact"] += 1
            continue
        group = family(source["url"])
        if group in reserved_families:
            counts["reserved_source_family"] += 1
            continue
        expansions = (strict_expanded_bodies(source["text"]) if strict_expansions
                      else expanded_bodies(source["text"]))
        if not expansions:
            counts["no_body_expansion"] += 1
            continue
        for paragraph_start, paragraph in source_paragraphs(source["text"]):
            if ARTIFACT.search(paragraph):
                continue
            for local_start, local_end, sentence in sentence_slices(paragraph):
                acronyms = prose_acronyms(sentence, expansions)
                if not acronyms:
                    continue
                counts["acronym_prose"] += 1
                if strict_expansions and NUMBERED_HEADING.search(sentence):
                    counts["numbered_heading"] += 1
                    continue
                if strict_expansions and (TRUNCATED_CITATION_START.search(sentence) or
                                          STRUCTURED_FIELD_START.search(sentence)):
                    counts["structured_or_truncated_sentence"] += 1
                    continue
                if (strict_expansions and
                        any(re.search(r"https?://\S*" + re.escape(acronym), sentence, re.I)
                            for acronym in acronyms)):
                    counts["acronym_inside_url"] += 1
                    continue
                if (not 12 <= len(sentence.split()) <= 100 or
                        not END_PUNCT.search(sentence) or
                        not re.match(r"[A-Z]", sentence)):
                    counts["length_or_boundary"] += 1
                    continue
                normalized = normalize_surface(sentence)
                if forbidden.search(normalized):
                    counts["evaluation_surface"] += 1
                    continue
                if alias_pattern is not None and alias_pattern.search(normalized):
                    counts["evaluation_alias"] += 1
                    continue
                candidate_addresses = address_screen(sentence)
                if candidate_addresses & addresses:
                    counts["evaluation_address"] += 1
                    continue
                if (address_pair_screen and
                        {key[1:] for key in candidate_addresses} & address_pairs):
                    counts["evaluation_street_number"] += 1
                    continue
                if has_person_alias(sentence, people):
                    counts["evaluation_person_alias"] += 1
                    continue
                words = folded_words(sentence)
                if any(tuple(words[index:index + 4]) in address_phrases
                       for index in range(len(words) - 3)):
                    counts["evaluation_partial_address"] += 1
                    continue
                if evaluation_near.prior(sentence):
                    counts["evaluation_near_text"] += 1
                    continue
                if training_near.prior(sentence):
                    counts["training_near_text"] += 1
                    continue
                start = paragraph_start + local_start
                end = paragraph_start + local_end
                if source["text"][start:end] != sentence:
                    raise ValueError(f"sentence changed from source: {source_id}")
                acronym = acronyms[0]
                evidence_start, evidence_end = expansions[acronym]
                candidates.append((prose_score(sentence, acronym), source_id,
                                   group, source, snapshot[source_id], start,
                                   end, sentence, acronym, evidence_start,
                                   evidence_end, source["date"][:7]))
    selected = []
    families = set()
    source_ids = set()
    months = collections.Counter()
    acronyms_used = collections.Counter()
    for (_, source_id, group, source, raw_sha, start, end, sentence, acronym,
         evidence_start, evidence_end, month) in ordered_candidates(candidates, month_cap):
        if month_cap is not None and months[month] >= month_cap:
            counts["month_cap"] += 1
            continue
        if source_id in source_ids:
            counts["same_source"] += 1
            continue
        if group in families:
            counts["source_family"] += 1
            continue
        if max_acronym_uses is not None and acronyms_used[acronym] >= max_acronym_uses:
            counts["acronym_surface_cap"] += 1
            continue
        if selected_near.prior(sentence):
            counts["packet_near_text"] += 1
            continue
        source_ids.add(source_id)
        families.add(group)
        months[month] += 1
        acronyms_used[acronym] += 1
        selected_near.add(source_id, sentence)
        source_text = source["text"]
        selected.append({
            "name": f"us-{year}-acronym-prose-{source_id}-{start}",
            "country": "US", "source_id": source_id,
            "source_url": source["url"], "text_url": source["text_url"],
            "source_group": group, "raw_sha256": raw_sha,
            "source_byte_start": len(source_text[:start].encode()),
            "source_byte_end": len(source_text[:end].encode()),
            "acronym": acronym,
            "expansion_evidence": source_text[evidence_start:evidence_end],
            "expansion_byte_start": len(source_text[:evidence_start].encode()),
            "expansion_byte_end": len(source_text[:evidence_end].encode()),
            "input": sentence,
        })
    selected.sort(key=lambda row: row["name"])
    return selected, counts, input_hashes, snapshot


def main():
    """Write the immutable blind packet and its input manifest."""
    previous = json.loads(PREVIOUS_MANIFEST.read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_sha != previous["sha256"] or previous["training_eligible"]:
        raise ValueError("original acronym packet changed")
    selected, counts, input_hashes, snapshot = selected_cases()
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in selected) + "\n").encode()
    manifest = {
        "kind": "us_2024_acronym_prose_blind_packet",
        "version": 2,
        "supersedes_sha256": previous_sha,
        "intended_use": "training_candidate_after_two_blind_reviews",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_group",
        "source_snapshot_sha256": digest(SNAPSHOT.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes())
                              for path in (*GOLD, *CONTACT_HOLDOUTS, CONTACT_GOLD)},
        "reserved_sha256": {path.name: digest(path.read_bytes())
                            for path in (HOLDOUT, *HOLDOUT_REVIEWS)},
        "input_sha256": input_hashes,
        "capture_count": len(snapshot),
        "cases": len(selected),
        "selection": dict(counts),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen 2024 acronym v2 packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2024 acronym v2 packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2024 acronym v2 packet inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind 2024 acronym prose cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
