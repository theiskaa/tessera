"""Freeze US action sentences with source-proven organization acronyms."""

import collections
import json
import re
import sys
from pathlib import Path

from active_sources import V3_SILVER
from address_keys import address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from build_us_v3_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v3_eval_exclusions import exclusion_rows
from freeze_2024_acronym_packet import END_PUNCT, prose_acronyms, prose_score
from freeze_2024_acronym_packet import sentence_slices, source_paragraphs
from freeze_2024_acronym_packet import strict_expanded_bodies
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from screen_2026_failure_contacts_v2 import CAPTURE, ROOT, digest
from silver_dedupe import NearTextIndex


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family, verify_manifest


PRIOR_PACKETS = (
    CAPTURE / "screened-contact-blind-v1.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-failure-target-v2/blind-v2.jsonl",
)
OUT = ROOT / "data/raw/candidates/federal-register-us-acronym-prose-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ARTIFACT = re.compile(r"\\\d+\\|\[\[Page \d+\]\]|<[^>]+>|-{8,}|https?://\S+", re.I)
MAX_CASES = 40
MAX_PER_MONTH = 6
MAX_PER_ACRONYM = 2
MAX_PER_AGENCY = 4


def active_inputs():
    """Read the present US silver sources and their labeled acronym surfaces."""
    paths = [ROOT / f"data/interim/silver/{name}.jsonl" for name, _ in V3_SILVER]
    rows = [row for path in paths for row in lines(path)]
    acronyms = set()
    for row in rows:
        source = row["text"].encode()
        for span in row["entities"]:
            if span["kind"] == "org":
                value = source[span["start"]:span["end"]].decode()
                if re.fullmatch(r"[A-Z]{2,9}", value):
                    acronyms.add(value)
    return paths, rows, acronyms


def selected_rows():
    """Select source-distinct prose after current train and evaluation screens."""
    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 source capture is incomplete")
    evaluation = exclusion_rows()
    if len(evaluation) != 343:
        raise ValueError("US evaluation changed")
    active_paths, active, trained_acronyms = active_inputs()
    prior = [row for path in PRIOR_PACKETS for row in lines(path)]
    blocked_groups = {row.get("source_group") for row in evaluation + active + prior}
    blocked_ids = {row.get("source_id") for row in evaluation + active + prior}
    blocked_groups.discard(None)
    blocked_ids.discard(None)
    forbidden = excluded_surface_pattern(evaluation)
    aliases = active_aliases(evaluation, include_roster=True)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(aliases, key=len, reverse=True)), re.I) if aliases else None
    people = person_alias_patterns(evaluation)
    eval_addresses = set().union(*(address_keys_in_text(span["text"])
                                   for row in evaluation for span in row["expected"]
                                   if span["kind"] == "address"))
    prior_texts = [(row["name"], row["input"]) for row in evaluation + prior]
    prior_texts.extend((row["id"], row["text"]) for row in active)
    near = NearTextIndex()
    for name, text in prior_texts:
        near.add(name, text)
    counts = collections.Counter()
    candidates = []
    for source in capture["sources"]:
        source_id = source["source_id"]
        raw_bytes = (CAPTURE / "sources" / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw.get("id") != source_id or raw.get("url") != source["source_url"] or
                raw.get("date", "")[:7] != source["month"] or
                source["source_group"] != source_family(raw["url"])):
            raise ValueError(f"2026 acronym source changed: {source_id}")
        if source_id in blocked_ids or source["source_group"] in blocked_groups:
            counts["reserved_source"] += 1
            continue
        expansions = strict_expanded_bodies(raw["text"])
        if not expansions:
            continue
        for paragraph_start, paragraph in source_paragraphs(raw["text"]):
            if paragraph.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
                continue
            for start, end, sentence in sentence_slices(paragraph):
                if (not 15 <= len(sentence.split()) <= 100 or
                        not END_PUNCT.search(sentence) or ARTIFACT.search(sentence)):
                    continue
                acronyms = [name for name in prose_acronyms(sentence, expansions)
                            if name not in trained_acronyms]
                if not acronyms:
                    continue
                counts["unseen_acronym_action"] += 1
                normalized = normalize_surface(sentence)
                if forbidden.search(normalized):
                    counts["evaluation_surface"] += 1
                    continue
                if alias_pattern and alias_pattern.search(normalized):
                    counts["evaluation_alias"] += 1
                    continue
                if has_person_alias(sentence, people):
                    counts["evaluation_person"] += 1
                    continue
                if address_keys_in_text(sentence) & eval_addresses:
                    counts["evaluation_address"] += 1
                    continue
                if near.prior(sentence):
                    counts["prior_near_text"] += 1
                    continue
                acronym = max(acronyms, key=lambda name: prose_score(sentence, name))
                source_start = paragraph_start + start
                source_end = paragraph_start + end
                start_byte = len(raw["text"][:source_start].encode())
                end_byte = len(raw["text"][:source_end].encode())
                if raw["text"].encode()[start_byte:end_byte].decode() != sentence:
                    raise ValueError(f"2026 acronym sentence cannot replay: {source_id}")
                evidence_start, evidence_end = expansions[acronym]
                candidates.append({
                    "name": f"us-acronym-prose-{source_id}-{start_byte}",
                    "country": "US", "source_id": source_id,
                    "source_url": source["source_url"],
                    "source_group": source["source_group"],
                    "raw_sha256": source["raw_sha256"],
                    "input": sentence, "acronym": acronym,
                    "expansion_evidence": raw["text"][evidence_start:evidence_end],
                    "source_start_byte": start_byte, "source_end_byte": end_byte,
                    "month": source["month"], "agency": source["agency"],
                    "template": source["template"],
                })
    ranked = sorted(candidates, key=lambda row: (
        row["template"],
        tuple(-value if isinstance(value, int) else value
              for value in prose_score(row["input"], row["acronym"])),
        digest(f"tessera-us-acronym-prose-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    months = collections.Counter()
    agencies = collections.Counter()
    acronyms = collections.Counter()
    for row in ranked:
        if len(selected) >= MAX_CASES:
            break
        if (row["source_group"] in groups or months[row["month"]] >= MAX_PER_MONTH or
                agencies[row["agency"]] >= MAX_PER_AGENCY or
                acronyms[row["acronym"]] >= MAX_PER_ACRONYM):
            continue
        if near.prior(row["input"]):
            counts["packet_near_text"] += 1
            continue
        selected.append(row)
        groups.add(row["source_group"])
        months[row["month"]] += 1
        agencies[row["agency"]] += 1
        acronyms[row["acronym"]] += 1
        near.add(row["name"], row["input"])
    if (len(selected) < 30 or len(months) < 7 or len(acronyms) < 20 or
            sum(row["template"] for row in selected) > 6):
        raise ValueError(f"too few diverse acronym sentences: {len(selected)}, "
                         f"{dict(months)}, {dict(acronyms)}, {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [EVALUATION_PATH, *PRIOR_PACKETS, *active_paths]}
    return selected, counts, inputs


def main():
    """Write a pinned unlabeled packet for separate reviews."""
    selected, counts, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "federal_register_us_acronym_prose_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(selected),
        "source_group_key": "source_group",
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "selection": dict(sorted(counts.items())),
        "months": dict(sorted(collections.Counter(row["month"] for row in selected).items())),
        "acronyms": dict(sorted(collections.Counter(row["acronym"] for row in selected).items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US acronym packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US acronym packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US acronym packet inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} US acronym action sentences frozen from "
          f"{len(manifest['acronyms'])} acronyms")


if __name__ == "__main__":
    main()
