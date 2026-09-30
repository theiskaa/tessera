"""Freeze source-distinct 2026 US contact passages for blind label review."""

import collections
import hashlib
import json
import re
from pathlib import Path

from active_sources import V3_SILVER
from address_keys import address_keys, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from build_us_v3_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v3_eval_exclusions import exclusion_rows
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import ORG_TERM
from screen_us_error_target_contacts import LONG_ADDRESS, MALFORMED_MARKUP, ROLE
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "data/raw/candidates/federal-register-2026-contacts-v1"
OLDER_PACKET = CAPTURE / "screened-contact-blind-v1.jsonl"
OUT = ROOT / "data/raw/candidates/federal-register-us-failure-target-v2/blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,8}\b")
CATEGORIES = ("acronym_context", "org_chain", "role_near_name", "address_boundary")
MAX_CASES = 40
MAX_PER_MONTH = 6
MAX_PER_CATEGORY = 12


def digest(data):
    return hashlib.sha256(data).hexdigest()


def categories(text):
    """Describe measured failure layouts without assigning labels."""
    found = set()
    terms = {match.group().casefold() for match in ORG_TERM.finditer(text)}
    acronyms = {match.group() for match in ACRONYM.finditer(text)} - {"US", "USA"}
    if len(acronyms) >= 2 and terms:
        found.add("acronym_context")
    if len(terms) >= 2 and terms & {"branch", "division", "office", "center"}:
        found.add("org_chain")
    if ROLE.search(text) and (EMAIL.search(text) or PHONE.search(text)):
        found.add("role_near_name")
    if LONG_ADDRESS.search(text) and address_keys_in_text(text):
        found.add("address_boundary")
    return found


def active_rows():
    """Read exactly the current US real training inputs."""
    paths = [ROOT / f"data/interim/silver/{name}.jsonl" for name, _ in V3_SILVER]
    return paths, [row for path in paths for row in lines(path)]


def selected_rows():
    """Require current split identities and verify every captured source."""
    import sys

    sys.path.insert(0, str(ROOT / "bench/silver"))
    from fetch_federal_register_2026_candidates import source_family, verify_manifest

    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 capture is incomplete")
    evaluation = exclusion_rows()
    if len(evaluation) != 343:
        raise ValueError("US evaluation changed")
    active_paths, active = active_rows()
    older = lines(OLDER_PACKET)
    forbidden = excluded_surface_pattern(evaluation)
    aliases = active_aliases(evaluation, include_roster=True)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(aliases, key=len, reverse=True)), re.I) if aliases else None
    people = person_alias_patterns(evaluation)
    addresses = set().union(*(address_keys(span["text"])
                              for row in evaluation for span in row["expected"]
                              if span["kind"] == "address"))
    streets = {key[1:] for key in addresses}
    blocked_groups = {row.get("source_group") for row in evaluation + active + older}
    blocked_ids = {row.get("source_id") for row in evaluation + active + older}
    blocked_groups.discard(None)
    blocked_ids.discard(None)
    prior = [(row["name"], row["input"]) for row in evaluation + older]
    prior.extend((row["id"], row["text"]) for row in active)
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
        raw_bytes = (CAPTURE / "sources" / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw.get("id") != source_id or raw.get("url") != source["source_url"] or
                raw.get("date", "")[:7] != source["month"] or
                source["source_group"] != source_family(raw["url"])):
            raise ValueError(f"2026 source changed: {source_id}")
        if source_id in blocked_ids or source["source_group"] in blocked_groups:
            counts["reserved_source"] += 1
            continue
        for number, text in enumerate(raw["text"].split("\n\n"), 1):
            if not text.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
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
                counts["outside_target_layout"] += 1
                continue
            normalized = normalize_surface(text)
            keys = address_keys_in_text(text)
            emails = {match.casefold() for match in EMAIL.findall(text)}
            phones = {re.sub(r"\D", "", match) for match in PHONE.findall(text)}
            if forbidden.search(normalized):
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
                "name": f"us-failure-target-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": source["source_url"],
                "source_group": source["source_group"],
                "raw_sha256": source["raw_sha256"], "input": text,
                "month": source["month"], "agency": source["agency"],
                "template": source["template"], "categories": sorted(found),
            })
    selected = []
    groups = set()
    months = collections.Counter()
    category_counts = collections.Counter()
    ranked = sorted(candidates, key=lambda row: (
        len(row["categories"]) != 1,
        row["template"],
        digest(f"tessera-us-failure-target-v2:{row['name']}".encode()),
    ))
    for category in CATEGORIES:
        for row in ranked:
            if len(selected) >= MAX_CASES or category_counts[category] >= MAX_PER_CATEGORY:
                break
            if category not in row["categories"] or row["source_group"] in groups:
                continue
            if months[row["month"]] >= MAX_PER_MONTH:
                continue
            if near.prior(row["input"]):
                counts["packet_near_text"] += 1
                continue
            selected.append(row)
            groups.add(row["source_group"])
            months[row["month"]] += 1
            category_counts[category] += 1
            near.add(row["name"], row["input"])
    if (len(selected) < 24 or len(months) < 6 or
            category_counts["acronym_context"] < 4 or
            category_counts["org_chain"] < 4 or
            category_counts["role_near_name"] < 4 or
            category_counts["address_boundary"] < 4):
        raise ValueError(f"too few source-distinct US contacts: {len(selected)}, "
                         f"{dict(category_counts)}, {dict(months)}, {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [EVALUATION_PATH, OLDER_PACKET, *active_paths]}
    return selected, counts, category_counts, inputs


def main():
    """Write a pinned unlabeled packet for independent review."""
    selected, counts, category_counts, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "federal_register_us_failure_target_blind_v2",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(selected),
        "source_group_key": "source_group",
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "selection": dict(sorted(counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US failure packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US failure packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US failure packet inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind US contacts frozen: {dict(category_counts)}")


if __name__ == "__main__":
    main()
