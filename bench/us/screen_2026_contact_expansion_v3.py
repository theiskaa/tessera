"""Freeze source-distinct US contact passages with address and staff layouts."""

import collections
import json
import re
import sys

from active_sources import V4_BASE_SILVER
from address_keys import address_keys_in_text
from build_contact_snippets import lines
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from screen_2026_failure_contacts_v2 import CAPTURE, ROOT, digest
from screen_org_rich_contact_candidates import ORG_TERM
from screen_us_error_target_contacts import MALFORMED_MARKUP, ROLE
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family, verify_manifest


PRIOR_PACKETS = (
    CAPTURE / "screened-contact-blind-v1.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-failure-target-v2/blind-v2.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-acronym-prose-v1/blind-v1.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-nonorganization-prose-v1/blind-v1.jsonl",
)
OUT = ROOT / "data/raw/candidates/federal-register-us-contact-expansion-v3/blind-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
CATEGORIES = ("address_contact", "staff_contact")
MAX_PER_CATEGORY = 40
MAX_PER_MONTH = 12
MAX_PER_AGENCY = 5


def words(text):
    """Normalize prose for long partial evaluation overlap checks."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def shingles(text, size):
    """Return contiguous normalized word windows."""
    tokens = words(text)
    return {tuple(tokens[index:index + size])
            for index in range(len(tokens) - size + 1)}


def categories(text):
    """Select paragraphs with observable contact and organization cues."""
    if not (EMAIL.search(text) or PHONE.search(text)):
        return set()
    found = set()
    if address_keys_in_text(text):
        found.add("address_contact")
    if ROLE.search(text) and ORG_TERM.search(text):
        found.add("staff_contact")
    return found


def selected_rows():
    """Select balanced 2026 paragraphs outside all current training and gold sources."""
    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 source capture is incomplete")
    evaluation, _ = exclusion_rows_v4()
    if len(evaluation) != 343:
        raise ValueError("US V4 evaluation changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in V4_BASE_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    prior = [row for path in PRIOR_PACKETS for row in lines(path)]
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    people = person_alias_patterns(evaluation)
    eval_address_keys = set().union(*(address_keys_in_text(span["text"])
                                      for row in evaluation for span in row["expected"]
                                      if span["kind"] == "address"))
    eval_shingles = set().union(*(shingles(row["input"], 12) for row in evaluation))
    prior_text = [(row["name"], row["input"]) for row in evaluation + prior]
    prior_text.extend((row["id"], row["text"]) for row in active)
    near = NearTextIndex()
    for name, text in prior_text:
        near.add(name, text)
    prior_emails = {match.casefold() for _, text in prior_text
                    for match in EMAIL.findall(text)}
    prior_phones = {re.sub(r"\D", "", match) for _, text in prior_text
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
            raise ValueError(f"2026 contact expansion source changed: {source_id}")
        key = document_key({"source_id": source_id, "source_url": source["source_url"]})
        if key in blocked_keys or source["source_group"] in blocked_groups:
            counts["reserved_source"] += 1
            continue
        for number, paragraph in enumerate(raw["text"].split("\n\n"), 1):
            if not paragraph.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
                continue
            counts["contact_paragraph"] += 1
            if (not 20 <= len(paragraph.split()) <= 120 or
                    not paragraph.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(paragraph) or MALFORMED_MARKUP.search(paragraph) or
                    raw["text"].count(paragraph) != 1):
                counts["boundary_or_artifact"] += 1
                continue
            found = categories(paragraph)
            if not found:
                counts["outside_target_layout"] += 1
                continue
            if has_person_alias(paragraph, people):
                counts["evaluation_person"] += 1
                continue
            if address_keys_in_text(paragraph) & eval_address_keys:
                counts["evaluation_address"] += 1
                continue
            if shingles(paragraph, 12) & eval_shingles:
                counts["evaluation_prose"] += 1
                continue
            emails = {match.casefold() for match in EMAIL.findall(paragraph)}
            phones = {re.sub(r"\D", "", match) for match in PHONE.findall(paragraph)}
            if emails & prior_emails or phones & prior_phones:
                counts["contact_surface"] += 1
                continue
            if near.prior(paragraph):
                counts["prior_near_text"] += 1
                continue
            candidates.append({
                "name": f"us-contact-expansion-{source_id}-{number}",
                "country": "US", "source_id": source_id,
                "source_url": source["source_url"],
                "source_group": source["source_group"],
                "raw_sha256": source["raw_sha256"], "input": paragraph,
                "month": source["month"], "agency": source["agency"],
                "template": source["template"], "categories": sorted(found),
            })
    ranked = sorted(candidates, key=lambda row: (
        row["template"], len(row["input"]),
        digest(f"tessera-us-contact-expansion-v3:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    months = collections.Counter()
    agencies = collections.Counter()
    selected_categories = collections.Counter()
    for category in CATEGORIES:
        for row in ranked:
            if selected_categories[category] >= MAX_PER_CATEGORY:
                break
            if (category not in row["categories"] or row["source_group"] in groups or
                    months[row["month"]] >= MAX_PER_MONTH or
                    agencies[row["agency"]] >= MAX_PER_AGENCY or near.prior(row["input"])):
                continue
            selected.append({**row, "primary_category": category})
            groups.add(row["source_group"])
            months[row["month"]] += 1
            agencies[row["agency"]] += 1
            selected_categories[category] += 1
            near.add(row["name"], row["input"])
    if (len(selected) < 60 or len(months) < 7 or
            any(selected_categories[category] < 25 for category in CATEGORIES)):
        raise ValueError(f"too few source-distinct US contacts: {len(selected)}, "
                         f"{dict(selected_categories)}, {dict(months)}, {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [EVALUATION_PATH, *PRIOR_PACKETS, *active_paths]}
    return selected, counts, selected_categories, inputs


def main():
    """Freeze an unlabeled packet for two separate full-span reviews."""
    selected, counts, categories, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "federal_register_us_contact_expansion_blind_v3",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(selected),
        "source_group_key": "source_group",
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "selection": dict(sorted(counts.items())),
        "primary_categories": dict(sorted(categories.items())),
        "months": dict(sorted(collections.Counter(row["month"] for row in selected).items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US contact expansion packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US contact expansion packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US contact expansion packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} US contact passages frozen across "
          f"{len(manifest['months'])} months: {dict(categories)}")


if __name__ == "__main__":
    main()
