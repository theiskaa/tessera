"""Freeze long, organization-bearing US notice prose for blind span review."""

import collections
import json
import re
import sys

from active_sources import V4_BASE_SILVER
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from screen_2024_2025_acronym_prose_v1 import OUT as ACRONYM_PACKET
from screen_2024_2025_long_context_v1 import OUT as LONG_PACKET
from screen_2024_2025_long_context_v1 import RAW, ROOT
from screen_2026_acronym_prose_v1 import ARTIFACT
from screen_2026_contact_expansion_v3 import OUT as CONTACT_PACKET
from screen_2026_contact_expansion_v3 import PRIOR_PACKETS, shingles
from screen_2026_long_context_clean_v2 import OUT as CLEAN_PACKET
from screen_2026_long_context_v1 import OUT as FIRST_PACKET
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family


OUT = ROOT / "data/raw/candidates/federal-register-us-long-org-prose-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PRIOR = (*PRIOR_PACKETS, CONTACT_PACKET, FIRST_PACKET, CLEAN_PACKET,
         LONG_PACKET, ACRONYM_PACKET)
ACTIVE_AT_CAPTURE = V4_BASE_SILVER + (
    ("us-v4-reviewed-additions-v2", "evaluation_sha256"),
    ("us-v4-reviewed-long-additions-v1", "evaluation_sha256"),
    ("us-v4-reviewed-historical-long-v1", "evaluation_sha256"),
    ("us-v4-reviewed-historical-acronyms-v1", "evaluation_sha256"),
)
ORG_CUE = re.compile(
    r"\b(?:Department|Agency|Office|Board|Commission|Bureau|Committee|Service|"
    r"Administration|University|Division|Corporation|Institute)\b"
)
ACRONYM = re.compile(r"\b[A-Z]{2,8}\b")
GENERIC_ACRONYMS = {"US", "USA", "U.S", "FR", "CFR", "RIN", "OMB", "PRA", "NEPA",
                    "SUPPLEMENTARY", "INFORMATION", "ADDRESSES", "CONTACT"}
MAX_PER_MONTH = 8
MAX_PER_FAMILY_PREFIX = 4
MAX_REPATRIATION = 5


def acronym_count(text):
    """Count distinct all-capitals mentions after common regulatory abbreviations."""
    return len(set(ACRONYM.findall(text)) - GENERIC_ACRONYMS)


def selected_rows(extra_prior=(), min_chars=750, max_chars=1800,
                  ideal_chars=1250,
                  category_quotas=(("acronym_context", 28), ("org_context", 20)),
                  paragraphs_per_sample=1, strict_text_disjoint=False):
    """Select distinct, reviewable paragraphs outside active and held-out text."""
    evaluation, _ = exclusion_rows_v4()
    prior_paths = (*PRIOR, *extra_prior)
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_AT_CAPTURE]
    active = [row for path in active_paths for row in lines(path)]
    prior = [row for path in prior_paths for row in lines(path)]
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    blocked_groups.update(
        source_family(row["source_url"])
        for row in evaluation if row.get("source_url", "").startswith(
            "https://www.federalregister.gov/documents/"))
    forbidden = excluded_surface_pattern(evaluation)
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
    blocked_shingles = eval_shingles
    if strict_text_disjoint:
        blocked_shingles = blocked_shingles | set().union(*(
            shingles(row["input"], 12) for row in prior)) | set().union(*(
            shingles(row["text"], 12) for row in active))
    near = NearTextIndex()
    for row in evaluation + prior:
        near.add(row["name"], row["input"])
    for row in active:
        near.add(row["id"], row["text"])
    candidates = []
    inventory = []
    for path in sorted(RAW.glob("*.json")):
        raw_bytes = path.read_bytes()
        raw_sha = digest(raw_bytes)
        inventory.append((path.name, raw_sha))
        raw = json.loads(raw_bytes)
        source_id = path.stem
        if (raw["id"] != source_id or raw["source"] != "federal-register" or
                not raw["url"].startswith("https://www.federalregister.gov/documents/") or
                raw["date"][:4] not in {"2024", "2025"}):
            raise ValueError(f"historical raw notice changed: {path.name}")
        group = source_family(raw["url"])
        key = document_key({"source_id": source_id, "source_url": raw["url"]})
        if key in blocked_keys or group in blocked_groups:
            continue
        parts = raw["text"].split("\n\n")
        windows = (parts if paragraphs_per_sample == 1 else
                   ["\n\n".join(parts[i:i + paragraphs_per_sample])
                    for i in range(len(parts) - paragraphs_per_sample + 1)])
        for paragraph in windows:
            if (not min_chars <= len(paragraph) <= max_chars or
                    not paragraph.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(paragraph) or not ORG_CUE.search(paragraph) or
                    shingles(paragraph, 12) & blocked_shingles or
                    forbidden.search(normalize_surface(paragraph)) or
                    near.prior(paragraph) or raw["text"].count(paragraph) != 1):
                continue
            start = raw["text"].index(paragraph)
            end = start + len(paragraph)
            category = "acronym_context" if acronym_count(paragraph) >= 2 else "org_context"
            candidates.append({
                "name": f"us-long-org-prose-{source_id}-{len(raw['text'][:start].encode())}",
                "country": "US", "source_id": source_id,
                "source_url": raw["url"], "source_group": group,
                "raw_sha256": raw_sha, "input": paragraph,
                "source_start_byte": len(raw["text"][:start].encode()),
                "source_end_byte": len(raw["text"][:end].encode()),
                "month": raw["date"][:7], "primary_category": category,
                "acronym_count": acronym_count(paragraph),
            })
    ranked = sorted(candidates, key=lambda row: (
        abs(len(row["input"]) - ideal_chars),
        -row["acronym_count"],
        digest(f"tessera-us-long-org-prose-v1:{row['name']}".encode()),
    ))
    selected = []
    selected_groups = set()
    selected_shingles = set()
    months = collections.Counter()
    prefixes = collections.Counter()
    categories = collections.Counter()
    repatriation = 0
    for category, quota in category_quotas:
        for row in ranked:
            if categories[category] >= quota:
                break
            prefix = "-".join(row["source_group"].split("-")[:4])
            repeated_template = row["source_group"].startswith((
                "notice-of-inventory-completion", "notice-of-intended-repatriation"))
            if (row["primary_category"] != category or
                    row["source_group"] in selected_groups or
                    months[row["month"]] >= MAX_PER_MONTH or
                    prefixes[prefix] >= MAX_PER_FAMILY_PREFIX or
                    repeated_template and repatriation >= MAX_REPATRIATION or
                    shingles(row["input"], 12) & selected_shingles or
                    near.prior(row["input"])):
                continue
            selected.append(row)
            selected_groups.add(row["source_group"])
            selected_shingles.update(shingles(row["input"], 12))
            months[row["month"]] += 1
            prefixes[prefix] += 1
            categories[category] += 1
            repatriation += repeated_template
            near.add(row["name"], row["input"])
    if len(selected) != sum(quota for _, quota in category_quotas) or len(months) < 9:
        raise ValueError(f"too few diverse long organization passages: {len(selected)}, "
                         f"{dict(categories)}, {dict(months)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION_PATH, *prior_paths, *active_paths)}
    inventory_hash = digest(json.dumps(inventory, separators=(",", ":")).encode())
    return selected, categories, months, inputs, inventory_hash


def main():
    """Pin an unlabeled packet for two independent complete-span reviews."""
    rows, categories, months, inputs, inventory_hash = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_long_org_prose_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "primary_categories": dict(sorted(categories.items())),
        "months": dict(sorted(months.items())),
        "raw_inventory_sha256": inventory_hash,
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long organization packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long organization packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long organization packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} long organization passages frozen: {dict(categories)} "
          f"across {len(months)} months")
    return rows


if __name__ == "__main__":
    main()
