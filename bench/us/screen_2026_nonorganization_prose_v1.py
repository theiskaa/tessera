"""Freeze US prose with common organization false-positive contexts."""

import collections
import json
import re
import sys

from active_sources import V3_SILVER
from address_keys import address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from build_us_v3_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v3_eval_exclusions import exclusion_rows
from freeze_2024_acronym_packet import source_paragraphs
from freeze_2025_contact_packet import EMAIL, PHONE
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from org_aliases import active_aliases
from screen_2026_failure_contacts_v2 import CAPTURE, ROOT, digest
from silver_dedupe import NearTextIndex


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family, verify_manifest


PRIOR_PACKETS = (
    CAPTURE / "screened-contact-blind-v1.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-failure-target-v2/blind-v2.jsonl",
    ROOT / "data/raw/candidates/federal-register-us-acronym-prose-v1/blind-v1.jsonl",
)
OUT = ROOT / "data/raw/candidates/federal-register-us-nonorganization-prose-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PATTERNS = {
    "law": re.compile(r"\b(?:[A-Z][A-Za-z-]+\s+){1,8}(?:Act|Code|Regulations)\b"),
    "decision": re.compile(r"\b(?:Record of Decision|Finding of No Significant Impact|"
                           r"Environmental Impact Statement|Confidential Business Information|"
                           r"Administrative Record)\b", re.I),
    "program": re.compile(r"\b(?:[A-Z][A-Za-z-]+\s+){1,8}(?:Program|Plan|Project|Survey)\b"),
    "heading": re.compile(r"^(?:SUPPLEMENTARY INFORMATION|Background|Executive Summary|"
                           r"Discussion|Summary)\s*:", re.I),
}
ARTIFACT = re.compile(r"\\\d+\\|\[\[Page \d+\]\]|<[^>]+>|-{8,}|https?://\S+", re.I)
MAX_PER_CATEGORY = 8
MAX_PER_MONTH = 6


def selected_rows():
    """Select source-distinct paragraphs with current evaluation exclusions."""
    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 source capture is incomplete")
    evaluation = exclusion_rows()
    if len(evaluation) != 343:
        raise ValueError("US evaluation changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl" for name, _ in V3_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    prior = [row for path in PRIOR_PACKETS for row in lines(path)]
    blocked_groups = {row.get("source_group") for row in evaluation + active + prior}
    blocked_ids = {row.get("source_id") for row in evaluation + active + prior}
    forbidden = excluded_surface_pattern(evaluation)
    aliases = active_aliases(evaluation, include_roster=True)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(aliases, key=len, reverse=True)), re.I) if aliases else None
    people = person_alias_patterns(evaluation)
    eval_addresses = set().union(*(address_keys_in_text(span["text"])
                                   for row in evaluation for span in row["expected"]
                                   if span["kind"] == "address"))
    near = NearTextIndex()
    for row in evaluation + prior:
        near.add(row["name"], row["input"])
    for row in active:
        near.add(row["id"], row["text"])
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
            raise ValueError(f"2026 nonorganization source changed: {source_id}")
        if source_id in blocked_ids or source["source_group"] in blocked_groups:
            counts["reserved_source"] += 1
            continue
        for start, paragraph in source_paragraphs(raw["text"]):
            if (paragraph.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")) or
                    not 30 <= len(paragraph.split()) <= 120 or
                    not paragraph.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(paragraph) or EMAIL.search(paragraph) or
                    PHONE.search(paragraph)):
                continue
            matches = {kind: match.group() for kind, pattern in PATTERNS.items()
                       if (match := pattern.search(paragraph))}
            if not matches:
                continue
            counts["target_paragraph"] += 1
            normalized = normalize_surface(paragraph)
            if forbidden.search(normalized) or (alias_pattern and alias_pattern.search(normalized)):
                counts["evaluation_surface"] += 1
                continue
            if has_person_alias(paragraph, people):
                counts["evaluation_person"] += 1
                continue
            if address_keys_in_text(paragraph) & eval_addresses:
                counts["evaluation_address"] += 1
                continue
            if near.prior(paragraph):
                counts["prior_near_text"] += 1
                continue
            source_start = len(raw["text"][:start].encode())
            source_end = source_start + len(paragraph.encode())
            if raw["text"].encode()[source_start:source_end].decode() != paragraph:
                raise ValueError(f"2026 nonorganization paragraph cannot replay: {source_id}")
            candidates.append({
                "name": f"us-nonorganization-prose-{source_id}-{source_start}",
                "country": "US", "source_id": source_id,
                "source_url": source["source_url"],
                "source_group": source["source_group"],
                "raw_sha256": source["raw_sha256"],
                "source_start_byte": source_start, "source_end_byte": source_end,
                "input": paragraph, "targets": matches,
                "month": source["month"], "agency": source["agency"],
                "template": source["template"],
            })
    selected = []
    groups = set()
    months = collections.Counter()
    categories = collections.Counter()
    ranked = sorted(candidates, key=lambda row: (
        row["template"], len(row["input"]),
        digest(f"tessera-us-nonorganization-prose-v1:{row['name']}".encode()),
    ))
    for category in PATTERNS:
        for row in ranked:
            if categories[category] >= MAX_PER_CATEGORY:
                break
            if (category not in row["targets"] or row["source_group"] in groups or
                    months[row["month"]] >= MAX_PER_MONTH or near.prior(row["input"])):
                continue
            selected.append({**row, "primary_category": category})
            groups.add(row["source_group"])
            months[row["month"]] += 1
            categories[category] += 1
            near.add(row["name"], row["input"])
    if (len(selected) < 30 or len(months) < 7 or
            any(categories[k] < 6 for k in PATTERNS)):
        raise ValueError(f"too few diverse nonorganization paragraphs: "
                         f"{len(selected)}, {dict(categories)}, {dict(months)}, {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [EVALUATION_PATH, *PRIOR_PACKETS, *active_paths]}
    return selected, counts, categories, inputs


def main():
    """Write a pinned unlabeled packet for independent review."""
    selected, counts, categories, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "federal_register_us_nonorganization_prose_blind_v1",
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
        raise ValueError("US nonorganization packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US nonorganization packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US nonorganization packet inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind US nonorganization paragraphs frozen: "
          f"{dict(categories)}")


if __name__ == "__main__":
    main()
