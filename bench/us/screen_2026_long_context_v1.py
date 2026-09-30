"""Freeze source-distinct long US notices for independent full-span review."""

import collections
import json
import sys

from active_sources import V4_BASE_SILVER
from address_keys import address_keys_in_text
from build_contact_snippets import lines
from build_silver import normalize_surface
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE
from screen_2026_contact_expansion_v3 import CAPTURE, OUT as CONTACT_PACKET
from screen_2026_contact_expansion_v3 import PRIOR_PACKETS, ROOT, shingles
from screen_us_error_target_contacts import MALFORMED_MARKUP
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family, verify_manifest


OUT = ROOT / "data/raw/candidates/federal-register-us-long-context-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
CATEGORIES = ("address_context", "staff_context")
MAX_PER_CATEGORY = 24
MAX_PER_MONTH = 8
MAX_PER_AGENCY = 3


def source_window(raw_text, evaluation_shingles, near, forbidden=None):
    """Choose a complete 750–1800-character passage around a contact section."""
    paragraphs = raw_text.split("\n\n")
    found = []
    for index, paragraph in enumerate(paragraphs):
        if paragraph.startswith("ADDRESSES:"):
            category = "address_context"
        elif paragraph.startswith("FOR FURTHER INFORMATION CONTACT:"):
            category = "staff_context"
        else:
            continue
        for count in range(2, 6):
            if index + count > len(paragraphs):
                break
            text = "\n\n".join(paragraphs[index:index + count])
            if not 750 <= len(text) <= 1800:
                continue
            if (not text.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(text) or MALFORMED_MARKUP.search(text) or
                    not (EMAIL.search(text) or PHONE.search(text)) or
                    category == "address_context" and not address_keys_in_text(text) or
                    shingles(text, 12) & evaluation_shingles or near.prior(text) or
                    forbidden is not None and forbidden.search(normalize_surface(text)) or
                    raw_text.count(text) != 1):
                continue
            found.append((abs(len(text) - 1200), category, text, index, count))
    return min(found) if found else None


def selected_rows():
    """Select balanced long passages outside every active source and held-out text."""
    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 source capture is incomplete")
    evaluation, _ = exclusion_rows_v4()
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in V4_BASE_SILVER]
    prior_paths = (*PRIOR_PACKETS, CONTACT_PACKET)
    active = [row for path in active_paths for row in lines(path)]
    prior = [row for path in prior_paths for row in lines(path)]
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    evaluation_shingles = set().union(*(shingles(row["input"], 12)
                                        for row in evaluation))
    near = NearTextIndex()
    for row in evaluation + prior:
        near.add(row["name"], row["input"])
    for row in active:
        near.add(row["id"], row["text"])
    candidates = []
    for source in capture["sources"]:
        source_id = source["source_id"]
        raw_bytes = (CAPTURE / "sources" / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source_id or raw["url"] != source["source_url"] or
                raw.get("date", "")[:7] != source["month"] or
                source["source_group"] != source_family(raw["url"])):
            raise ValueError(f"2026 long-context source changed: {source_id}")
        key = document_key({"source_id": source_id, "source_url": source["source_url"]})
        if key in blocked_keys or source["source_group"] in blocked_groups:
            continue
        window = source_window(raw["text"], evaluation_shingles, near)
        if window is None:
            continue
        distance, category, text, paragraph, count = window
        candidates.append({
            "name": f"us-long-context-{source_id}-{paragraph + 1}",
            "country": "US", "source_id": source_id,
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"], "input": text,
            "month": source["month"], "agency": source["agency"],
            "template": source["template"], "primary_category": category,
            "paragraph": paragraph + 1, "paragraph_count": count,
            "length_distance": distance,
        })
    ranked = sorted(candidates, key=lambda row: (
        row["template"], row["length_distance"],
        digest(f"tessera-us-long-context-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    selected_shingles = set()
    months = collections.Counter()
    agencies = collections.Counter()
    categories = collections.Counter()
    for category in CATEGORIES:
        for row in ranked:
            if categories[category] >= MAX_PER_CATEGORY:
                break
            if (row["primary_category"] != category or row["source_group"] in groups or
                    months[row["month"]] >= MAX_PER_MONTH or
                    agencies[row["agency"]] >= MAX_PER_AGENCY or near.prior(row["input"]) or
                    shingles(row["input"], 12) & selected_shingles):
                continue
            selected.append(row)
            groups.add(row["source_group"])
            months[row["month"]] += 1
            agencies[row["agency"]] += 1
            categories[category] += 1
            near.add(row["name"], row["input"])
            selected_shingles.update(shingles(row["input"], 12))
    if (len(selected) != 48 or len(months) < 7 or len(agencies) < 15 or
            any(categories[category] != MAX_PER_CATEGORY for category in CATEGORIES)):
        raise ValueError(f"too few diverse long US passages: {len(selected)}, "
                         f"{dict(categories)}, {dict(months)}, {len(agencies)} agencies")
    selected.sort(key=lambda row: row["name"])
    for row in selected:
        row.pop("length_distance")
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION_PATH, *prior_paths, *active_paths)}
    return selected, categories, months, agencies, inputs


def main():
    """Freeze unlabeled passages for two separate complete-span reviews."""
    rows, categories, months, agencies, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_long_context_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "primary_categories": dict(sorted(categories.items())),
        "months": dict(sorted(months.items())),
        "agencies": len(agencies),
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US long-context packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US long-context packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US long-context packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} long US passages frozen: {dict(categories)} across "
          f"{len(months)} months and {len(agencies)} agencies")
    return rows


if __name__ == "__main__":
    main()
