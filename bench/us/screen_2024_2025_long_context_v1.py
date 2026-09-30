"""Freeze distinct long US notices from the captured 2024–2025 source pool."""

import collections
import json
import sys

from active_sources import V4_BASE_SILVER
from build_contact_snippets import excluded_surface_pattern, lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from screen_2026_contact_expansion_v3 import OUT as CONTACT_PACKET
from screen_2026_contact_expansion_v3 import PRIOR_PACKETS, shingles
from screen_2026_long_context_clean_v2 import OUT as CLEAN_PACKET
from screen_2026_long_context_v1 import OUT as FIRST_PACKET
from screen_2026_long_context_v1 import ROOT, source_window
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family


RAW = ROOT / "data/raw/silver/federal-register"
OUT = ROOT / "data/raw/candidates/federal-register-us-long-historical-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ADDITIONS = ROOT / "data/interim/silver/us-v4-reviewed-additions-v2.jsonl"
LONG_ADDITIONS = ROOT / "data/interim/silver/us-v4-reviewed-long-additions-v1.jsonl"
PER_CATEGORY = {"address_context": 13, "staff_context": 20}
MAX_PER_MONTH = 6


def selected_rows():
    """Select address and staff passages outside active and held-out sources."""
    evaluation, _ = exclusion_rows_v4()
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in V4_BASE_SILVER] + [ADDITIONS, LONG_ADDITIONS]
    prior_paths = (*PRIOR_PACKETS, CONTACT_PACKET, FIRST_PACKET, CLEAN_PACKET)
    active = [row for path in active_paths for row in lines(path)]
    prior = [row for path in prior_paths for row in lines(path)]
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    forbidden = excluded_surface_pattern(evaluation)
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
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
            raise ValueError(f"historical raw notice identity changed: {path.name}")
        group = source_family(raw["url"])
        key = document_key({"source_id": source_id, "source_url": raw["url"]})
        if key in blocked_keys or group in blocked_groups:
            continue
        window = source_window(raw["text"], eval_shingles, near, forbidden)
        if window is None:
            continue
        distance, category, text, paragraph, count = window
        candidates.append({
            "name": f"us-long-historical-{source_id}-{paragraph + 1}",
            "country": "US", "source_id": source_id,
            "source_url": raw["url"], "source_group": group,
            "raw_sha256": raw_sha, "input": text,
            "month": raw["date"][:7], "primary_category": category,
            "paragraph": paragraph + 1, "paragraph_count": count,
            "length_distance": distance,
        })
    ranked = sorted(candidates, key=lambda row: (
        row["length_distance"],
        digest(f"tessera-us-long-historical-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    selected_shingles = set()
    months = collections.Counter()
    categories = collections.Counter()
    for category in ("address_context", "staff_context"):
        for row in ranked:
            if categories[category] >= PER_CATEGORY[category]:
                break
            if (row["primary_category"] != category or
                    row["source_group"] in groups or
                    months[row["month"]] >= MAX_PER_MONTH or
                    shingles(row["input"], 12) & selected_shingles or
                    near.prior(row["input"])):
                continue
            selected.append(row)
            groups.add(row["source_group"])
            months[row["month"]] += 1
            categories[category] += 1
            selected_shingles.update(shingles(row["input"], 12))
            near.add(row["name"], row["input"])
    if (len(selected) != 33 or len(months) < 9 or
            categories != PER_CATEGORY):
        raise ValueError(f"too few diverse historical passages: {len(selected)}, "
                         f"{dict(categories)}, {dict(months)}")
    selected.sort(key=lambda row: row["name"])
    for row in selected:
        row.pop("length_distance")
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION_PATH, *prior_paths, *active_paths)}
    inventory_hash = digest(json.dumps(inventory, separators=(",", ":")).encode())
    return selected, categories, months, inputs, inventory_hash


def main():
    """Freeze a source-bound, unlabeled packet for independent review."""
    rows, categories, months, inputs, inventory_hash = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_long_historical_blind_v1",
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
        raise ValueError("historical long-context packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical long-context packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical long-context packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    groups = {row["source_group"] for row in rows}
    print(f"{len(rows)} historical long US passages frozen: {dict(categories)} "
          f"across {len(months)} months and {len(groups)} source groups")
    return rows


if __name__ == "__main__":
    main()
