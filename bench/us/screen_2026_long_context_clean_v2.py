"""Screen long US notices against held-out names before blind review."""

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
from screen_2026_long_context_v1 import CAPTURE, OUT as FIRST_PACKET
from screen_2026_long_context_v1 import ROOT, source_window
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family, verify_manifest


OUT = ROOT / "data/raw/candidates/federal-register-us-long-context-clean-v2/blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MAX_CASES = 20
MAX_PER_MONTH = 5
MAX_PER_AGENCY = 2


def selected_rows():
    """Select diverse complete passages with no held-out entity surface in the text."""
    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 source capture is incomplete")
    evaluation, _ = exclusion_rows_v4()
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in V4_BASE_SILVER]
    prior_paths = (*PRIOR_PACKETS, CONTACT_PACKET, FIRST_PACKET)
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
    for source in capture["sources"]:
        source_id = source["source_id"]
        raw_bytes = (CAPTURE / "sources" / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source_id or raw["url"] != source["source_url"] or
                raw.get("date", "")[:7] != source["month"] or
                source["source_group"] != source_family(raw["url"])):
            raise ValueError(f"clean long-context source changed: {source_id}")
        key = document_key({"source_id": source_id, "source_url": source["source_url"]})
        if key in blocked_keys or source["source_group"] in blocked_groups:
            continue
        window = source_window(raw["text"], eval_shingles, near, forbidden)
        if window is None:
            continue
        distance, category, text, paragraph, count = window
        candidates.append({
            "name": f"us-long-clean-{source_id}-{paragraph + 1}",
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
        digest(f"tessera-us-long-clean-v2:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    selected_shingles = set()
    months = collections.Counter()
    agencies = collections.Counter()
    for row in ranked:
        if len(selected) >= MAX_CASES:
            break
        if (row["source_group"] in groups or
                months[row["month"]] >= MAX_PER_MONTH or
                agencies[row["agency"]] >= MAX_PER_AGENCY or
                shingles(row["input"], 12) & selected_shingles or
                near.prior(row["input"])):
            continue
        selected.append(row)
        groups.add(row["source_group"])
        months[row["month"]] += 1
        agencies[row["agency"]] += 1
        selected_shingles.update(shingles(row["input"], 12))
        near.add(row["name"], row["input"])
    if len(selected) != MAX_CASES or len(months) < 6 or len(agencies) < 12:
        raise ValueError(f"too few clean long passages: {len(selected)}, "
                         f"{dict(months)}, {len(agencies)} agencies")
    selected.sort(key=lambda row: row["name"])
    for row in selected:
        row.pop("length_distance")
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION_PATH, *prior_paths, *active_paths)}
    return selected, months, agencies, inputs


def main():
    """Freeze an unlabeled, surface-screened packet for independent reviewers."""
    rows, months, agencies, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_long_context_clean_blind_v2",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "months": dict(sorted(months.items())),
        "agencies": len(agencies),
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("clean long-context packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("clean long-context packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("clean long-context packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-distinct, evaluation-surface-free long passages frozen")
    return rows


if __name__ == "__main__":
    main()
