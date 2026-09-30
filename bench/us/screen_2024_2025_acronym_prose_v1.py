"""Freeze distinct historical US organization-acronym prose for blind review."""

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
from freeze_2024_acronym_packet import END_PUNCT, prose_acronyms, prose_score
from freeze_2024_acronym_packet import sentence_slices, source_paragraphs
from freeze_2024_acronym_packet import strict_expanded_bodies
from screen_2024_2025_long_context_v1 import OUT as LONG_PACKET
from screen_2024_2025_long_context_v1 import RAW, ROOT
from screen_2026_acronym_prose_v1 import ARTIFACT
from silver_dedupe import NearTextIndex
from source_identity import document_key


sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import source_family


OUT = ROOT / "data/raw/candidates/federal-register-us-historical-acronym-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ADDITIONS = ROOT / "data/interim/silver/us-v4-reviewed-additions-v2.jsonl"
LONG_ADDITIONS = ROOT / "data/interim/silver/us-v4-reviewed-long-additions-v1.jsonl"
MAX_CASES = 22
MAX_PER_YEAR = 14
MAX_PER_MONTH = 4
MAX_PER_ACRONYM = 2


def selected_rows():
    """Select unseen, source-proven acronym mentions outside held-out text."""
    evaluation, _ = exclusion_rows_v4()
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in V4_BASE_SILVER] + [ADDITIONS, LONG_ADDITIONS]
    active = [row for path in active_paths for row in lines(path)]
    prior = lines(LONG_PACKET)
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    forbidden = excluded_surface_pattern(evaluation)
    trained = set()
    for row in active:
        source = row["text"].encode()
        for span in row["entities"]:
            if span["kind"] == "org":
                surface = source[span["start"]:span["end"]].decode()
                if re.fullmatch(r"[A-Z]{2,9}", surface):
                    trained.add(surface)
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
            raise ValueError(f"historical acronym raw source changed: {path.name}")
        group = source_family(raw["url"])
        key = document_key({"source_id": source_id, "source_url": raw["url"]})
        if key in blocked_keys or group in blocked_groups:
            continue
        expansions = strict_expanded_bodies(raw["text"])
        if not expansions:
            continue
        for paragraph_start, paragraph in source_paragraphs(raw["text"]):
            if paragraph.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
                continue
            for start, end, sentence in sentence_slices(paragraph):
                if (not 15 <= len(sentence.split()) <= 100 or
                        not END_PUNCT.search(sentence) or ARTIFACT.search(sentence) or
                        near.prior(sentence) or
                        forbidden.search(normalize_surface(sentence))):
                    continue
                acronyms = [name for name in prose_acronyms(sentence, expansions)
                            if name not in trained]
                if not acronyms:
                    continue
                acronym = max(acronyms, key=lambda value: prose_score(sentence, value))
                source_start = paragraph_start + start
                source_end = paragraph_start + end
                start_byte = len(raw["text"][:source_start].encode())
                end_byte = len(raw["text"][:source_end].encode())
                if raw["text"].encode()[start_byte:end_byte].decode() != sentence:
                    raise ValueError(f"historical acronym sentence cannot replay: {source_id}")
                evidence_start, evidence_end = expansions[acronym]
                candidates.append({
                    "name": f"us-historical-acronym-{source_id}-{start_byte}",
                    "country": "US", "source_id": source_id,
                    "source_url": raw["url"], "source_group": group,
                    "raw_sha256": raw_sha, "input": sentence,
                    "acronym": acronym,
                    "expansion_evidence": raw["text"][evidence_start:evidence_end],
                    "source_start_byte": start_byte,
                    "source_end_byte": end_byte,
                    "month": raw["date"][:7],
                })
    ranked = sorted(candidates, key=lambda row: (
        tuple(-value if isinstance(value, int) else value
              for value in prose_score(row["input"], row["acronym"])),
        digest(f"tessera-historical-acronym-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    months = collections.Counter()
    years = collections.Counter()
    acronyms = collections.Counter()
    for row in ranked:
        if len(selected) >= MAX_CASES:
            break
        year = row["month"][:4]
        if (row["source_group"] in groups or
                years[year] >= MAX_PER_YEAR or
                months[row["month"]] >= MAX_PER_MONTH or
                acronyms[row["acronym"]] >= MAX_PER_ACRONYM or
                near.prior(row["input"])):
            continue
        selected.append(row)
        groups.add(row["source_group"])
        months[row["month"]] += 1
        years[year] += 1
        acronyms[row["acronym"]] += 1
        near.add(row["name"], row["input"])
    if (len(selected) != MAX_CASES or min(years.values()) < 8 or
            len(acronyms) < 16):
        raise ValueError(f"too few historical acronym passages: {len(selected)}, "
                         f"{dict(years)}, {dict(acronyms)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION_PATH, LONG_PACKET, *active_paths)}
    inventory_hash = digest(json.dumps(inventory, separators=(",", ":")).encode())
    return selected, months, acronyms, inputs, inventory_hash


def main():
    """Pin an unlabeled acronym packet for two independent reviews."""
    rows, months, acronyms, inputs, inventory_hash = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "federal_register_us_historical_acronym_blind_v1",
        "training_eligible": False,
        "label_status": "unlabeled",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "months": dict(sorted(months.items())),
        "acronyms": dict(sorted(acronyms.items())),
        "raw_inventory_sha256": inventory_hash,
        "input_sha256": inputs,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical acronym packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical acronym packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical acronym packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} historical acronym passages frozen from {len(acronyms)} acronyms")
    return rows


if __name__ == "__main__":
    main()
