"""Freeze clean full-name replacements for the long US person blind packet."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import CONFIG, EVALUATION, ROOT, rows
from screen_2026_contact_expansion_v3 import shingles
from screen_us_person_long_blind_v2 import ArticleBlocks, digest, proposed_real_silver
from source_identity import document_key


CAPTURE = ROOT / "data/raw/candidates/us-person-long-v3"
INDEX = CAPTURE / "capture-index.json"
OUT = CAPTURE / "blind-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PREVIOUS = ROOT / "data/raw/candidates/us-person-long-v2/blind-v2.jsonl"
WINDOWS = {
    "berkeley-keltner": (4, 11, "Dacher Keltner"),
    "berkeley-puthussery": (5, 10, "Teresa Puthussery"),
    "maryland-clean": (5, 10, "Michael Brown"),
    "uchicago-reinitz": (2, 7, "John Bertram Reinitz"),
    "uchicago-rice": (3, 7, "Stuart Alan Rice"),
}
REPLACED_V2 = (
    "us-person-long-v2-kingcounty-taylor",
    "us-person-long-v2-maryland-distinguished-professors",
    "us-person-long-v2-nasa-ehsan-gharib-nezhad",
    "us-person-long-v2-psu-early-career",
)


def passage(source_id, raw):
    """Extract a contiguous article-block window with a complete person name."""
    parser = ArticleBlocks()
    parser.feed(raw.decode("utf-8"))
    first_h1 = next(index for index, (tag, _) in enumerate(parser.blocks)
                    if tag == "h1")
    start, end, full_name = WINDOWS[source_id]
    selected = parser.blocks[first_h1 + start:first_h1 + end + 1]
    if len(selected) != end - start + 1:
        raise ValueError(f"person replacement article window changed: {source_id}")
    text = "\n\n".join(value for _, value in selected)
    if (full_name not in text or not 200 <= len(text.split()) <= 500 or
            text[-1] not in '.!?…”' or "\ufffd" in text):
        raise ValueError(f"person replacement passage invalid: {source_id}")
    return text, {"method": "heading_paragraph_blocks_v1",
                  "start": start, "end": end}


def selected_rows():
    """Check every replacement against reserved evaluation and proposed real silver."""
    capture = json.loads(INDEX.read_text())
    if capture["kind"] != "us_person_long_v3_official_page_capture":
        raise ValueError("person replacement source capture kind changed")
    sources = {source["source_id"]: source for source in capture["sources"]}
    if set(sources) != set(WINDOWS):
        raise ValueError("person replacement source membership changed")
    selected = []
    for source_id in sorted(sources):
        source = sources[source_id]
        path = ROOT / source["raw_path"]
        raw = path.read_bytes()
        if digest(raw) != source["raw_sha256"] or len(raw) != source["bytes"]:
            raise ValueError(f"person replacement raw source changed: {source_id}")
        body, locator = passage(source_id, raw)
        selected.append({
            "name": f"us-person-long-v3-{source_id}",
            "country": "US", "input": body,
            "source": "official-page", "source_id": source_id,
            "source_url": source["source_url"],
            "source_group": f"official-page-v3:{source_id}",
            "source_document_key": f"url:{source['source_url']}",
            "source_file": source["raw_path"],
            "raw_sha256": source["raw_sha256"],
            "source_window": locator,
            "input_sha256": digest(body.encode()),
            "words": len(body.split()),
        })
    previous = rows(PREVIOUS)
    if ({row["name"] for row in previous} & set(REPLACED_V2) !=
            set(REPLACED_V2) or
            {row["input"] for row in selected} &
            {row["input"] for row in previous}):
        raise ValueError("person replacement copied or lost a v2 passage")
    gold = [row for path in EVALUATION for row in rows(path)]
    if len(gold) != 371:
        raise ValueError("reserved evaluation membership changed")
    hits = heldout_reasons(selected, gold)
    if hits:
        raise ValueError(f"person replacements overlap reserved evaluation: {hits}")
    silver_paths, silver = proposed_real_silver()
    silver_keys = {document_key(row) for row in silver}
    silver_urls = {row.get("source_url", "").rstrip("/") for row in silver}
    silver_fragments = set().union(*(shingles(row["text"], 12) for row in silver))
    for row in selected:
        if (document_key(row) in silver_keys or
                row["source_url"].rstrip("/") in silver_urls or
                shingles(row["input"], 12) & silver_fragments):
            raise ValueError(f"person replacement overlaps proposed silver: {row['name']}")
    keys = [document_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("person replacement repeats a source document")
    return selected, silver_paths, len(silver)


def main():
    """Pin a separate unlabeled packet and its screening inputs."""
    selected, silver_paths, silver_count = selected_rows()
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in selected).encode()
    inputs = (INDEX, PREVIOUS, CONFIG,
              ROOT / "bench/us/check_us_blind_packet_holdout.py",
              ROOT / "bench/us/screen_us_person_long_blind_v2.py",
              *EVALUATION, *silver_paths)
    manifest = {
        "kind": "us_person_long_blind_v3",
        "label_status": "unlabeled",
        "training_eligible": False,
        "evaluation_eligible": False,
        "cases": len(selected),
        "source_documents": len(selected),
        "source_organizations": len({urlsplit(row["source_url"]).netloc
                                     for row in selected}),
        "reserved_evaluation_cases": len([row for path in EVALUATION
                                          for row in rows(path)]),
        "proposed_real_silver_cases": silver_count,
        "replaces_weak_v2": list(REPLACED_V2),
        "screening": "strict person/org/address/contact/source/12-word eval overlap; "
                     "source and 12-word proposed real silver overlap",
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(packet),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("person replacement packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != packet:
        raise ValueError("frozen person replacement packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("person replacement screening inputs changed")
    if not OUT.exists():
        OUT.write_bytes(packet)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind full-name person replacements from "
          f"{manifest['source_organizations']} official organizations")


if __name__ == "__main__":
    main()
