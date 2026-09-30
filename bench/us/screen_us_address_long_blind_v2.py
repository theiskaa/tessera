"""Freeze source-pinned official US address passages for independent annotation."""

import hashlib
import json
import re
from pathlib import Path

from address_keys import (
    NUMBERED_STREET_IN_TEXT, NUMBER_WORD, STREET_TYPES,
    first_street_word,
)
from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import (
    ADDITIONS, EVALUATION, REPLACEMENTS, ROOT, proposed_sources, rows,
    verify_source_manifest,
)
from screen_us_official_mail_code_packet_v1 import visible_text
from silver_dedupe import NearTextIndex
from source_identity import document_key, source_url_key


CAPTURE = ROOT / "data/raw/candidates/us-address-long-official-v2"
INDEX = CAPTURE / "capture-index.json"
SELECTIONS = tuple(CAPTURE / f"selections-{group}-v1.json"
                   for group in ("university", "government"))
BASE = ROOT / "data/raw/candidates/us-address-long-v2"
OUT = BASE / "blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"


def digest(path):
    """Hash exact captured or reviewed bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def street_keys(value):
    """Conservatively screen numbered streets even when the page omits its ZIP."""
    keys = set()
    matches = [(match[1], match.start(2)) for match in NUMBER_WORD.finditer(value)]
    for match in NUMBERED_STREET_IN_TEXT.finditer(value):
        number, _ = match.group().split(None, 1)
        matches.append((number, match.start() + len(number) + 1))
    for number, start in matches:
        tail = re.split(r"[,;\n]", value[start:start + 90], maxsplit=1)[0]
        parts = re.findall(r"[a-z]+|\d+", tail.casefold())
        word = first_street_word(parts)
        type_index = next((index for index, part in enumerate(parts[:10])
                           if part in STREET_TYPES), None)
        directions = {"to", "onto", "on", "along", "toward", "towards", "past", "via", "from", "then"}
        if word and type_index is not None and not directions.intersection(parts[:type_index]):
            keys.add((number, word))
    return keys


def source_rows():
    """Extract only explicitly selected contiguous windows from captured HTML."""
    capture = json.loads(INDEX.read_text())
    sources = {source["source_id"]: source for source in capture["sources"]}
    if len(sources) != len(capture["sources"]):
        raise ValueError("official capture repeats a source identifier")
    selections = [selection for path in SELECTIONS
                  for selection in json.loads(path.read_text())["selections"]]
    ids = [selection["source_id"] for selection in selections]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("official selection is empty or repeats a source")
    selected = []
    raw_paths = []
    for selection in sorted(selections, key=lambda row: row["source_id"]):
        source_id = selection["source_id"]
        source = sources[source_id]
        path = ROOT / source["raw_path"]
        if digest(path) != source["raw_sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError(f"official HTML changed: {source_id}")
        full = visible_text(path.read_bytes())
        lines = full.splitlines()
        start, stop = selection["start_line"], selection["stop_line"]
        if not 0 <= start < stop <= len(lines):
            raise ValueError(f"invalid official source window: {source_id}")
        text = "\n".join(lines[start:stop])
        words = len(text.split())
        if (words != selection["words"] or not 200 <= words <= 500 or
                "\ufffd" in text or not street_keys(text)):
            raise ValueError(f"invalid long US address passage: {source_id}")
        begin = len("\n".join(lines[:start]).encode()) + (1 if start else 0)
        end = begin + len(text.encode())
        if full.encode()[begin:end].decode() != text:
            raise ValueError(f"official source byte offsets changed: {source_id}")
        selected.append({
            "name": f"us-address-long-v2-{source_id}", "country": "US", "input": text,
            "source": "official-page", "source_id": source_id,
            "source_url": source["source_url"], "source_group": f"address-long-v2:{source_id}",
            "source_document_key": document_key(source), "source_file": source["raw_path"],
            "raw_sha256": source["raw_sha256"], "words": words,
            "source_window": {"method": "official_page_visible_lines_v1", "start": start,
                              "end": stop - 1, "text_byte_start": begin, "text_byte_end": end},
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "training_eligible": False, "evaluation_eligible": False,
        })
        raw_paths.append(path)
    return selected, raw_paths


def screen(selected):
    """Keep future promotion from changing the original screening baseline."""
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text())
        silver_paths = [ROOT / name for name in frozen["input_sha256"]
                        if name.startswith("data/interim/silver/") and name.endswith(".jsonl")]
    else:
        silver_paths = proposed_sources()
    new_names = set(ADDITIONS) | set(REPLACEMENTS.values())
    for path in silver_paths:
        verify_source_manifest(path, require_eligible=path.stem in new_names)
    silver = [row for path in silver_paths for row in rows(path)]
    gold = [row for path in EVALUATION for row in rows(path)]
    if len(silver_paths) != 32 or len(silver) != 1794 or len(gold) != 371:
        raise ValueError("official address screening baseline changed")
    source_keys = {document_key(row) for row in silver}
    source_urls = {source_url_key(row) for row in silver} - {None}
    heldout_streets = set().union(*(street_keys(span["text"])
                                   for row in gold for span in row["expected"]
                                   if span["kind"] == "address"))
    near = NearTextIndex()
    for row in silver:
        near.add(row["id"], row["text"])
    physical = set()
    for row in selected:
        key, url = document_key(row), source_url_key(row)
        if key is None or url is None or key in source_keys or url in source_urls:
            raise ValueError(f"official address source repeats an existing document: {row['name']}")
        source_keys.add(key)
        source_urls.add(url)
        if near.prior(row["input"]):
            raise ValueError(f"official address passage near duplicate: {row['name']}")
        near.add(row["name"], row["input"])
        keys = street_keys(row["input"])
        if keys & heldout_streets:
            raise ValueError(f"official passage shares a held-out street: {row['name']}")
        if physical & keys:
            raise ValueError(f"official address repeated inside new packet: {row['name']}")
        physical.update(keys)
    hits = heldout_reasons(selected, gold)
    if hits:
        raise ValueError(f"official address passage overlaps evaluation: {hits}")
    return silver_paths


def main():
    """Pin an unlabeled packet and its annotation policy without activating training."""
    selected, raw_paths = source_rows()
    silver_paths = screen(selected)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    inputs = (INDEX, *SELECTIONS, *raw_paths, *silver_paths, *EVALUATION, POLICY,
              Path(__file__), ROOT / "bench/us/check_us_blind_packet_holdout.py",
              ROOT / "bench/us/screen_us_official_mail_code_packet_v1.py",
              ROOT / "bench/us/address_keys.py", ROOT / "bench/us/source_identity.py",
              ROOT / "bench/us/silver_dedupe.py")
    manifest = {
        "kind": "us_address_long_blind_v2", "label_status": "unlabeled",
        "training_eligible": False, "evaluation_eligible": False,
        "cases": len(selected), "source_documents": len(selected),
        "reserved_evaluation_cases": 371, "proposed_real_silver_cases": 1794,
        "annotation_policy": str(POLICY.relative_to(ROOT)),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "packet_sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("official address packet or manifest is missing")
    if OUT.exists() and (OUT.read_bytes() != data or MANIFEST.read_bytes() != encoded):
        raise ValueError("frozen official address packet or inputs changed")
    BASE.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded)
    print(f"{len(selected)} official long address passages frozen for blind review")


if __name__ == "__main__":
    main()
