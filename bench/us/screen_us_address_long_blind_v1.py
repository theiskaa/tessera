"""Freeze source-pinned long US address passages for blind review."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from address_keys import address_keys_in_text
from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION, ROOT, proposed_sources, rows
from screen_us_official_mail_code_packet_v1 import visible_text
from silver_dedupe import NearTextIndex
from source_identity import document_key


BASE = ROOT / "data/raw/candidates/us-address-long-v1"
INDEX = BASE / "capture-index.json"
OUT = BASE / "blind-v1.jsonl"
MANIFEST = BASE / "blind-v1.manifest.json"
FR_WINDOWS = {
    "2024-27385": (0, 2),
    "2024-29116": (0, 3),
    "2024-29931": (0, 1),
    "2025-01941": (0, 4),
    "2025-02202": (0, 2),
    "2025-02285": (5, 11),
    "2025-02929": (0, 4),
    "2025-03259": (0, 5),
    "2025-07095": (0, 2),
    "2025-18609": (1, 4),
    "2025-21350": (5, 7),
    "2025-22823": (0, 4),
    "2025-22958": (0, 0),
    "2025-23779": (0, 4),
    "2026-01791": (15, 16),
    "2026-02204": (2, 3),
    "2026-04864": (0, 3),
    "2026-08096": (0, 2),
    "2026-10375": (0, 1),
    "2026-12726": (2, 6),
    "2026-16328": (0, 1),
}
WEB_WINDOWS = {"fws-fish-springs": (182, 211), "jpl-buzzanga": (43, 69)}
STATE_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")


def digest(data):
    """Hash the exact captured bytes."""
    return hashlib.sha256(data).hexdigest()


def source_specs():
    """Pin one raw document for each selected passage."""
    specs = []
    for source_id in sorted(FR_WINDOWS):
        folder = ("data/raw/candidates/federal-register-2026-contacts-v1/sources"
                  if source_id.startswith("2026-") else "data/raw/silver/federal-register")
        path = ROOT / folder / f"{source_id}.json"
        raw = json.loads(path.read_bytes())
        specs.append({"source_id": source_id, "source_url": raw["url"],
                      "raw_path": str(path.relative_to(ROOT)),
                      "raw_sha256": digest(path.read_bytes()), "bytes": path.stat().st_size,
                      "kind": "federal-register"})
    for index_name in ("capture-index.json", "capture-jpl-index.json"):
        capture = json.loads((ROOT / "data/raw/candidates/us-address-long-official-v1" /
                              index_name).read_text())
        for source in capture["sources"]:
            if source["source_id"] in WEB_WINDOWS:
                specs.append({**source, "kind": "official-page"})
    if {s["source_id"] for s in specs} != set(FR_WINDOWS) | set(WEB_WINDOWS):
        raise ValueError("address source membership changed")
    return sorted(specs, key=lambda value: value["source_id"])


def passage(spec, raw):
    """Extract a complete source paragraph or visible-page line window."""
    source_id = spec["source_id"]
    if spec["kind"] == "federal-register":
        source = json.loads(raw)
        if source["id"] != source_id or source["url"] != spec["source_url"]:
            raise ValueError(f"Federal Register source changed: {source_id}")
        full = source["text"]
        units = full.split("\n\n")
        start, end = FR_WINDOWS[source_id]
        excerpt = "\n\n".join(units[start:end + 1])
        begin = len("\n\n".join(units[:start]).encode()) + (2 if start else 0)
        method = "federal_register_text_paragraphs_v1"
    else:
        full = visible_text(raw)
        units = full.splitlines()
        start, stop = WEB_WINDOWS[source_id]
        excerpt = "\n".join(units[start:stop])
        begin = len("\n".join(units[:start]).encode()) + (1 if start else 0)
        end = stop - 1
        method = "official_page_visible_lines_v1"
    finish = begin + len(excerpt.encode())
    if full.encode()[begin:finish].decode() != excerpt:
        raise ValueError(f"address excerpt offset changed: {source_id}")
    complete_end = (spec["kind"] == "official-page" or
                    excerpt.rstrip().endswith((".", "!", "?")))
    if (not 200 <= len(excerpt.split()) <= 500 or "\ufffd" in excerpt or
            not address_keys_in_text(excerpt) or not STATE_ZIP.search(excerpt) or
            not complete_end):
        raise ValueError(f"invalid complete address passage: {source_id}")
    return excerpt, {"method": method, "start": start, "end": end,
                     "text_byte_start": begin, "text_byte_end": finish}


def selected_rows(specs):
    """Screen against frozen evaluation and the proposed silver baseline."""
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text())
        if frozen["kind"] != "us_address_long_blind_v1":
            raise ValueError("address manifest kind changed")
        silver_paths = [ROOT / name for name in frozen["input_sha256"]
                        if name.startswith("data/interim/silver/")]
        silver = [row for path in silver_paths for row in rows(path)]
    else:
        silver_paths = proposed_sources()
        silver = [row for path in silver_paths for row in rows(path)]
    if len(silver) != 1777 or len(silver_paths) != 30:
        raise ValueError("address silver baseline changed")
    gold = [row for path in EVALUATION for row in rows(path)]
    if len(gold) != 371:
        raise ValueError("reserved evaluation changed")
    silver_documents = {document_key(row) for row in silver}
    silver_urls = {row.get("source_url", "").rstrip("/") for row in silver}
    near_silver = NearTextIndex()
    for row in silver:
        near_silver.add(row["id"], row["text"])
    near_packet = NearTextIndex()
    selected = []
    physical = set()
    for spec in specs:
        raw = (ROOT / spec["raw_path"]).read_bytes()
        if len(raw) != spec["bytes"] or digest(raw) != spec["raw_sha256"]:
            raise ValueError(f"raw address source changed: {spec['source_id']}")
        excerpt, locator = passage(spec, raw)
        keys = address_keys_in_text(excerpt)
        row = {"name": f"us-address-long-v1-{spec['source_id']}", "country": "US",
               "input": excerpt, "source": spec["kind"],
               "source_id": spec["source_id"], "source_url": spec["source_url"],
               "source_group": f"address-long-v1:{spec['source_id']}",
               "source_document_key": document_key(spec),
               "source_file": spec["raw_path"], "raw_sha256": spec["raw_sha256"],
               "source_window": locator, "input_sha256": digest(excerpt.encode()),
               "words": len(excerpt.split()), "training_eligible": False,
               "evaluation_eligible": False}
        if document_key(row) in silver_documents or row["source_url"].rstrip("/") in silver_urls:
            raise ValueError(f"address source overlaps silver: {row['name']}")
        if near_silver.prior(excerpt) or near_packet.prior(excerpt):
            raise ValueError(f"address passage near duplicate: {row['name']}")
        if physical & keys:
            raise ValueError(f"address repeated within packet: {row['name']}")
        physical.update(keys)
        near_packet.add(row["name"], excerpt)
        selected.append(row)
    hits = heldout_reasons(selected, gold)
    if hits:
        raise ValueError(f"reserved evaluation overlap: {hits}")
    if len({document_key(row) for row in selected}) != len(selected):
        raise ValueError("address packet repeats a source document")
    return selected, silver_paths, len(silver)


def main():
    """Freeze and replay a strictly screened unlabeled address packet."""
    BASE.mkdir(parents=True, exist_ok=True)
    specs = source_specs()
    index = {"kind": "us_address_long_source_capture_v1", "training_eligible": False,
             "evaluation_eligible": False, "sources": specs}
    index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    if INDEX.exists() and INDEX.read_bytes() != index_bytes:
        raise ValueError("frozen address capture index changed")
    selected, silver_paths, silver_count = selected_rows(specs)
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in selected).encode()
    inputs = [ROOT / spec["raw_path"] for spec in specs]
    inputs.extend([*EVALUATION, *silver_paths,
                   ROOT / "bench/us/check_us_blind_packet_holdout.py",
                   ROOT / "bench/us/address_keys.py",
                   ROOT / "bench/us/silver_dedupe.py"])
    manifest = {"kind": "us_address_long_blind_v1", "label_status": "unlabeled",
                "training_eligible": False, "evaluation_eligible": False,
                "cases": len(selected), "source_documents": len(selected),
                "source_domains": len({urlsplit(row["source_url"]).netloc for row in selected}),
                "reserved_evaluation_cases": 371, "proposed_real_silver_cases": silver_count,
                "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                 for path in inputs}, "packet_sha256": digest(packet)}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("address packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != packet:
        raise ValueError("frozen address packet changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != manifest_bytes:
        raise ValueError("address screening inputs changed")
    if not OUT.exists():
        INDEX.write_bytes(index_bytes)
        OUT.write_bytes(packet)
        MANIFEST.write_bytes(manifest_bytes)
    print(f"{len(selected)} unlabeled long address passages; packet SHA-256 {digest(packet)}")


if __name__ == "__main__":
    main()
