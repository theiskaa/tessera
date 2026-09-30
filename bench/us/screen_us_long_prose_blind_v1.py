"""Freeze source-pinned Federal Register prose for independent US annotation."""

import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION, ROOT, proposed_sources, rows
from silver_dedupe import NearTextIndex
from source_identity import document_key, source_url_key


BASE = ROOT / "data/raw/candidates/us-long-prose-v1"
OUT = BASE / "blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
OTHER_PACKETS = (
    ROOT / "data/raw/candidates/us-address-long-v1/blind-v1.jsonl",
    ROOT / "data/raw/candidates/us-address-long-v2/blind-v2.jsonl",
)
SOURCES = (
    ("2025-14191", "data/raw/silver/federal-register/2025-14191.json", 7),
    ("2025-22883", "data/raw/silver/federal-register/2025-22883.json", 3),
    ("2026-09372", "data/raw/candidates/federal-register-2026-contacts-v1/sources/2026-09372.json", 7),
)


def digest(path):
    """Hash the exact bytes used for screening and extraction."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows():
    """Extract whole paragraphs by stable ordinal from captured source text."""
    selected = []
    raw_paths = []
    for source_id, relative_path, ordinal in SOURCES:
        path = ROOT / relative_path
        raw = json.loads(path.read_bytes())
        if raw["id"] != source_id:
            raise ValueError(f"Federal Register source identifier changed: {source_id}")
        paragraphs = raw["text"].split("\n\n")
        if ordinal >= len(paragraphs):
            raise ValueError(f"Federal Register paragraph missing: {source_id}")
        body = paragraphs[ordinal]
        words = len(body.split())
        if (not 200 <= words <= 500 or "\ufffd" in body or
                not body[-1] in ".!?"):
            raise ValueError(f"Federal Register narrative paragraph invalid: {source_id}")
        start = len("\n\n".join(paragraphs[:ordinal]).encode()) + (2 if ordinal else 0)
        end = start + len(body.encode())
        if raw["text"].encode()[start:end].decode() != body:
            raise ValueError(f"Federal Register paragraph byte offsets changed: {source_id}")
        selected.append({
            "name": f"us-long-prose-v1-{source_id}", "country": "US", "input": body,
            "source": "federal-register", "source_id": source_id,
            "source_url": raw["url"], "source_group": f"fr:{source_id}",
            "source_document_key": f"fr:{source_id}", "source_file": relative_path,
            "raw_sha256": digest(path), "words": words,
            "source_window": {"method": "federal_register_text_paragraph_v1",
                              "paragraph": ordinal, "text_byte_start": start,
                              "text_byte_end": end},
            "input_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "training_eligible": False, "evaluation_eligible": False,
        })
        raw_paths.append(path)
    return selected, raw_paths


def screen(selected):
    """Screen against the frozen baseline, including source and near text reuse."""
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text())
        silver_paths = [ROOT / name for name in frozen["input_sha256"]
                        if name.startswith("data/interim/silver/") and name.endswith(".jsonl")]
    else:
        silver_paths = proposed_sources()
    silver = [row for path in silver_paths for row in rows(path)]
    gold = [row for path in EVALUATION for row in rows(path)]
    if len(silver_paths) != 33 or len(silver) != 1802 or len(gold) != 371:
        raise ValueError("long prose screening baseline changed")
    comparison = [*silver, *(row for path in OTHER_PACKETS for row in rows(path))]
    keys = {document_key(row) for row in comparison}
    urls = {source_url_key(row) for row in comparison} - {None}
    near = NearTextIndex()
    for index, row in enumerate(comparison):
        near.add(row.get("id", row.get("name", str(index))), row.get("text", row.get("input")))
    for row in selected:
        key, url = document_key(row), source_url_key(row)
        if key is None or url is None or key in keys or url in urls:
            raise ValueError(f"long prose source repeats an existing document: {row['name']}")
        keys.add(key)
        urls.add(url)
        if near.prior(row["input"]):
            raise ValueError(f"long prose passage near duplicate: {row['name']}")
        near.add(row["name"], row["input"])
    hits = heldout_reasons(selected, gold)
    if hits:
        raise ValueError(f"long prose passage overlaps evaluation: {hits}")
    return silver_paths


def main():
    """Write and replay the unlabeled packet against its pinned screening inputs."""
    selected, raw_paths = source_rows()
    silver_paths = screen(selected)
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in selected).encode()
    inputs = (POLICY, *raw_paths, *OTHER_PACKETS, *silver_paths, *EVALUATION,
              Path(__file__), ROOT / "bench/us/check_us_blind_packet_holdout.py",
              ROOT / "bench/us/source_identity.py", ROOT / "bench/us/silver_dedupe.py")
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    manifest = {
        "kind": "us_long_prose_blind_v1", "label_status": "unlabeled",
        "training_eligible": False, "evaluation_eligible": False,
        "cases": len(selected), "source_documents": len(selected),
        "reserved_evaluation_cases": len(gold := [row for path in EVALUATION for row in rows(path)]),
        "proposed_real_silver_cases": 1802,
        "annotation_policy": str(POLICY.relative_to(ROOT)),
        "input_sha256": hashes, "packet_sha256": hashlib.sha256(packet).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long prose packet or manifest missing")
    if OUT.exists() and (OUT.read_bytes() != packet or MANIFEST.read_bytes() != encoded):
        raise ValueError("frozen long prose packet or screening inputs changed")
    BASE.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(packet)
        MANIFEST.write_bytes(encoded)
    print(f"{len(selected)} Federal Register prose passages frozen for blind review")


if __name__ == "__main__":
    main()
