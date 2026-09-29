"""Recapture clipped 2024 notices separately from their frozen reviewed excerpts."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from fetch_federal_register import extract, get, page_text


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "data/raw/silver/federal-register"
OUT = ROOT / "data/raw/silver/federal-register-full-2024"
STRICT_MANIFEST = ROOT / "data/interim/silver/us-reviewed-strict-v1.manifest.json"
MARKER = "SUPPLEMENTARY INFORMATION:\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recapture(source_id, out=OUT):
    old_path = OLD / f"{source_id}.json"
    old_bytes = old_path.read_bytes()
    old = json.loads(old_bytes)
    if (old["id"] != source_id or MARKER not in old["text"] or
            len(old["text"].split(MARKER, 1)[1]) != 2000):
        raise ValueError(f"source is not a frozen legacy cutoff: {source_id}")
    if not old["text_url"].startswith("https://www.govinfo.gov/content/pkg/FR-"):
        raise ValueError(f"unexpected official source URL: {source_id}")
    path = out / f"{source_id}.json"
    if path.exists():
        full_bytes = path.read_bytes()
        full = json.loads(full_bytes)
    else:
        full = dict(old)
        full["text"] = extract(page_text(get(old["text_url"]).decode("utf-8")))
        full_bytes = (json.dumps(full, ensure_ascii=False) + "\n").encode()
    if (full["id"] != old["id"] or full["url"] != old["url"] or
            full["text_url"] != old["text_url"] or
            len(full["text"]) <= len(old["text"]) or
            not full["text"].startswith(old["text"])):
        raise ValueError(f"recaptured source differs before the old cutoff: {source_id}")
    if not path.exists():
        write_new(path, full_bytes)
    return {"source_id": source_id, "old_sha256": digest(old_bytes),
            "full_sha256": digest(full_bytes), "old_chars": len(old["text"]),
            "full_chars": len(full["text"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest_bytes = STRICT_MANIFEST.read_bytes()
    ids = json.loads(manifest_bytes)["legacy_cutoff_excluded_ids"]
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    selected = ids[:args.limit] if args.limit else ids
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [recapture(source_id) for source_id in selected]
    manifest = (json.dumps({
        "kind": "full_2024_recapture",
        "strict_manifest_sha256": digest(manifest_bytes),
        "sources": sources,
    }, indent=2, sort_keys=True) + "\n").encode()
    write_new(OUT / "manifest.json", manifest)
    print(f"recaptured {len(sources)}/{len(ids)} clipped sources in {OUT}")


if __name__ == "__main__":
    main()
