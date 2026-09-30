"""Combine every frozen US evaluation case for future generator exclusions."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
BASE = REVIEW / "us-eval-exclusions-v1.jsonl"
CONTACT = REVIEW / "us-2025-contact-holdout-v2.jsonl"
YEAR = REVIEW / "us-2025-year-contact-holdout-v2.jsonl"
STRICT = REVIEW / "us-2026-contact-strict-holdout-v2.jsonl"
OUT = REVIEW / "us-eval-exclusions-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def combined_rows():
    """Check the newer holdout and unique case membership before concatenating."""
    strict_manifest = json.loads(STRICT.with_suffix(".manifest.json").read_text())
    if (strict_manifest["sha256"] != digest(STRICT.read_bytes()) or
            strict_manifest["cases"] != 28 or
            not strict_manifest["strict_entity_holdout"] or
            strict_manifest["training_eligible"]):
        raise ValueError("strict US holdout changed")
    sources = (BASE, CONTACT, YEAR, STRICT)
    rows = [json.loads(line) for path in sources for line in path.read_text().splitlines()]
    if (len(rows) != 339 or len({row["name"] for row in rows}) != 339 or
            len({digest(row["input"].encode()) for row in rows}) != 339 or
            any(row["country"] != "US" for row in rows)):
        raise ValueError("full US evaluation membership changed")
    return rows, {path.name: digest(path.read_bytes()) for path in sources}


def main():
    """Write a stable exclusion file without changing either evaluation source."""
    rows, source_hashes = combined_rows()
    data = b"".join(path.read_bytes() for path in (BASE, CONTACT, YEAR, STRICT))
    manifest = {
        "kind": "us_full_eval_exclusions_v2",
        "cases": len(rows), "source_sha256": source_hashes,
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("full US exclusion file or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("full US exclusions changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("full US exclusion inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} frozen US evaluation cases in generator exclusions")


if __name__ == "__main__":
    main()
