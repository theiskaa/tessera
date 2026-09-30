"""Freeze the current reserved US evaluation union for future exclusions."""

import hashlib
import json
from pathlib import Path

from check_us_next_data_split import EVALUATION, ROOT


SOURCES = tuple(ROOT / "data/interim/review" / name for name in (
    "us-eval-exclusions-v4.jsonl",
    "us-long-org-eval-gold-v4.jsonl",
    "us-doe-org-overviews-strict-gold-v1.jsonl",
))
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
OUT = ROOT / "data/interim/review/us-eval-exclusions-v5.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
KINDS = {"person", "org", "address", "email", "phone"}


def digest(path):
    """Hash the exact bytes of a frozen source or policy file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(path):
    """Verify source and lineage hashes before accepting its reviewed rows."""
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    data = path.read_bytes()
    if manifest.get("sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError(f"evaluation source hash changed: {path}")
    if manifest.get("training_eligible") is not False:
        raise ValueError(f"evaluation source became training eligible: {path}")
    for field in ("source_sha256", "input_sha256"):
        for relative, expected in manifest.get(field, {}).items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"evaluation source dependency changed: {relative}")
    if not data.endswith(b"\n"):
        raise ValueError(f"evaluation source lacks a final newline: {path}")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != manifest.get("cases"):
        raise ValueError(f"evaluation source case count changed: {path}")
    return rows, data, manifest_path


def validate_rows(rows):
    """Validate unique case text and exact UTF-8 byte spans across the union."""
    if len(rows) != 371:
        raise ValueError(f"expected 371 reserved evaluation cases; got {len(rows)}")
    names = [row["name"] for row in rows]
    texts = [row["input"] for row in rows]
    if len(set(names)) != len(rows) or len(set(texts)) != len(rows):
        raise ValueError("reserved evaluation has duplicate names or text")
    for row in rows:
        if row["country"] != "US":
            raise ValueError(f"non-US reserved case: {row['name']}")
        source = row["input"].encode()
        ordered = []
        for span in row["expected"]:
            start, end = span["start"], span["end"]
            if (span["kind"] not in KINDS or not 0 <= start < end <= len(source) or
                    source[start:end].decode("utf-8") != span["text"]):
                raise ValueError(f"invalid reserved span: {row['name']}: {span}")
            ordered.append((start, end))
        ordered.sort()
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"overlapping reserved spans: {row['name']}")


def main():
    """Write or replay a byte-preserving, source-pinned exclusion union."""
    if not MANIFEST.exists() and tuple(EVALUATION) != SOURCES:
        raise ValueError("current US evaluation membership differs from the v5 sources")
    grouped = [source_rows(path) for path in SOURCES]
    rows = [row for source, _, _ in grouped for row in source]
    validate_rows(rows)
    data = b"".join(raw for _, raw, _ in grouped)
    source_sha256 = {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}
    inputs = (*[manifest for _, _, manifest in grouped], Path(__file__), POLICY)
    manifest = {
        "kind": "us_next_eval_exclusions_v5",
        "cases": len(rows),
        "training_eligible": False,
        "source_sha256": source_sha256,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("v5 exclusions or manifest missing")
    if OUT.exists() and (OUT.read_bytes() != data or MANIFEST.read_bytes() != encoded):
        raise ValueError("frozen v5 exclusions or screening inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded)
    print(f"{len(rows)} reserved US cases frozen in v5 evaluation exclusions")


if __name__ == "__main__":
    main()
