"""Correct audited definite-description labels while preserving the original silver."""

import hashlib
import json
from pathlib import Path

from check_us_next_data_split import ROOT, rows, verify_source_manifest


SOURCE = ROOT / "data/interim/silver/us-v4-reviewed-additions-strict-v1.jsonl"
SOURCE_SHA256 = "a7d2a4cf96ddb652e3b01b91019fdd0610f2681665994271bb4ec6b0f2c0a712"
OUT = ROOT / "data/interim/silver/us-v4-reviewed-additions-strict-v2.jsonl"
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
CORRECTIONS = {
    "us-nonorganization-prose-2026-07713-174": {
        "text_sha256": "b404d92fb824f2af31a2ab591bd251b683a846fd5fecf4c5a5f204460170b5cd",
        "remove": [{"kind": "org", "start": 53, "end": 63, "text": "Commission"}],
    },
    "us-nonorganization-prose-2026-16607-27851": {
        "text_sha256": "363754120f369e00e91e871dc922ac5b8b1f613ba403362a92e62e517a895a55",
        "remove": [{"kind": "org", "start": 4, "end": 14, "text": "Department"},
                   {"kind": "org", "start": 83, "end": 93, "text": "Department"}],
    },
}


def digest(path):
    """Hash exact provenance bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Keep two genuine negative passages and remove their three incorrect labels."""
    verify_source_manifest(SOURCE, require_eligible=True)
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("generic organization correction source changed")
    revised = rows(SOURCE)
    found = set()
    for row in revised:
        correction = CORRECTIONS.get(row["id"])
        if correction is None:
            continue
        if row["id"] in found:
            raise ValueError("generic organization correction source repeats a row")
        found.add(row["id"])
        raw = row["text"].encode()
        if hashlib.sha256(raw).hexdigest() != correction["text_sha256"]:
            raise ValueError("generic organization correction text changed")
        expected = []
        for span in correction["remove"]:
            if raw[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError("generic organization correction boundary changed")
            expected.append({key: span[key] for key in ("kind", "start", "end")})
        if row["entities"] != expected:
            raise ValueError("generic organization correction annotations changed")
        row["entities"] = []
    if found != CORRECTIONS.keys() or len(revised) != 106:
        raise ValueError("generic organization correction membership changed")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in revised).encode()
    inputs = (SOURCE, SOURCE.with_suffix(".manifest.json"), POLICY, Path(__file__))
    manifest = {
        "kind": "us_strict_referent_safe_silver_policy_v2",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(revised),
        "corrections": CORRECTIONS,
        "reason": "Definite descriptions the Commission and the Department are not proper organization names under the existing annotation guidelines.",
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = OUT.with_suffix(".manifest.json")
    if OUT.exists() != manifest_path.exists():
        raise ValueError("corrected silver or manifest is missing")
    if OUT.exists() and (OUT.read_bytes() != data or manifest_path.read_bytes() != encoded):
        raise ValueError("frozen generic organization correction changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        manifest_path.write_bytes(encoded)
    print("106 passages preserved; 3 incorrect organization labels removed from 2 negative passages")


if __name__ == "__main__":
    main()
