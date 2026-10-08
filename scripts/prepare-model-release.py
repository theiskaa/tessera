#!/usr/bin/env python3
"""Prepare an allowlisted local model upload folder without publishing it."""

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path


def model_card(readme: str) -> str:
    asset_paths = {
        "models/tessera-v1.safetensors": "tessera-v1.safetensors",
        "models/tessera-v1.sha256": "tessera-v1.sha256",
        "models/bundle.json": "bundle.json",
        "NOTICE": "NOTICE",
    }

    def link(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith(("https://", "http://", "#")):
            return match.group(0)
        target = asset_paths.get(
            target, f"https://github.com/theiskaa/tessera/blob/main/{target}"
        )
        return f"]({target})"

    body = re.sub(r"\]\(([^)]+)\)", link, readme)
    metadata = """---
language:
- en
license: cc-by-4.0
pipeline_tag: token-classification
library_name: tessera
tags:
- tessera
- contact-extraction
- named-entity-recognition
- wasm
- experimental
---

"""
    return metadata + body


def prepare(out: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    model = root / "models/tessera-v1.safetensors"
    checksum_file = root / "models/tessera-v1.sha256"
    checksum = checksum_file.read_text().strip()
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if checksum != f"sha256-{actual}":
        raise ValueError("model checksum does not match the distributed checksum")
    with model.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        if length > 10_000_000 or length > model.stat().st_size - 8:
            raise ValueError("invalid Safetensors header length")
        metadata = json.loads(source.read(length))["__metadata__"]
    required = {
        "format": "2",
        "status": "experimental",
        "license": "CC-BY-4.0",
        "model_version": "0.4.0",
        "runtime_version": "0.2.0",
        "detector_training_updates": "6160",
        "detector_training_complete": "true",
        "learning_gate_passed": "true",
        "detector_input_policy": "known_us",
        "general_accuracy_claim": "false",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"release metadata differs: {key}")
    copies = {
        "tessera-v1.safetensors": model,
        "tessera-v1.sha256": checksum_file,
        "NOTICE": root / "NOTICE",
    }
    readme = model_card((root / "README.md").read_text())
    for source in copies.values():
        if not source.is_file():
            raise ValueError(f"missing release input: {source.name}")
    out.mkdir(parents=True, exist_ok=False)
    for name, source in copies.items():
        shutil.copyfile(source, out / name)
    (out / "README.md").write_text(readme)
    descriptor = {
        "runtime": "tessera",
        "file": model.name,
        "bytes": model.stat().st_size,
        "sha256": actual,
        "metadata": metadata,
    }
    (out / "bundle.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(f"Prepared {len(copies) + 2} files in {out}; nothing uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
