#!/usr/bin/env python3
"""Prepare an allowlisted local model upload folder without publishing it."""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


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
        "detector_training_updates": "4000",
        "detector_training_complete": "false",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"release metadata differs: {key}")
    copies = {
        "tessera-v1.safetensors": model,
        "tessera-v1.sha256": checksum_file,
        "README.md": root / "models/README.md",
        "NOTICE": root / "NOTICE",
    }
    for source in copies.values():
        if not source.is_file():
            raise ValueError(f"missing release input: {source.name}")
    out.mkdir(parents=True, exist_ok=False)
    for name, source in copies.items():
        shutil.copyfile(source, out / name)
    descriptor = {
        "runtime": "tessera",
        "file": model.name,
        "bytes": model.stat().st_size,
        "sha256": actual,
        "metadata": metadata,
    }
    (out / "bundle.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(f"Prepared {len(copies) + 1} files in {out}; nothing uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
