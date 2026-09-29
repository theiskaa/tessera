"""Combine frozen US evaluation sets into the training exclusion input."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "data/interim/review/us-dev-office-exclusions-v1.jsonl",
    ROOT / "data/interim/review/us-staff-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-park-address-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-ky-superintendent-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-ca-superintendent-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-or-district-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-pa-room-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-pa-acronym-challenge-v1.jsonl",
    ROOT / "data/interim/review/us-pa-agriculture-staff-challenge-v1.jsonl",
)
OUTPUT = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
MANIFEST = ROOT / "data/interim/review/us-eval-exclusions-v1.manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    names = set()
    texts = set()
    pieces = []
    counts = {}
    for path in SOURCES:
        data = path.read_bytes()
        count = 0
        for line in data.splitlines():
            row = json.loads(line)
            name = row["name"]
            text = row["input"].encode()
            if row["country"] != "US" or name in names or digest(text) in texts:
                raise ValueError(f"duplicate or non-US evaluation: {name}")
            names.add(name)
            texts.add(digest(text))
            for span in row["expected"]:
                if text[span["start"]:span["end"]].decode() != span["text"]:
                    raise ValueError(f"bad evaluation label: {name}")
            count += 1
        counts[path.name] = {"cases": count, "sha256": digest(data)}
        pieces.append(data)
    if not all(piece.endswith(b"\n") for piece in pieces):
        raise ValueError("evaluation source missing final newline")
    output = b"".join(pieces)
    OUTPUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "us_training_evaluation_exclusions",
        "sources": counts,
        "cases": len(names),
        "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(names)} evaluation cases excluded from training")


if __name__ == "__main__":
    main()
