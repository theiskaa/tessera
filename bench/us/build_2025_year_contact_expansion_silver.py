"""Export strictly screened additional 2025 contacts as detector silver."""

import collections
import json
from pathlib import Path

from active_sources import BASE_SILVER
from build_2025_year_contact_expansion_gold import gold_rows
from build_full_notice_gold import digest, read_rows, validate_spans
from build_silver import family
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_2025_year_contact_expansion_train_candidates import (
    MANIFEST as CANDIDATE_MANIFEST, OUT as CANDIDATE,
    SILVER_NAME, candidate_rows,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/silver/federal-register"
SILVER = ROOT / "data/interim/silver"
OUT = SILVER / f"{SILVER_NAME}.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
MODEL_KINDS = {"person", "org", "address"}


def silver_rows():
    manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    candidates = read_rows(CANDIDATE)
    rebuilt = candidate_rows()
    prior_hashes = {name: digest((SILVER / f"{name}.jsonl").read_bytes())
                    for name, _ in BASE_SILVER if name != SILVER_NAME}
    if (manifest["sha256"] != digest(CANDIDATE.read_bytes()) or
            manifest["training_eligible"] or candidates != rebuilt or
            manifest["active_silver_sha256"] != prior_hashes):
        raise ValueError("additional contact candidate split changed")
    gold, _ = gold_rows()
    by_name = {row["name"]: row for row in gold}
    if len(by_name) != len(gold) or any(by_name.get(row["name"]) != row
                                        for row in candidates):
        raise ValueError("additional contact differs from blind reviewed gold")
    raw_hashes = {}
    result = []
    for row in candidates:
        name = row["name"]
        source_id = row["source_id"]
        raw_bytes = (RAW / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (source_id in raw_hashes or raw["id"] != source_id or
                raw["source"] != "federal-register" or
                raw["url"] != row["source_url"] or
                not raw["date"].startswith("2025-") or
                family(raw["url"]) != row["source_group"] or
                raw["text"].count(row["input"]) != 1):
            raise ValueError(f"additional contact source invalid: {name}")
        expected = validate_spans(name, row["input"], row["expected"])
        raw_hashes[source_id] = digest(raw_bytes)
        result.append({"id": name, "source": "federal-register",
                       "source_id": source_id, "source_url": row["source_url"],
                       "source_group": row["source_group"], "country": "US",
                       "text": row["input"],
                       "entities": [{"kind": span["kind"], "start": span["start"],
                                     "end": span["end"]} for span in expected]})
    return result, prior_hashes, raw_hashes


def main():
    rows, prior_hashes, raw_hashes = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["entities"])
    manifest = {
        "kind": "us_2025_year_contact_expansion_silver_v1",
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in EVALUATION},
        "prior_silver_sha256": prior_hashes,
        "raw_sha256": dict(sorted(raw_hashes.items())),
        "documents": len(rows),
        "labels": dict(sorted(counts.items())),
        "model_labels": {kind: counts[kind] for kind in sorted(MODEL_KINDS)},
        "rule_labels": {kind: counts[kind] for kind in ("email", "phone")},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("additional contact silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("additional contact silver changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("additional contact silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} additional reviewed contacts with {dict(sorted(counts.items()))} labels")


if __name__ == "__main__":
    main()
