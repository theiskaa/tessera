"""Export reviewed, evaluation-disjoint 2025 agency prose as detector silver."""

import collections
import json
from pathlib import Path

from active_sources import BASE_SILVER
from build_2025_year_acronym_gold import gold_rows
from build_full_notice_gold import digest, read_rows, validate_spans
from build_silver import family
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_2025_year_acronym_train_candidates import (
    MANIFEST as CANDIDATE_MANIFEST, OUT as CANDIDATE, candidate_rows,
)
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/silver/federal-register"
SILVER = ROOT / "data/interim/silver"
PREVIOUS_OUT = SILVER / "us-2025-year-acronym-prose-v2.jsonl"
OUT = SILVER / "us-2025-year-acronym-prose-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def silver_rows():
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    candidates = read_rows(CANDIDATE)
    if (candidate_manifest["sha256"] != digest(CANDIDATE.read_bytes()) or
            candidate_manifest["training_eligible"] or
            candidates != candidate_rows()):
        raise ValueError("reviewed 2025 acronym candidate split changed")
    gold, _ = gold_rows()
    by_name = {row["name"]: row for row in gold}
    if len(by_name) != len(gold) or any(by_name.get(row["name"]) != row
                                        for row in candidates):
        raise ValueError("acronym candidate differs from blind reviewed gold")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    eval_ids = {row["source_id"] for row in evaluation if row.get("source_id")}
    eval_groups = {family(row["source_url"]) for row in evaluation
                   if row.get("source_url")}
    if collisions(evaluation, ((row["name"], row["input"], row["expected"])
                               for row in candidates)):
        raise ValueError("acronym silver labels overlap evaluation")
    prior = NearTextIndex()
    prior_hashes = {}
    for name, _ in BASE_SILVER:
        if name in {"us-2025-year-acronym-prose-v3",
                    "us-2025-year-contact-expansion-v1"}:
            continue
        path = SILVER / f"{name}.jsonl"
        prior_hashes[name] = digest(path.read_bytes())
        for row in read_rows(path):
            prior.add(row["id"], row["text"])
    ids = set()
    groups = set()
    raw_hashes = {}
    rows = []
    for row in candidates:
        name, source_id = row["name"], row["source_id"]
        raw_bytes = (RAW / f"{source_id}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (source_id in ids or row["source_group"] in groups or
                source_id in eval_ids or row["source_group"] in eval_groups or
                raw["id"] != source_id or raw["source"] != "federal-register" or
                raw["url"] != row["source_url"] or
                not raw["date"].startswith("2025-") or
                family(raw["url"]) != row["source_group"] or
                raw["text"].count(row["input"]) != 1):
            raise ValueError(f"acronym source or family invalid: {name}")
        spans = validate_spans(name, row["input"], row["expected"])
        if any(span["kind"] != "org" for span in spans) or not spans:
            raise ValueError(f"acronym prose has unsupported labels: {name}")
        if prior.prior(row["input"]):
            raise ValueError(f"acronym prose nearly duplicates earlier silver: {name}")
        ids.add(source_id)
        groups.add(row["source_group"])
        raw_hashes[source_id] = digest(raw_bytes)
        prior.add(name, row["input"])
        rows.append({"id": name, "source": "federal-register",
                     "source_id": source_id, "source_url": row["source_url"],
                     "source_group": row["source_group"], "country": "US",
                     "text": row["input"],
                     "entities": [{"kind": span["kind"], "start": span["start"],
                                   "end": span["end"]} for span in spans]})
    return rows, prior_hashes, raw_hashes


def main():
    rows, prior_hashes, raw_hashes = silver_rows()
    previous_manifest = json.loads(PREVIOUS_OUT.with_suffix(".manifest.json").read_text())
    previous_sha = digest(PREVIOUS_OUT.read_bytes())
    if previous_manifest["sha256"] != previous_sha:
        raise ValueError("previous acronym silver changed")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["entities"])
    manifest = {
        "kind": "us_2025_year_acronym_prose_silver_v3",
        "supersedes_sha256": previous_sha,
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "evaluation_sha256": {path.name: digest(path.read_bytes()) for path in EVALUATION},
        "prior_silver_sha256": prior_hashes,
        "raw_sha256": dict(sorted(raw_hashes.items())),
        "documents": len(rows), "labels": dict(sorted(counts.items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("acronym silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("acronym silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("acronym silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed year acronym cases with {counts['org']} organization labels")


if __name__ == "__main__":
    main()
