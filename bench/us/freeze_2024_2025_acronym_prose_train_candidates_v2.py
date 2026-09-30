"""Freeze independently agreed historical US acronym prose."""

import collections
import json

from active_sources import V4_BASE_SILVER
from build_contact_snippets import lines
from build_full_notice_gold import validate_spans
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2026_long_context_train_candidates_v1 import uncertain_names
from screen_2024_2025_acronym_prose_v1 import MANIFEST as PACKET_MANIFEST
from screen_2024_2025_acronym_prose_v1 import OUT as PACKET
from screen_2024_2025_acronym_prose_v1 import RAW, ROOT
from screen_2024_2025_acronym_prose_v1 import main as verify_packet
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key


REVIEW_A = ROOT / "data/interim/review/us-historical-acronym-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-historical-acronym-review-b-v1.jsonl"
UNCERTAIN_A = ROOT / "data/interim/review/us-historical-acronym-review-a-uncertain-v1.json"
UNCERTAIN_B = ROOT / "data/interim/review/us-historical-acronym-review-b-uncertain-v1.json"
OUT = ROOT / "data/interim/silver/r25/us-historical-acronym-candidate-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ACTIVE_PATHS = tuple(ROOT / f"data/interim/silver/{name}.jsonl"
                     for name, _ in V4_BASE_SILVER) + (
    ROOT / "data/interim/silver/us-v4-reviewed-additions-v2.jsonl",
    ROOT / "data/interim/silver/us-v4-reviewed-long-additions-v1.jsonl",
    ROOT / "data/interim/silver/us-v4-reviewed-historical-long-v1.jsonl",
)


def candidate_rows():
    """Keep complete agreement for organization acronyms outside the current split."""
    packet = verify_packet()
    frozen = lines(PACKET)
    metadata = json.loads(PACKET_MANIFEST.read_text())
    if (packet != frozen or len(packet) != 22 or
            metadata["sha256"] != digest(PACKET.read_bytes()) or
            metadata["training_eligible"] is not False or
            metadata["label_status"] != "unlabeled"):
        raise ValueError("historical acronym blind packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names):
        raise ValueError("historical acronym reviews do not cover every passage")
    uncertain = uncertain_names(UNCERTAIN_A) | uncertain_names(UNCERTAIN_B)
    if not uncertain <= set(names):
        raise ValueError("historical acronym uncertainty refers to an unknown row")
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
    active = [row for path in ACTIVE_PATHS for row in lines(path)]
    active_keys = {document_key(row) for row in active}
    disposition = collections.Counter()
    accepted = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        if a["input"] != source["input"] or b["input"] != source["input"]:
            raise ValueError(f"historical acronym review text changed: {name}")
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].encode()[source["source_start_byte"]:
                                     source["source_end_byte"]].decode() != source["input"] or
                source["expansion_evidence"] not in raw["text"] or
                document_key(source) in eval_keys | active_keys):
            raise ValueError(f"historical acronym raw source changed: {name}")
        a_spans = validate_spans(name, source["input"], a["expected"])
        b_spans = validate_spans(name, source["input"], b["expected"])
        if name in uncertain:
            disposition["uncertain"] += 1
            continue
        if a_spans != b_spans:
            disposition["disagree"] += 1
            continue
        if not any(span["kind"] == "org" and span["text"] == source["acronym"]
                   for span in a_spans):
            disposition["missing_target"] += 1
            continue
        sample = [(name, source["input"], a_spans)]
        if collisions(evaluation, sample) or contact_collisions(evaluation, sample):
            disposition["evaluation_overlap"] += 1
            continue
        if shingles(source["input"], 12) & eval_shingles:
            disposition["evaluation_prose_overlap"] += 1
            continue
        accepted.append({
            "name": name, "country": "US", "doc_type": "notice_prose",
            "input": source["input"], "expected": a_spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
        disposition["agreed"] += 1
    if (disposition != {"agreed": 14, "evaluation_overlap": 5,
                        "evaluation_prose_overlap": 1, "uncertain": 2} or
            len({document_key(row) for row in accepted}) != len(accepted)):
        raise ValueError(f"historical acronym disposition changed: {dict(disposition)}")
    return accepted, disposition


def main():
    """Pin the agreed acronym prose without activating it."""
    rows, disposition = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET_MANIFEST, REVIEW_A, REVIEW_B,
              UNCERTAIN_A, UNCERTAIN_B, EVALUATION_PATH, *ACTIVE_PATHS)
    manifest = {
        "kind": "us_historical_acronym_candidate_v2",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "disposition": dict(sorted(disposition.items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical acronym candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical acronym candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical acronym candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} historical acronym passages passed strict split checks")


if __name__ == "__main__":
    main()
