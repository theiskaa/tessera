"""Freeze the citation-sensitive acronym row resolved by a third review."""

import collections
import json

from build_contact_snippets import lines
from build_full_notice_gold import validate_spans
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2026_long_context_train_candidates_v1 import uncertain_names
from screen_2024_2025_acronym_prose_v1 import OUT as PACKET
from screen_2024_2025_acronym_prose_v1 import RAW, ROOT
from source_identity import document_key


REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-historical-acronym-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-historical-acronym-review-b-v1.jsonl"
UNCERTAIN_A = REVIEW / "us-historical-acronym-review-a-uncertain-v1.json"
UNCERTAIN_B = REVIEW / "us-historical-acronym-review-b-uncertain-v1.json"
THIRD_BLIND = REVIEW / "us-historical-acronym-disputed-blind-v1.jsonl"
THIRD_REVIEW = REVIEW / "us-historical-acronym-third-review-v1.jsonl"
THIRD_UNCERTAIN = REVIEW / "us-historical-acronym-third-review-uncertain-v1.json"
APPROVED = "us-historical-acronym-2025-22898-2557"
OUT = ROOT / "data/interim/silver/r25/us-historical-acronym-third-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def candidate_rows():
    """Keep the row whose citation boundaries match two independent passes."""
    packet = {row["name"]: row for row in lines(PACKET)}
    left = {row["name"]: row for row in lines(REVIEW_A)}
    right = {row["name"]: row for row in lines(REVIEW_B)}
    blind, third = lines(THIRD_BLIND), lines(THIRD_REVIEW)
    if (len(blind) != 2 or len(third) != 2 or
            [row["name"] for row in blind] != [row["name"] for row in third] or
            set(packet) != set(left) or set(packet) != set(right)):
        raise ValueError("historical acronym third review coverage changed")
    a_uncertain = uncertain_names(UNCERTAIN_A)
    b_uncertain = uncertain_names(UNCERTAIN_B)
    c_uncertain = uncertain_names(THIRD_UNCERTAIN)
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    accepted = []
    for blind_row, review in zip(blind, third):
        name = review["name"]
        source = packet[name]
        if (blind_row["input"] != source["input"] or
                review["input"] != source["input"]):
            raise ValueError(f"historical acronym third-review text changed: {name}")
        spans = validate_spans(name, source["input"], review["expected"])
        if name != APPROVED:
            continue
        first = validate_spans(name, source["input"], left[name]["expected"])
        second = validate_spans(name, source["input"], right[name]["expected"])
        if (name not in a_uncertain or name not in b_uncertain or
                name in c_uncertain or spans != second or spans == first or
                not any(span["kind"] == "org" and span["text"] == source["acronym"]
                        for span in spans)):
            raise ValueError("historical acronym citation decision changed")
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].encode()[source["source_start_byte"]:
                                     source["source_end_byte"]].decode() != source["input"] or
                source["expansion_evidence"] not in raw["text"] or
                document_key(source) in eval_keys or
                collisions(evaluation, [(name, source["input"], spans)]) or
                contact_collisions(evaluation, [(name, source["input"], spans)])):
            raise ValueError("historical acronym third source failed strict checks")
        accepted.append({
            "name": name, "country": "US", "doc_type": "notice_prose",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
    if len(accepted) != 1 or accepted[0]["name"] != APPROVED:
        raise ValueError("historical acronym third-review accepted set changed")
    return accepted


def main():
    """Pin the resolved acronym passage without activating it."""
    rows = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET.with_suffix(".manifest.json"), REVIEW_A,
              REVIEW_B, UNCERTAIN_A, UNCERTAIN_B, THIRD_BLIND,
              THIRD_REVIEW, THIRD_UNCERTAIN, EVALUATION_PATH)
    manifest = {
        "kind": "us_historical_acronym_third_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "decision": "Exclude surnames in a bibliographic citation; retain AE and the named authors.",
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical acronym third candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical acronym third candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical acronym third candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("1 citation-sensitive acronym passage resolved after a third review")


if __name__ == "__main__":
    main()
