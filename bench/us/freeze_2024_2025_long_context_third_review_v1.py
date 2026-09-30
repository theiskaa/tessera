"""Freeze resolved historical long passages after three independent reviews."""

import collections
import json

from build_contact_snippets import lines
from build_full_notice_gold import validate_spans
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2025_contact_packet import EMAIL, PHONE
from freeze_2026_long_context_train_candidates_v1 import phone_key, uncertain_names
from screen_2024_2025_long_context_v1 import OUT as PACKET
from screen_2024_2025_long_context_v1 import RAW, ROOT
from source_identity import document_key


REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-long-historical-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-long-historical-review-b-v1.jsonl"
UNCERTAIN_A = REVIEW / "us-long-historical-review-a-uncertain-v1.json"
UNCERTAIN_B = REVIEW / "us-long-historical-review-b-uncertain-v1.json"
THIRD_BLIND = REVIEW / "us-long-historical-disputed-blind-v1.jsonl"
THIRD_REVIEW = REVIEW / "us-long-historical-third-review-v1.jsonl"
THIRD_UNCERTAIN = REVIEW / "us-long-historical-third-review-uncertain-v1.json"
APPROVED = {
    "us-long-historical-2025-02757-1": "Board is generic; the postal address is complete.",
    "us-long-historical-2025-05093-1": "The named survey is a program, not an organization.",
    "us-long-historical-2025-16313-1": "MSPB in a record ID is excluded; the possessive agency mention is included.",
    "us-long-historical-2025-19368-1": "Council and named committee are separate organization spans.",
    "us-long-historical-2025-19610-2": "ETAC and the named offices are organizational units.",
}
OUT = ROOT / "data/interim/silver/r25/us-long-historical-third-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def candidate_rows():
    """Accept only manually resolved rows with exact third-review agreement."""
    packet = lines(PACKET)
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names):
        raise ValueError("historical first reviews changed")
    a_uncertain = uncertain_names(UNCERTAIN_A)
    b_uncertain = uncertain_names(UNCERTAIN_B)
    disputed = [row for row, a, b in zip(packet, left, right)
                if row["name"] in a_uncertain | b_uncertain or
                a["expected"] != b["expected"]]
    blind, third = lines(THIRD_BLIND), lines(THIRD_REVIEW)
    if (len(disputed) != 21 or
            [row["name"] for row in blind] != [row["name"] for row in disputed] or
            [row["name"] for row in third] != [row["name"] for row in disputed]):
        raise ValueError("historical third review does not cover the disputed packet")
    third_uncertain = uncertain_names(THIRD_UNCERTAIN)
    if not third_uncertain <= {row["name"] for row in third}:
        raise ValueError("historical third uncertainty refers to an unknown row")
    first = {row["name"]: row for row in left}
    second = {row["name"]: row for row in right}
    source_by_name = {row["name"]: row for row in packet}
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    accepted = []
    for blind_row, review in zip(blind, third):
        name = review["name"]
        source = source_by_name[name]
        if (blind_row["input"] != source["input"] or
                review["input"] != source["input"]):
            raise ValueError(f"historical third-review text changed: {name}")
        spans = validate_spans(name, source["input"], review["expected"])
        if name not in APPROVED:
            continue
        a_spans = validate_spans(name, source["input"], first[name]["expected"])
        b_spans = validate_spans(name, source["input"], second[name]["expected"])
        if name in third_uncertain or spans not in (a_spans, b_spans):
            raise ValueError(f"historical third-review decision changed: {name}")
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1 or
                document_key(source) in eval_keys):
            raise ValueError(f"historical third-review source changed: {name}")
        labeled = collections.defaultdict(set)
        for span in spans:
            if span["kind"] in {"email", "phone"}:
                labeled[span["kind"]].add(span["text"])
        sample = [(name, source["input"], spans)]
        if (set(EMAIL.findall(source["input"])) - labeled["email"] or
                {phone_key(value) for value in PHONE.findall(source["input"])} -
                {phone_key(value) for value in labeled["phone"]} or
                collisions(evaluation, sample) or contact_collisions(evaluation, sample)):
            raise ValueError(f"historical third-review labels failed strict checks: {name}")
        accepted.append({
            "name": name, "country": "US", "doc_type": "notice_long_context",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
    if ({row["name"] for row in accepted} != set(APPROVED) or
            len({document_key(row) for row in accepted}) != len(accepted)):
        raise ValueError("historical third-review accepted set changed")
    return accepted


def main():
    """Pin resolved historical passages without activating them."""
    rows = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET.with_suffix(".manifest.json"), REVIEW_A,
              REVIEW_B, UNCERTAIN_A, UNCERTAIN_B, THIRD_BLIND,
              THIRD_REVIEW, THIRD_UNCERTAIN, EVALUATION_PATH)
    manifest = {
        "kind": "us_long_historical_third_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "decisions": APPROVED,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical third candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical third candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical third candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} historical long passages resolved after a third review")


if __name__ == "__main__":
    main()
