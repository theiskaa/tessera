"""Freeze clear third-review resolutions from the two long-notice packets."""

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
from screen_2026_long_context_clean_v2 import OUT as CLEAN_PACKET
from screen_2026_long_context_v1 import CAPTURE, OUT as FIRST_PACKET
from screen_2026_long_context_v1 import ROOT
from source_identity import document_key


REVIEW = ROOT / "data/interim/review"
THIRD_BLIND = REVIEW / "us-long-disputed-blind-v1.jsonl"
THIRD_REVIEW = REVIEW / "us-long-third-review-v1.jsonl"
THIRD_UNCERTAIN = REVIEW / "us-long-third-review-uncertain-v1.json"
PACKETS = (
    (FIRST_PACKET, "us-long-context-review-a-v1.jsonl",
     "us-long-context-review-b-v1.jsonl",
     "us-long-context-review-a-uncertain-v1.json",
     "us-long-context-review-b-uncertain-v1.json"),
    (CLEAN_PACKET, "us-long-clean-review-a-v2.jsonl",
     "us-long-clean-review-b-v2.jsonl",
     "us-long-clean-review-a-uncertain-v2.json",
     "us-long-clean-review-b-uncertain-v2.json"),
)
APPROVED = {
    "us-long-context-2026-08652-1",
    "us-long-clean-2026-05578-2",
}
OUT = ROOT / "data/interim/silver/r25/us-long-third-review-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def candidate_rows():
    """Accept a disputed row only after a certain, exact third agreement."""
    packet_rows = []
    disputed = []
    reviews = {}
    uncertainties = {}
    inputs = [THIRD_BLIND, THIRD_REVIEW, THIRD_UNCERTAIN, EVALUATION_PATH]
    for packet_path, a_name, b_name, ua_name, ub_name in PACKETS:
        packet = lines(packet_path)
        a_path, b_path = REVIEW / a_name, REVIEW / b_name
        ua_path, ub_path = REVIEW / ua_name, REVIEW / ub_name
        inputs.extend((packet_path, packet_path.with_suffix(".manifest.json"),
                       a_path, b_path, ua_path, ub_path))
        a, b = lines(a_path), lines(b_path)
        if ([row["name"] for row in packet] != [row["name"] for row in a] or
                [row["name"] for row in packet] != [row["name"] for row in b]):
            raise ValueError("third-review source reviews changed")
        ua, ub = uncertain_names(ua_path), uncertain_names(ub_path)
        for source, left, right in zip(packet, a, b):
            name = source["name"]
            packet_rows.append(source)
            reviews[name] = (left, right)
            uncertainties[name] = name in ua or name in ub
            if uncertainties[name] or left["expected"] != right["expected"]:
                disputed.append(source)
    blind = lines(THIRD_BLIND)
    third = lines(THIRD_REVIEW)
    if ([row["name"] for row in blind] != [row["name"] for row in disputed] or
            [row["name"] for row in third] != [row["name"] for row in disputed] or
            len(third) != 33):
        raise ValueError("third review does not cover exactly the disputed rows")
    by_name = {row["name"]: row for row in packet_rows}
    third_uncertain = {row["name"] for row in json.loads(THIRD_UNCERTAIN.read_text())}
    if not third_uncertain <= {row["name"] for row in third}:
        raise ValueError("third reviewer uncertainty refers to an unknown row")
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    accepted = []
    for blind_row, review in zip(blind, third):
        name = review["name"]
        source = by_name[name]
        if (blind_row["name"] != name or
                blind_row["input"] != source["input"] or
                review["input"] != source["input"]):
            raise ValueError(f"third-review source text changed: {name}")
        spans = validate_spans(name, source["input"], review["expected"])
        if name not in APPROVED:
            continue
        left, right = reviews[name]
        if (name in third_uncertain or uncertainties[name] or
                spans not in (validate_spans(name, source["input"], left["expected"]),
                              validate_spans(name, source["input"], right["expected"])) or
                left["expected"] == right["expected"]):
            raise ValueError(f"third-review agreement changed: {name}")
        raw_bytes = (CAPTURE / "sources" / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1 or
                document_key(source) in eval_keys):
            raise ValueError(f"third-review source identity changed: {name}")
        labeled = collections.defaultdict(set)
        for span in spans:
            if span["kind"] in {"email", "phone"}:
                labeled[span["kind"]].add(span["text"])
        if (set(EMAIL.findall(source["input"])) - labeled["email"] or
                {phone_key(value) for value in PHONE.findall(source["input"])} -
                {phone_key(value) for value in labeled["phone"]} or
                collisions(evaluation, [(name, source["input"], spans)]) or
                contact_collisions(evaluation, [(name, source["input"], spans)])):
            raise ValueError(f"third-review labels failed strict checks: {name}")
        accepted.append({
            "name": name, "country": "US", "doc_type": "notice_long_context",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
    if ({row["name"] for row in accepted} != APPROVED or
            len({document_key(row) for row in accepted}) != len(accepted)):
        raise ValueError("third-review accepted set changed")
    return accepted, inputs


def main():
    """Pin the two resolved passages without activating them."""
    rows, inputs = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_long_third_review_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("third-review candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("third-review candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("third-review candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} disputed long passages resolved by a certain third review")


if __name__ == "__main__":
    main()
