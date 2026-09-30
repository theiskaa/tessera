"""Freeze agreed long historical US notices after strict split checks."""

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
from screen_2024_2025_long_context_v1 import MANIFEST as PACKET_MANIFEST
from screen_2024_2025_long_context_v1 import OUT as PACKET
from screen_2024_2025_long_context_v1 import RAW, ROOT
from screen_2024_2025_long_context_v1 import main as verify_packet
from source_identity import document_key


REVIEW_A = ROOT / "data/interim/review/us-long-historical-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-long-historical-review-b-v1.jsonl"
UNCERTAIN_A = ROOT / "data/interim/review/us-long-historical-review-a-uncertain-v1.json"
UNCERTAIN_B = ROOT / "data/interim/review/us-long-historical-review-b-uncertain-v1.json"
OUT = ROOT / "data/interim/silver/r25/us-long-historical-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def candidate_rows():
    """Keep only fully agreed, source-bound passages outside evaluation."""
    packet = verify_packet()
    frozen = lines(PACKET)
    metadata = json.loads(PACKET_MANIFEST.read_text())
    if (packet != frozen or len(packet) != 33 or
            metadata["sha256"] != digest(PACKET.read_bytes()) or
            metadata["training_eligible"] is not False or
            metadata["label_status"] != "unlabeled"):
        raise ValueError("historical long-context blind packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names):
        raise ValueError("historical long-context reviews do not cover every passage")
    uncertain = uncertain_names(UNCERTAIN_A) | uncertain_names(UNCERTAIN_B)
    if not uncertain <= set(names):
        raise ValueError("historical uncertainty refers to an unknown passage")
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    disposition = collections.Counter()
    accepted = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        if a["input"] != source["input"] or b["input"] != source["input"]:
            raise ValueError(f"historical review text changed: {name}")
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"historical raw source changed: {name}")
        if document_key(source) in eval_keys:
            raise ValueError(f"historical source crossed evaluation: {name}")
        a_spans = validate_spans(name, source["input"], a["expected"])
        b_spans = validate_spans(name, source["input"], b["expected"])
        if name in uncertain:
            disposition["uncertain"] += 1
            continue
        if a_spans != b_spans:
            disposition["disagree"] += 1
            continue
        if not a_spans:
            disposition["empty"] += 1
            continue
        labeled = collections.defaultdict(set)
        for span in a_spans:
            if span["kind"] in {"email", "phone"}:
                labeled[span["kind"]].add(span["text"])
        if (set(EMAIL.findall(source["input"])) - labeled["email"] or
                {phone_key(value) for value in PHONE.findall(source["input"])} -
                {phone_key(value) for value in labeled["phone"]}):
            raise ValueError(f"historical contact data is incompletely labeled: {name}")
        sample = [(name, source["input"], a_spans)]
        if collisions(evaluation, sample) or contact_collisions(evaluation, sample):
            disposition["evaluation_overlap"] += 1
            continue
        accepted.append({
            "name": name, "country": "US", "doc_type": "notice_long_context",
            "input": source["input"], "expected": a_spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
        disposition["agreed"] += 1
    if (disposition != {"uncertain": 20, "disagree": 1,
                        "evaluation_overlap": 2, "agreed": 10} or
            len({document_key(row) for row in accepted}) != len(accepted)):
        raise ValueError(f"historical long-context disposition changed: {dict(disposition)}")
    return accepted, disposition


def main():
    """Pin the strictly reviewed historical passages without activating them."""
    rows, disposition = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_long_historical_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "disposition": dict(sorted(disposition.items())),
        "input_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (PACKET, PACKET_MANIFEST, REVIEW_A, REVIEW_B,
                         UNCERTAIN_A, UNCERTAIN_B, EVALUATION_PATH)
        },
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("historical candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("historical candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("historical candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} historical long US passages passed strict split checks")


if __name__ == "__main__":
    main()
