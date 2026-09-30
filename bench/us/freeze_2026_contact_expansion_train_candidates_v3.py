"""Freeze only agreed, certain US contact labels with no evaluation overlap."""

import collections
import json
from pathlib import Path

from build_contact_snippets import lines
from build_full_notice_gold import validate_spans
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2025_contact_packet import EMAIL, PHONE
from screen_2026_contact_expansion_v3 import CAPTURE
from screen_2026_contact_expansion_v3 import MANIFEST as PACKET_MANIFEST
from screen_2026_contact_expansion_v3 import OUT as PACKET
from screen_2026_contact_expansion_v3 import main as verify_packet
from screen_2026_contact_expansion_v3 import selected_rows
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
REVIEW_A = ROOT / "data/interim/review/us-contact-expansion-review-a-v3.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-contact-expansion-review-b-v3.jsonl"
UNCERTAIN_A = ROOT / "data/interim/review/us-contact-expansion-review-a-uncertain-v3.json"
UNCERTAIN_B = ROOT / "data/interim/review/us-contact-expansion-review-b-uncertain-v3.json"
OUT = ROOT / "data/interim/silver/r25/us-contact-expansion-candidate-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def uncertain_names(path):
    """Read either reviewer's separate uncertainty notes."""
    data = json.loads(path.read_text())
    notes = data["uncertainties"] if isinstance(data, dict) else data
    names = [note["name"] for note in notes]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate uncertainty note: {path}")
    return set(names)


def candidate_rows():
    """Require full independent agreement, source integrity, and a clean holdout."""
    verify_packet()
    packet, _, _, _ = selected_rows()
    frozen = lines(PACKET)
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet != frozen or len(packet) != 80 or
            packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"] is not False or
            packet_manifest["label_status"] != "unlabeled"):
        raise ValueError("US contact expansion blind packet changed")
    a = lines(REVIEW_A)
    b = lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in a] != names or
            [row["name"] for row in b] != names):
        raise ValueError("US contact expansion reviews do not cover the packet")
    uncertain = uncertain_names(UNCERTAIN_A) | uncertain_names(UNCERTAIN_B)
    if not uncertain <= set(names):
        raise ValueError("US contact expansion uncertainty refers to an unknown row")
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation}
    disposition = collections.Counter()
    rows = []
    for source, left, right in zip(packet, a, b):
        name = source["name"]
        if left["input"] != source["input"] or right["input"] != source["input"]:
            raise ValueError(f"US contact expansion review text changed: {name}")
        raw_bytes = (CAPTURE / "sources" / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"US contact expansion source changed: {name}")
        if document_key(source) in eval_keys:
            raise ValueError(f"US contact expansion evaluation source reused: {name}")
        left_spans = validate_spans(name, source["input"], left["expected"])
        right_spans = validate_spans(name, source["input"], right["expected"])
        if name in uncertain:
            disposition["uncertain"] += 1
            continue
        if left_spans != right_spans:
            disposition["disagree"] += 1
            continue
        if not left_spans:
            disposition["empty"] += 1
            continue
        labeled = collections.defaultdict(set)
        for span in left_spans:
            if span["kind"] in {"email", "phone"}:
                labeled[span["kind"]].add(span["text"])
        if (set(EMAIL.findall(source["input"])) - labeled["email"] or
                set(PHONE.findall(source["input"])) - labeled["phone"]):
            raise ValueError(f"US contact expansion has unlabeled contact data: {name}")
        sample = [(name, source["input"], left_spans)]
        if collisions(evaluation, sample) or contact_collisions(evaluation, sample):
            disposition["evaluation_overlap"] += 1
            continue
        rows.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": left_spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
        disposition["agreed"] += 1
    if (disposition != {"uncertain": 16, "disagree": 2,
                        "evaluation_overlap": 26, "agreed": 36} or
            len({document_key(row) for row in rows}) != len(rows)):
        raise ValueError(f"US contact expansion disposition changed: {dict(disposition)}")
    return rows, disposition, []


def main():
    """Pin the reviewed candidate without activating it for model training."""
    rows, disposition, _ = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_contact_expansion_candidate_v3",
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
        raise ValueError("US contact expansion candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US contact expansion candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US contact expansion candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-distinct US contact candidates passed independent review")


if __name__ == "__main__":
    main()
