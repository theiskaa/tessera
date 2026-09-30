"""Freeze agreed Maine office tables outside reserved offices and evaluation."""

import collections
import json
from pathlib import Path

from active_sources import PRIOR_MAINE_SILVER
from address_keys import address_keys
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from check_ready import normalized, person_key, spans as labeled_spans
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_nrcs_maine_office_packet import (
    MANIFEST as PACKET_MANIFEST, OUT as PACKET, SOURCE_MANIFEST, packet_rows,
)
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import STRICT
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW_A = ROOT / "data/interim/review/us-nrcs-maine-office-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-nrcs-maine-office-review-b-v1.jsonl"
OUT = ROOT / "data/interim/silver/r26/us-nrcs-maine-office-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
HELDOUT = {
    "us-nrcs-maine-office-bangor-field-office",
    "us-nrcs-maine-office-fort-kent-field-office",
    "us-nrcs-maine-office-scarborough-field-office",
    "us-nrcs-maine-office-south-paris-field-office",
}
EXCLUDED = {
    "us-nrcs-maine-office-farmington-field-office",
    "us-nrcs-maine-office-houlton-field-office",
}


def candidate_rows():
    """Rebuild source-backed labels with an office-disjoint reserve."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["source_manifest_sha256"] !=
            digest(SOURCE_MANIFEST.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled" or
            packet != packet_rows()):
        raise ValueError("USDA Maine packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names or
            not HELDOUT <= set(names) or not EXCLUDED <= set(names) or
            HELDOUT & EXCLUDED):
        raise ValueError("USDA Maine reviews do not cover packet")
    evaluation_paths = [*EVALUATION, STRICT]
    strict = [row for path in evaluation_paths for row in lines(path)]
    evaluation_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in evaluation_paths}
    if len(strict) != 339 or packet_manifest["evaluation_sha256"] != evaluation_hashes:
        raise ValueError("US evaluation membership changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_MAINE_SILVER]
    prior_people = {person_key(value)
                    for path in active_paths for row in lines(path)
                    for kind, _, _, value in labeled_spans(row) if kind == "person"}
    reserve = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a["uncertain"] or b["uncertain"] or a_spans != b_spans:
            raise ValueError(f"unresolved USDA Maine office: {name}")
        overlaps_prior = any(span["kind"] == "person" and
                             person_key(span["text"]) in prior_people
                             for span in a_spans)
        if overlaps_prior != (name in EXCLUDED):
            raise ValueError(f"USDA Maine person alias split changed: {name}")
        if name in HELDOUT:
            reserve.append({"name": name, "input": source["input"],
                            "expected": a_spans})
    if len(reserve) != 4:
        raise ValueError("USDA Maine reserve membership changed")
    evaluation = strict + reserve
    prior_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                    for path in active_paths}
    seen = NearTextIndex()
    for row in evaluation:
        seen.add(row["name"], row["input"])
    for path in active_paths:
        for row in lines(path):
            seen.add(row["id"], row["text"])
    aliases = active_aliases(evaluation, include_roster=True)
    eval_people = {person_key(span["text"]) for row in evaluation
                   for span in row["expected"] if span["kind"] == "person"}
    eval_addresses = set().union(*(address_keys(span["text"]) for row in evaluation
                                   for span in row["expected"] if span["kind"] == "address"))
    rows = []
    for source, a in zip(packet, left):
        name = source["name"]
        if name in HELDOUT | EXCLUDED:
            continue
        spans = validate_spans(name, source["input"], a["entities"])
        if seen.prior(source["input"]):
            raise ValueError(f"USDA Maine office nearly repeats evaluation or training: {name}")
        for span in spans:
            value = span["text"]
            if ((span["kind"] == "org" and normalized(value) in aliases) or
                    (span["kind"] == "person" and person_key(value) in eval_people) or
                    (span["kind"] == "address" and address_keys(value) & eval_addresses)):
                raise ValueError(f"USDA Maine office label overlaps evaluation: {name}")
        counts = collections.Counter(span["kind"] for span in spans)
        if (counts["org"] != 1 or counts["address"] != 1 or
                not counts["person"] or counts["person"] != counts["phone"] or
                counts["person"] != counts["email"]):
            raise ValueError(f"USDA Maine office label coverage changed: {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "agency_mailing_address",
            "input": source["input"], "expected": spans,
            "source_block": source["source_block"],
            "source_url": source["source_url"], "raw_sha256": source["raw_sha256"],
        })
        seen.add(name, source["input"])
    if len(rows) != 8:
        raise ValueError("USDA Maine training membership changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("USDA Maine office labels overlap evaluation")
    return rows, reserve, evaluation_hashes, prior_hashes


def main():
    """Pin the agreed train subset while reserving four office tables."""
    rows, reserve, evaluation_hashes, prior_hashes = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_nrcs_maine_office_candidate_v1", "cases": len(rows),
        "heldout_names": [row["name"] for row in reserve],
        "excluded_names": sorted(EXCLUDED),
        "labels": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": evaluation_hashes,
        "prior_silver_sha256": prior_hashes,
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("USDA Maine candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("USDA Maine candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("USDA Maine candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed USDA Maine offices pass the split gate; "
          f"{len(reserve)} reserved")


if __name__ == "__main__":
    main()
