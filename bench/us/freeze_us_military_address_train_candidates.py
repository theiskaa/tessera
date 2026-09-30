"""Freeze agreed NARA military address labels after US split checks."""

import collections
import json
from pathlib import Path

from active_sources import PRIOR_MILITARY_SILVER
from address_keys import address_keys
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from check_ready import normalized
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_military_address_packet import MANIFEST as PACKET_MANIFEST
from freeze_us_military_address_packet import OUT as PACKET
from freeze_us_military_address_packet import RAW, packet_rows
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import STRICT
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-military-address-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-military-address-review-b-v1.jsonl"
OUT = ROOT / "data/interim/silver/r26/us-military-address-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-military-address-031": "processing center has an uncertain entity boundary",
    "us-military-address-034": "processing center has an uncertain entity boundary",
    "us-military-address-038": "mail stop and agency boundary is uncertain",
    "us-military-address-039": "attention line organization boundary is uncertain",
    "us-military-address-041": "claims intake center boundary is uncertain",
    "us-military-address-043": "records collection and unit boundary is uncertain",
    "us-military-address-044": "records center may name a facility or unit",
    "us-military-address-045": "data management center boundary is uncertain",
}


def candidate_rows():
    """Rebuild source labels and reject review disagreement or evaluation reuse."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["raw_sha256"] != digest(RAW.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled" or
            packet != packet_rows()):
        raise ValueError("military address packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if ([row["name"] for row in left] != names or
            [row["name"] for row in right] != names or
            not set(EXCLUDED) <= set(names)):
        raise ValueError("military address reviews do not cover packet")
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    evaluation_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in evaluation_paths}
    if packet_manifest["evaluation_sha256"] != evaluation_hashes:
        raise ValueError("military address packet evaluation changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_MILITARY_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    seen = NearTextIndex()
    for row in evaluation:
        seen.add(row["name"], row["input"])
    for row in active:
        seen.add(row["id"], row["text"])
    aliases = active_aliases(evaluation, include_roster=True)
    eval_addresses = set().union(*(address_keys(span["text"]) for row in evaluation
                                   for span in row["expected"] if span["kind"] == "address"))
    rows = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if name in EXCLUDED:
            continue
        if a["uncertain"] or b["uncertain"] or a_spans != b_spans or not a_spans:
            raise ValueError(f"unresolved military address: {name}")
        if seen.prior(source["input"]):
            raise ValueError(f"military address nearly repeats active or evaluation: {name}")
        for span in a_spans:
            value = span["text"]
            if ((span["kind"] == "org" and normalized(value) in aliases) or
                    (span["kind"] == "address" and address_keys(value) & eval_addresses)):
                raise ValueError(f"military address label overlaps evaluation: {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "agency_mailing_address",
            "input": source["input"], "expected": a_spans,
            "source_block": source["source_block"],
            "source_url": source["source_url"],
            "raw_sha256": source["raw_sha256"],
        })
        seen.add(name, source["input"])
    if len(rows) != 6:
        raise ValueError("military address agreed membership changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("military address labels overlap evaluation")
    return rows, evaluation_hashes, {
        str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in active_paths}


def main():
    """Freeze split-safe candidates without activating them in training."""
    rows, evaluation_hashes, prior_hashes = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_military_address_candidate_v1",
        "cases": len(rows),
        "labels": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "excluded": EXCLUDED,
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": evaluation_hashes,
        "prior_silver_sha256": prior_hashes,
        "training_eligible": False,
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("military address candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("military address candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("military address candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed military addresses pass the split gate")


if __name__ == "__main__":
    main()
