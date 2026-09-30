"""Freeze agreed DOL office contacts outside the reserved states and US evaluation."""

import collections
import json
from pathlib import Path

from active_sources import PRIOR_DOL_SILVER
from address_keys import address_keys
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from check_ready import normalized, person_key
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_dol_whd_office_packet import MANIFEST as PACKET_MANIFEST
from freeze_us_dol_whd_office_packet import OUT as PACKET
from freeze_us_dol_whd_office_packet import SOURCE_MANIFEST, packet_rows
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import STRICT
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW_A = ROOT / "data/interim/review/us-dol-whd-office-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-dol-whd-office-review-b-v1.jsonl"
OUT = ROOT / "data/interim/silver/r26/us-dol-whd-office-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
HELDOUT_STATES = {
    "Alabama", "Florida", "Illinois", "Massachusetts", "New York", "Ohio",
    "Texas", "Virginia",
}
EXCLUDED_STATES = {"South Carolina": "evaluation physical address overlap"}


def candidate_rows():
    """Rebuild agreed spans from the pinned official page and blind reviews."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["source_manifest_sha256"] !=
            digest(SOURCE_MANIFEST.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled" or
            packet != packet_rows()):
        raise ValueError("DOL office packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if [row["name"] for row in left] != names or [row["name"] for row in right] != names:
        raise ValueError("DOL reviews do not cover the frozen packet")
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    evaluation_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in evaluation_paths}
    if packet_manifest["evaluation_sha256"] != evaluation_hashes:
        raise ValueError("DOL packet evaluation membership changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_DOL_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    seen = NearTextIndex()
    for row in evaluation:
        seen.add(row["name"], row["input"])
    for row in active:
        seen.add(row["id"], row["text"])
    aliases = active_aliases(evaluation, include_roster=True)
    eval_people = {person_key(span["text"]) for row in evaluation
                   for span in row["expected"] if span["kind"] == "person"}
    eval_addresses = set().union(*(address_keys(span["text"]) for row in evaluation
                                   for span in row["expected"] if span["kind"] == "address"))
    rows = []
    heldout = []
    excluded = {}
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if source["source_state"] in HELDOUT_STATES:
            heldout.append(name)
            continue
        if source["source_state"] in EXCLUDED_STATES:
            excluded[name] = EXCLUDED_STATES[source["source_state"]]
            continue
        if a["uncertain"] or b["uncertain"] or a_spans != b_spans:
            excluded[name] = "review uncertainty or disagreement"
            continue
        if seen.prior(source["input"]):
            excluded[name] = "near duplicate in evaluation or active silver"
            continue
        overlap = False
        for span in a_spans:
            value = span["text"]
            if ((span["kind"] == "org" and normalized(value) in aliases) or
                    (span["kind"] == "person" and person_key(value) in eval_people) or
                    (span["kind"] == "address" and address_keys(value) & eval_addresses)):
                overlap = True
        if overlap:
            excluded[name] = "evaluation entity overlap"
            continue
        kinds = collections.Counter(span["kind"] for span in a_spans)
        if kinds["org"] != 1 or kinds["address"] != 1 or kinds["phone"] != 1:
            excluded[name] = "incomplete office contact labels"
            continue
        rows.append({
            "name": name, "country": "US", "doc_type": "agency_mailing_address",
            "input": source["input"], "expected": a_spans,
            "source_state": source["source_state"],
            "source_block": source["source_block"],
            "source_url": source["source_url"], "raw_sha256": source["raw_sha256"],
        })
        seen.add(name, source["input"])
    if len(heldout) != len(HELDOUT_STATES) or len(rows) + len(heldout) + len(excluded) != 42:
        raise ValueError("DOL office training membership changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("DOL office labels overlap evaluation")
    return rows, heldout, excluded, evaluation_hashes, {
        str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in active_paths}


def main():
    """Pin conservative DOL candidates without activating them in training."""
    rows, heldout, excluded, evaluation_hashes, prior_hashes = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_dol_whd_office_candidate_v1", "cases": len(rows),
        "heldout_names": heldout, "heldout_states": sorted(HELDOUT_STATES),
        "excluded": excluded,
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
        raise ValueError("DOL office candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("DOL office candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("DOL office candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed DOL offices pass the split gate; "
          f"{len(heldout)} reserved; {len(excluded)} excluded")


if __name__ == "__main__":
    main()
