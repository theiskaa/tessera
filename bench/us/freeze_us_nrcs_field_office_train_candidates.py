"""Freeze reviewed USDA field-office rows outside held pages and US evaluation."""

import collections
import json
from pathlib import Path

from active_sources import PRIOR_NRCS_SILVER
from address_keys import address_keys
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from check_ready import normalized
from freeze_2025_ready_contact_train_candidates import EVALUATION
from freeze_us_nrcs_field_office_packet import MANIFEST as PACKET_MANIFEST
from freeze_us_nrcs_field_office_packet import OUT as PACKET
from freeze_us_nrcs_field_office_packet import SOURCE_MANIFEST, packet_rows
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import STRICT
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A_V1 = REVIEW / "us-nrcs-pa-field-office-review-a-v1.jsonl"
REVIEW_A_V2 = REVIEW / "us-nrcs-pa-field-office-review-a-v2.jsonl"
REVIEW_B = REVIEW / "us-nrcs-pa-field-office-review-b-v1.jsonl"
POLICY = ROOT / "bench/us/fixtures/us-2025-targeted-contact-adjudication-v1.json"
OUT = ROOT / "data/interim/silver/r26/us-nrcs-pa-field-office-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
HELDOUT_PAGES = {22, 26, 32, 38, 42}
EXCLUDED = {
    "us-nrcs-pa-field-office-40-01": "location alias in named office is unresolved",
}


def candidate_rows():
    """Rebuild agreed labels with PDF lineage and a page-disjoint reserve."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["source_manifest_sha256"] !=
            digest(SOURCE_MANIFEST.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled" or
            packet != packet_rows()):
        raise ValueError("USDA field-office packet changed")
    original, left, right = (lines(path) for path in
                             (REVIEW_A_V1, REVIEW_A_V2, REVIEW_B))
    names = [row["name"] for row in packet]
    if (any([row["name"] for row in review] != names
            for review in (original, left, right)) or
            not set(EXCLUDED) <= set(names)):
        raise ValueError("USDA field-office reviews do not cover packet")
    ruling = json.loads(POLICY.read_text())
    if "Orlando Airports District Office is a named office" not in json.dumps(ruling):
        raise ValueError("named-office adjudication policy changed")
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    evaluation_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in evaluation_paths}
    if packet_manifest["evaluation_sha256"] != evaluation_hashes:
        raise ValueError("USDA packet evaluation membership changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in PRIOR_NRCS_SILVER]
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
    heldout = []
    for source, first, a, b in zip(packet, original, left, right):
        name = source["name"]
        first_spans = validate_spans(name, source["input"], first["entities"])
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if first_spans != a_spans:
            raise ValueError(f"USDA adjudication changed blind spans: {name}")
        if not first["uncertain"] or (not a["uncertain"] and
                                     not a.get("adjudication_reason")):
            raise ValueError(f"USDA adjudication lacks rationale: {name}")
        if source["source_block"]["page"] in HELDOUT_PAGES:
            heldout.append(name)
            continue
        if name in EXCLUDED:
            if not (a["uncertain"] or b["uncertain"]):
                raise ValueError(f"USDA excluded row became certain: {name}")
            continue
        if a["uncertain"] or b["uncertain"] or a_spans != b_spans:
            raise ValueError(f"unresolved USDA field-office row: {name}")
        if seen.prior(source["input"]):
            raise ValueError(f"USDA field office nearly repeats training or evaluation: {name}")
        for span in a_spans:
            value = span["text"]
            if ((span["kind"] == "org" and normalized(value) in aliases) or
                    (span["kind"] == "address" and address_keys(value) & eval_addresses)):
                raise ValueError(f"USDA field-office label overlaps evaluation: {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "agency_mailing_address",
            "input": source["input"], "expected": a_spans,
            "source_block": source["source_block"],
            "source_url": source["source_url"],
            "raw_sha256": source["raw_sha256"],
            "pages_sha256": source["pages_sha256"],
        })
        seen.add(name, source["input"])
    if len(heldout) != 14 or len(rows) != 27:
        raise ValueError("USDA field-office training or held-page membership changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("USDA field-office labels overlap evaluation")
    return rows, heldout, evaluation_hashes, {
        str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in active_paths}


def main():
    """Freeze candidate bytes without activating them in training."""
    rows, heldout, evaluation_hashes, prior_hashes = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_nrcs_pa_field_office_candidate_v1",
        "cases": len(rows), "heldout_names": heldout,
        "heldout_pages": sorted(HELDOUT_PAGES), "excluded": EXCLUDED,
        "labels": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_v1_sha256": digest(REVIEW_A_V1.read_bytes()),
        "review_a_v2_sha256": digest(REVIEW_A_V2.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "named_office_policy_sha256": digest(POLICY.read_bytes()),
        "evaluation_sha256": evaluation_hashes,
        "prior_silver_sha256": prior_hashes,
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("USDA field-office candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("USDA field-office candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("USDA field-office candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed USDA field offices pass the split gate; {len(heldout)} reserved")


if __name__ == "__main__":
    main()
