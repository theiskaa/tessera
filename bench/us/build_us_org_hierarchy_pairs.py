"""Build source-backed US unit-parent pairs from two independent reviews."""

import json
import re

from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from freeze_2025_ready_contact_train_candidates import EVALUATION
from org_aliases import active_aliases, normalized
from screen_org_rich_contact_candidates import STRICT
from freeze_us_org_hierarchy_packet import MANIFEST as PACKET_MANIFEST
from freeze_us_org_hierarchy_packet import OUT as PACKET
from freeze_us_org_hierarchy_packet import RAW, ROOT


REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-org-hierarchy-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-org-hierarchy-review-b-v1.jsonl"
OUT = REVIEW / "us-org-hierarchy-pairs-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ROSTER = ROOT / "trainer/data/us-org-hierarchy-pairs.json"
UNIT = re.compile(r"\b(?:office|branch|division|section|directorate|bureau|center|unit)\b", re.I)


def reviewed_pairs():
    """Require relation agreement, source provenance, and name-disjoint evaluation."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"]):
        raise ValueError("organization hierarchy packet changed")
    packet, left, right = lines(PACKET), lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if (len(packet) != packet_manifest["cases"] or len(set(names)) != len(names) or
            [row["name"] for row in left] != names or
            [row["name"] for row in right] != names):
        raise ValueError("organization hierarchy reviews do not cover packet")
    for relative, expected in packet_manifest["silver_input_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"reviewed silver source changed: {relative}")
    evaluation = [row for path in EVALUATION for row in lines(path)] + lines(STRICT)
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    forbidden = {normalized(span["text"]) for row in evaluation
                 for span in row["expected"] if span["kind"] == "org"}
    forbidden.update(active_aliases(evaluation, include_roster=True))
    pairs = {}
    excluded = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"organization hierarchy raw source changed: {name}")
        org = validate_spans(name, source["input"], source["org_entities"])
        if org != source["org_entities"]:
            raise ValueError(f"organization spans are out of order: {name}")
        relation_sets = []
        for review in (a, b):
            relations = set()
            for relation in review["relations"]:
                child = relation["child_index"]
                parent = relation["parent_index"]
                if (not isinstance(child, int) or not isinstance(parent, int) or
                        child == parent or child < 0 or parent < 0 or
                        child >= len(org) or parent >= len(org) or
                        not relation.get("reason")):
                    raise ValueError(f"invalid organization relation: {name}")
                relations.add((child, parent))
            if len(relations) != len(review["relations"]):
                raise ValueError(f"duplicate organization relation: {name}")
            relation_sets.append(relations)
        if relation_sets[0] != relation_sets[1]:
            raise ValueError(f"independent organization relation reviews differ: {name}")
        if a["uncertain"] or b["uncertain"]:
            excluded.append(name)
            continue
        for child_index, parent_index in sorted(relation_sets[0]):
            child = org[child_index]["text"]
            parent = org[parent_index]["text"]
            if (normalized(child) == normalized(parent) or
                    normalized(child) in forbidden or
                    normalized(parent) in forbidden):
                raise ValueError(f"organization pair overlaps evaluation: {name}")
            key = (normalized(child), normalized(parent))
            evidence = {
                "source_id": source["source_id"],
                "source_url": source["source_url"],
                "raw_sha256": source["raw_sha256"],
                "silver_file": source["silver_file"],
                "silver_id": source["silver_id"],
                "child_start": org[child_index]["start"],
                "child_end": org[child_index]["end"],
                "parent_start": org[parent_index]["start"],
                "parent_end": org[parent_index]["end"],
            }
            if key not in pairs:
                pairs[key] = {"child": child, "parent": parent, "evidence": []}
            pairs[key]["evidence"].append(evidence)
    rows = [pairs[key] for key in sorted(pairs)]
    if len(rows) < 40 or len(excluded) < 1:
        raise ValueError("too few certain source-backed organization pairs")
    return rows, excluded


def main():
    """Freeze reviewed relation data before synthetic generation changes."""
    rows, excluded = reviewed_pairs()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_org_hierarchy_pairs_v1",
        "pairs": len(rows),
        "evidence_records": sum(len(row["evidence"]) for row in rows),
        "source_ids": len({item["source_id"] for row in rows
                           for item in row["evidence"]}),
        "excluded_uncertain_cases": sorted(excluded),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False,
        "intended_use": "verified_relation_pool_pending_split_and_generation_gate",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("organization pair file or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("organization pair file changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("organization pair inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    roster_pairs = [row for row in rows if UNIT.search(row["child"])]
    if len(roster_pairs) < 40:
        raise ValueError("too few named units for source-backed generation")
    roster = {
        "kind": "us_org_hierarchy_roster_v1",
        "source_pairs_sha256": digest(data),
        "review_a_sha256": manifest["review_a_sha256"],
        "review_b_sha256": manifest["review_b_sha256"],
        "pairs": roster_pairs,
    }
    roster_data = (json.dumps(roster, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if ROSTER.exists() and ROSTER.read_bytes() != roster_data:
        raise ValueError("source-backed organization roster changed after freezing")
    if not ROSTER.exists():
        ROSTER.write_bytes(roster_data)
    print(f"{len(rows)} source-backed unit-parent pairs from "
          f"{manifest['source_ids']} sources; {len(roster_pairs)} named-unit pairs for generation; "
          f"{len(excluded)} uncertain cases excluded")


if __name__ == "__main__":
    main()
