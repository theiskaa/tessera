"""Freeze agreed US contact labels after current source and evaluation checks."""

import collections
import json
from pathlib import Path

from active_sources import V3_SILVER
from build_contact_snippets import lines
from build_full_notice_gold import validate_spans
from build_us_v3_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v3_eval_exclusions import exclusion_rows
from check_holdout_overlap import collisions, contact_collisions
from freeze_2025_contact_packet import EMAIL, PHONE
from screen_2026_failure_contacts_v2 import CAPTURE, MANIFEST as PACKET_MANIFEST
from screen_2026_failure_contacts_v2 import OUT as PACKET
from screen_2026_failure_contacts_v2 import digest, selected_rows


ROOT = Path(__file__).resolve().parents[2]
REVIEW_A = ROOT / "data/interim/review/us-failure-target-review-a-v2.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-failure-target-review-b-v2.jsonl"
OUT = ROOT / "data/interim/silver/r25/us-failure-target-contacts-candidate-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def candidate_rows():
    """Keep only full agreement on source-distinct, held-out-safe paragraphs."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet = lines(PACKET)
    rebuilt, _, _, _ = selected_rows()
    if (packet != rebuilt or packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled"):
        raise ValueError("blind failure-target packet changed")
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    if (len(packet) != 40 or len(left) != len(packet) or len(right) != len(packet) or
            [row["name"] for row in left] != [row["name"] for row in packet] or
            [row["name"] for row in right] != [row["name"] for row in packet]):
        raise ValueError("independent US reviews do not cover the packet")
    evaluation = exclusion_rows()
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl" for name, _ in V3_SILVER]
    active = [row for path in active_paths for row in lines(path)]
    blocked_groups = {row.get("source_group") for row in evaluation + active}
    blocked_ids = {row.get("source_id") for row in evaluation + active}
    rows = []
    disposition = collections.Counter()
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        raw_bytes = (CAPTURE / "sources" / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"US failure-target source changed: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a["uncertain"] or b["uncertain"]:
            disposition["uncertain"] += 1
            continue
        if a_spans != b_spans:
            disposition["disagree"] += 1
            continue
        if not a_spans:
            disposition["empty"] += 1
            continue
        if (source["source_group"] in blocked_groups or
                source["source_id"] in blocked_ids):
            raise ValueError(f"US failure-target source crossed a split: {name}")
        labeled = collections.defaultdict(set)
        for span in a_spans:
            if span["kind"] in {"email", "phone"}:
                labeled[span["kind"]].add(span["text"])
        if (set(EMAIL.findall(source["input"])) - labeled["email"] or
                set(PHONE.findall(source["input"])) - labeled["phone"]):
            raise ValueError(f"US failure-target rules are incompletely labeled: {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": a_spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
            "raw_sha256": source["raw_sha256"],
        })
        disposition["agreed"] += 1
    if (disposition != {"agreed": 24, "uncertain": 14, "empty": 2} or
            len({row["source_group"] for row in rows}) != len(rows)):
        raise ValueError(f"US review disposition changed: {dict(disposition)}")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("agreed US labels overlap held-out evaluation")
    return rows, disposition, active_paths


def main():
    """Write a pinned candidate set without activating model training."""
    rows, disposition, active_paths = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_failure_target_contacts_candidate_v2",
        "training_eligible": False,
        "intended_use": "candidate_after_final_input_integration_gate",
        "cases": len(rows),
        "disposition": dict(sorted(disposition.items())),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "source_group_key": "source_group",
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": digest(EVALUATION_PATH.read_bytes()),
        "active_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                          for path in active_paths},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US candidate changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US candidate inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed US contacts pass the current split gate")


if __name__ == "__main__":
    main()
