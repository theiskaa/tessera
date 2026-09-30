"""Freeze agreed role, name, and address cases after a strict US split audit."""

import collections
import json
import re
from pathlib import Path

from active_sources import ACTIVE_SILVER
from build_contact_snippets import lines
from build_full_notice_gold import digest, validate_spans
from check_holdout_overlap import collisions, contact_collisions
from freeze_2025_ready_contact_train_candidates import EVALUATION
from org_aliases import active_aliases
from screen_org_rich_contact_candidates import STRICT
from screen_us_error_target_contacts import CAPTURE, MANIFEST as PACKET_MANIFEST
from screen_us_error_target_contacts import OUT as PACKET
from screen_us_error_target_contacts import RAW


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-error-target-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-error-target-review-b-v1.jsonl"
ORG_CANDIDATES = ROOT / "data/interim/silver/r25/us-org-rich-contacts-candidate-v1.jsonl"
OUT = ROOT / "data/interim/silver/r25/us-error-target-contacts-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-error-target-2024-26862-1": "organization name is embedded in a meeting venue",
    "us-error-target-2024-27533-1": "institution and venue boundaries remain uncertain",
    "us-error-target-2025-01757-2": "reviewers disagree whether a mail code belongs in the address",
    "us-error-target-2025-02926-2": "headquarters may be a building or an organization",
    "us-error-target-2025-05111-1": "same organization and numbered street as another candidate",
    "us-error-target-2025-22882-1": "military installation and affiliation relationship is unclear",
    "us-error-target-2026-07403-1": "floor and room boundary remains uncertain",
}


def candidate_rows():
    """Check both independent reviews and every selected source before use."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet_bytes = PACKET.read_bytes()
    if (packet_manifest["sha256"] != digest(packet_bytes) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled"):
        raise ValueError("error-target packet changed")
    packet = lines(PACKET)
    left, right = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if (len(packet) != packet_manifest["cases"] or len(set(names)) != len(names) or
            [row["name"] for row in left] != names or
            [row["name"] for row in right] != names or
            not set(EXCLUDED) <= set(names)):
        raise ValueError("error-target reviews do not cover the packet")
    evaluation_paths = [*EVALUATION, STRICT]
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_SILVER]
    for path in [*evaluation_paths, *active_paths]:
        relative = str(path.relative_to(ROOT))
        if digest(path.read_bytes()) != packet_manifest["evaluation_and_active_sha256"][relative]:
            raise ValueError(f"evaluation or active silver changed: {relative}")
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    aliases = active_aliases(evaluation, include_roster=True)
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(aliases, key=len, reverse=True)), re.I) if aliases else None
    capture_bytes = (CAPTURE / "manifest.json").read_bytes()
    if digest(capture_bytes) != packet_manifest["capture_sha256"]:
        raise ValueError("2026 source capture changed after screening")
    capture = json.loads(capture_bytes)
    capture_sources = {row["source_id"]: row for row in capture["sources"]}
    rows = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        source_id = source["source_id"]
        path = ((CAPTURE / "sources" / f"{source_id}.json")
                if source_id in capture_sources else RAW / f"{source_id}.json")
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source_id or raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"error-target source changed: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if name in EXCLUDED:
            continue
        if a_spans != b_spans or a["uncertain"] or b["uncertain"] or not a_spans:
            raise ValueError(f"unresolved error-target contact: {name}")
        if alias_pattern and alias_pattern.search(source["input"]):
            raise ValueError(f"held-out organization alias in {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": a_spans,
            "source_id": source_id, "source_url": source["source_url"],
            "source_group": source["source_group"],
        })
    if len(rows) != len(packet) - len(EXCLUDED):
        raise ValueError("error-target exclusion count changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("error-target labels overlap evaluation")
    eval_groups = {row["source_group"] for row in evaluation if row.get("source_group")}
    active_groups = {row["source_group"] for path in active_paths
                     for row in lines(path) if row.get("source_group")}
    prior = lines(ORG_CANDIDATES)
    prior_groups = {row["source_group"] for row in prior}
    groups = [row["source_group"] for row in rows]
    if (len(groups) != len(set(groups)) or
            set(groups) & (eval_groups | active_groups | prior_groups)):
        raise ValueError("error-target source family repeats")
    return rows, packet_manifest


def main():
    """Write a pinned candidate set without activating it for training."""
    rows, packet_manifest = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_error_target_contacts_candidate_v1",
        "cases": len(rows),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "excluded": EXCLUDED,
        "packet_sha256": packet_manifest["sha256"],
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "prior_candidate_sha256": digest(ORG_CANDIDATES.read_bytes()),
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "candidate_after_final_input_integration_gate",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("error-target candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("error-target candidate changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("error-target candidate inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} agreed error-target contacts pass the split gate")


if __name__ == "__main__":
    main()
