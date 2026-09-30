"""Freeze independently agreed US organization contacts for the next data gate."""

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
from screen_org_rich_contact_candidates import MANIFEST as PACKET_MANIFEST
from screen_org_rich_contact_candidates import PACKET, RAW, STRICT


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-org-rich-contact-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-org-rich-contact-review-b-v1.jsonl"
OUT = ROOT / "data/interim/silver/r25/us-org-rich-contacts-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDED = {
    "us-org-contact-2024-28359-1": "reviewers disagree on a support desk boundary; source contains broken markup",
    "us-org-contact-2024-28679-1": "FEMA aliases held-out organization",
    "us-org-contact-2024-31029-1": "BLM aliases held-out organization",
    "us-org-contact-2025-02601-1": "reviewers disagree on the committee acronym",
    "us-org-contact-2025-05466-2": "FAA aliases held-out organization",
    "us-org-contact-2025-07357-2": "FAA aliases held-out organization and repeats another contact layout",
    "us-org-contact-2025-09359-1": "BLM aliases held-out organization",
    "us-org-contact-2025-14154-1": "person, university, and email repeat held-out contact",
    "us-org-contact-2025-14284-2": "FAA aliases held-out organization and repeats another contact layout",
    "us-org-contact-2025-18576-1": "one reviewer cannot confirm whether the named function is an organization unit",
    "us-org-contact-2025-22457-1": "neither reviewer found a labeled entity",
    "us-org-contact-2025-23192-1": "person, university, and email repeat held-out contact",
}


def candidate_rows():
    """Require agreement, raw provenance, and a fresh evaluation split check."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    packet_bytes = PACKET.read_bytes()
    if (packet_manifest["sha256"] != digest(packet_bytes) or
            packet_manifest["training_eligible"] or
            packet_manifest["label_status"] != "unlabeled"):
        raise ValueError("organization contact packet changed")
    packet = lines(PACKET)
    review_a, review_b = lines(REVIEW_A), lines(REVIEW_B)
    names = [row["name"] for row in packet]
    if (len(packet) != packet_manifest["cases"] or len(set(names)) != len(names) or
            [row["name"] for row in review_a] != names or
            [row["name"] for row in review_b] != names or
            not set(EXCLUDED) <= set(names)):
        raise ValueError("organization contact reviews do not cover the packet")
    evaluation = [row for path in EVALUATION for row in lines(path)] + lines(STRICT)
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    evaluation_paths = [*EVALUATION, STRICT]
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_SILVER]
    for path in [*evaluation_paths, *active_paths]:
        relative = str(path.relative_to(ROOT))
        if digest(path.read_bytes()) != packet_manifest["evaluation_and_active_sha256"][relative]:
            raise ValueError(f"evaluation or active silver changed since screening: {relative}")
    alias_pattern = re.compile("|".join(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        for alias in sorted(active_aliases(evaluation, include_roster=True),
                            key=len, reverse=True)), re.I)
    rows = []
    for source, left, right in zip(packet, review_a, review_b):
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"organization contact source changed: {name}")
        a_spans = validate_spans(name, source["input"], left["entities"])
        b_spans = validate_spans(name, source["input"], right["entities"])
        if name in EXCLUDED:
            continue
        if a_spans != b_spans or left["uncertain"] or right["uncertain"] or not a_spans:
            raise ValueError(f"unresolved organization contact: {name}")
        if alias_pattern.search(source["input"]):
            raise ValueError(f"held-out organization alias in {name}")
        rows.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": a_spans,
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_group": source["source_group"],
        })
    if len(rows) != len(packet) - len(EXCLUDED):
        raise ValueError("organization contact exclusion count changed")
    samples = [(row["name"], row["input"], row["expected"]) for row in rows]
    if collisions(evaluation, samples) or contact_collisions(evaluation, samples):
        raise ValueError("reviewed organization contacts overlap evaluation labels")
    eval_groups = {row["source_group"] for row in evaluation if row.get("source_group")}
    active_groups = {row["source_group"] for path in active_paths
                     for row in lines(path)
                     if row.get("source_group")}
    groups = [row["source_group"] for row in rows]
    if (len(groups) != len(set(groups)) or
            set(groups) & (eval_groups | active_groups)):
        raise ValueError("reviewed organization contact source family repeats")
    return rows, packet_manifest


def main():
    """Write only the eligible candidate subset, pinned to both reviews."""
    rows, packet_manifest = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_org_rich_contacts_candidate_v1",
        "cases": len(rows),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "excluded": EXCLUDED,
        "packet_sha256": packet_manifest["sha256"],
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "candidate_after_final_input_integration_gate",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("organization contact candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("organization contact candidate changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("organization contact candidate inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} independently agreed organization contacts pass the split gate")


if __name__ == "__main__":
    main()
