"""Freeze strictly reviewed long US notice passages before silver promotion."""

import collections
import json
import re

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as EVALUATION_PATH
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2024_2025_long_org_eval_candidates_v2 import OUT as LONG_EVAL_PACKET
from freeze_2024_2025_long_org_eval_candidates_v2 import main as verify_long_eval
from freeze_2025_contact_packet import EMAIL, PHONE
from freeze_2026_long_context_train_candidates_v1 import phone_key
from reconcile_long_org_reviews import reconciled_rows
from screen_2024_2025_long_org_prose_v1 import ACTIVE_AT_CAPTURE
from screen_2024_2025_long_org_prose_v1 import OUT as PACKET
from screen_2024_2025_long_org_prose_v1 import RAW, ROOT
from screen_2024_2025_long_org_prose_v1 import main as verify_packet
from screen_2024_2025_long_org_prose_v1 import source_family
from screen_2026_contact_expansion_v3 import shingles
from silver_dedupe import NearTextIndex
from source_identity import document_key


REVIEW = ROOT / "data/interim/review"
REVIEW_PATHS = tuple(REVIEW / f"us-long-org-prose-{name}-v1.jsonl"
                     for name in ("review-a", "review-b", "disputes", "review-c"))
OUT = ROOT / "data/interim/silver/r25/us-long-org-prose-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXCLUDE_LABEL_AMBIGUITY = {
    "us-long-org-prose-2025-01999-18666":
        "CAS describes a registry number, not an organization mention in this excerpt.",
}
EXPECTED_FAMILY_EXCLUSIONS = {
    "2024-26948", "2024-29254", "2025-01493", "2025-07305", "2025-07438",
    "2025-14197", "2025-18679", "2025-21420", "2025-23895",
}
EXPECTED_UNRESOLVED = {
    "us-long-org-prose-2025-03134-2021",
    "us-long-org-prose-2025-14149-727",
    "us-long-org-prose-2025-21196-39750",
}


def candidate_rows():
    """Keep fully agreed rows outside existing training and reserved evaluation."""
    verify_packet()
    reserved = verify_long_eval()
    resolved, unresolved, decisions = reconciled_rows(PACKET, *REVIEW_PATHS)
    if (len(resolved) != 45 or set(unresolved) != EXPECTED_UNRESOLVED or
            decisions["third_agreed_a"] != 5 or decisions["third_agreed_b"] != 5):
        raise ValueError("long organization review resolution changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_AT_CAPTURE]
    active = [row for path in active_paths for row in lines(path)]
    active_keys = {document_key(row) for row in active}
    active_groups = {source_family(row["source_url"]) for row in active
                     if row.get("source_url", "").startswith(
                         "https://www.federalregister.gov/documents/")}
    evaluation, _ = exclusion_rows_v4()
    eval_keys = {document_key(row) for row in evaluation + reserved}
    eval_groups = {row["source_group"] for row in reserved}
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
    near = NearTextIndex()
    for row in active:
        near.add(row["id"], row["text"])
    selected = []
    family_exclusions = set()
    ambiguity_exclusions = set()
    for row in resolved:
        name = row["name"]
        if row["source_group"] in active_groups:
            family_exclusions.add(row["source_id"])
            continue
        if name in EXCLUDE_LABEL_AMBIGUITY:
            ambiguity_exclusions.add(name)
            continue
        key = document_key(row)
        if (key is None or key in active_keys or key in eval_keys or
                row["source_group"] in eval_groups or
                shingles(row["input"], 12) & eval_shingles or
                near.prior(row["input"])):
            raise ValueError(f"long organization source crossed a split: {name}")
        raw_bytes = (RAW / f"{row['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != row["raw_sha256"] or
                raw["id"] != row["source_id"] or
                raw["url"] != row["source_url"] or
                raw["text"].encode()[row["source_start_byte"]:
                                     row["source_end_byte"]].decode() != row["input"]):
            raise ValueError(f"long organization raw source changed: {name}")
        labels = collections.defaultdict(set)
        for span in row["expected"]:
            labels[span["kind"]].add(span["text"])
        sample = [(name, row["input"], row["expected"])]
        contacts = contact_collisions(evaluation, sample)
        if (set(EMAIL.findall(row["input"])) - labels["email"] or
                {phone_key(value) for value in PHONE.findall(row["input"])} -
                {phone_key(value) for value in labels["phone"]} or
                collisions(evaluation, sample) or
                any(kind != "phone" or any(
                    re.sub(r"\D", "", surface) != "711" for _, surface in matches)
                    for kinds in contacts.values() for kind, matches in kinds.items())):
            raise ValueError(f"long organization labels failed strict checks: {name}")
        active_keys.add(key)
        near.add(name, row["input"])
        selected.append(row)
    if (len(selected) != 35 or family_exclusions != EXPECTED_FAMILY_EXCLUSIONS or
            ambiguity_exclusions != set(EXCLUDE_LABEL_AMBIGUITY) or
            len({document_key(row) for row in selected}) != 35):
        raise ValueError("long organization candidate membership changed")
    return selected, decisions, active_paths


def main():
    """Pin reviewed source-backed training candidates without activating them."""
    rows, decisions, active_paths = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET.with_suffix(".manifest.json"), LONG_EVAL_PACKET,
              EVALUATION_PATH, *REVIEW_PATHS, *active_paths)
    manifest = {
        "kind": "us_long_org_prose_train_candidate_v1",
        "training_eligible": False,
        "label_policy": "exclude bare organizational nouns and unresolved complete rows",
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "review_decisions": decisions,
        "family_exclusions": sorted(EXPECTED_FAMILY_EXCLUSIONS),
        "label_ambiguity_exclusions": EXCLUDE_LABEL_AMBIGUITY,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long organization candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long organization candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long organization candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} strictly reviewed long organization passages frozen")
    return rows


if __name__ == "__main__":
    main()
