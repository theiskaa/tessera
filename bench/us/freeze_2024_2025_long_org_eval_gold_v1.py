"""Freeze independently reviewed, source- and name-disjoint long US evaluation."""

import collections
import json

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as PRIOR_EVALUATION
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2024_2025_long_org_eval_candidates_v2 import OUT as PACKET
from freeze_2024_2025_long_org_eval_candidates_v2 import main as verify_packet
from freeze_2024_2025_long_org_train_candidates_v1 import OUT as NEW_TRAIN
from freeze_2024_2025_long_org_train_candidates_v1 import main as verify_train
from reconcile_long_org_reviews import reconciled_rows
from screen_2024_2025_long_org_prose_v1 import ACTIVE_AT_CAPTURE, RAW, ROOT
from screen_2024_2025_long_org_prose_v1 import source_family
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key


REVIEW = ROOT / "data/interim/review"
REVIEW_PATHS = tuple(REVIEW / f"us-long-org-eval-{name}-v1.jsonl"
                     for name in ("review-a", "review-b", "disputes", "review-c"))
OUT = REVIEW / "us-long-org-eval-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXPECTED_UNRESOLVED = {
    "us-long-org-prose-2025-11858-1939",
    "us-long-org-prose-2025-23366-735",
}


def gold_rows():
    """Keep complete-review agreements with no V4 or new-train exposure."""
    verify_packet()
    new_train = verify_train()
    reviewed, unresolved, decisions = reconciled_rows(PACKET, *REVIEW_PATHS)
    if (len(reviewed) != 41 or set(unresolved) != EXPECTED_UNRESOLVED or
            decisions["third_agreed_a"] != 1 or decisions["third_agreed_b"] != 3):
        raise ValueError("long evaluation review resolution changed")
    active_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                    for name, _ in ACTIVE_AT_CAPTURE]
    active = [row for path in active_paths for row in lines(path)]
    samples = [(row["id"], row["text"], row["entities"]) for row in active]
    samples.extend((row["name"], row["input"], row["expected"])
                   for row in new_train)
    source_keys = {document_key(row) for row in active + new_train}
    source_groups = {source_family(row["source_url"])
                     for row in active + new_train
                     if row.get("source_url", "").startswith(
                         "https://www.federalregister.gov/documents/")}
    original_eval, _ = exclusion_rows_v4()
    original_keys = {document_key(row) for row in original_eval}
    labeled = collisions(reviewed, samples)
    contacts = contact_collisions(reviewed, samples)
    train_shingles = set().union(*(shingles(text, 12)
                                   for _, text, _ in samples))
    selected = []
    exclusions = collections.Counter()
    for row in reviewed:
        name = row["name"]
        if name in labeled:
            exclusions["shared_labeled_entity"] += 1
            continue
        if name in contacts:
            exclusions["shared_contact"] += 1
            continue
        if shingles(row["input"], 12) & train_shingles:
            exclusions["shared_prose"] += 1
            continue
        key = document_key(row)
        if (key is None or key in source_keys or key in original_keys or
                row["source_group"] in source_groups):
            raise ValueError(f"long evaluation source crossed a split: {name}")
        raw_bytes = (RAW / f"{row['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != row["raw_sha256"] or
                raw["id"] != row["source_id"] or
                raw["url"] != row["source_url"] or
                raw["text"].encode()[row["source_start_byte"]:
                                     row["source_end_byte"]].decode() != row["input"]):
            raise ValueError(f"long evaluation raw source changed: {name}")
        selected.append({
            "name": name, "country": "US", "doc_type": "notice_long_org_prose",
            "input": row["input"], "expected": row["expected"],
            "source_id": row["source_id"], "source_url": row["source_url"],
            "source_group": row["source_group"],
            "raw_sha256": row["raw_sha256"],
        })
    if (len(selected) != 22 or len({document_key(row) for row in selected}) != 22 or
            exclusions != {"shared_labeled_entity": 17, "shared_prose": 2}):
        raise ValueError(f"long evaluation membership changed: {dict(exclusions)}")
    return selected, exclusions, decisions, active_paths


def main():
    """Pin the long-form gold set outside the current training mixture."""
    rows, exclusions, decisions, active_paths = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET.with_suffix(".manifest.json"), NEW_TRAIN,
              PRIOR_EVALUATION, *REVIEW_PATHS, *active_paths)
    manifest = {
        "kind": "us_long_org_eval_gold_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "review_decisions": decisions,
        "excluded_after_review": dict(sorted(exclusions.items())),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("long evaluation gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("long evaluation gold changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("long evaluation gold inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} clean long US evaluation cases frozen")
    return rows


if __name__ == "__main__":
    main()
