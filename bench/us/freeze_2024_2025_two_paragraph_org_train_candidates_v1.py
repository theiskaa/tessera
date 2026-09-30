"""Freeze reviewed two-paragraph US notices outside every training and evaluation split."""

import collections
import json
import re

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from build_us_v4_eval_exclusions import OUT as V4_EVAL
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, contact_collisions
from freeze_2024_2025_long_org_eval_gold_v3 import OUT as LONG_EVAL
from freeze_2024_2025_long_org_eval_gold_v3 import main as verify_long_eval
from freeze_2025_contact_packet import EMAIL, PHONE
from freeze_2026_long_context_train_candidates_v1 import phone_key
from reconcile_extra_long_org_reviews import reconciled_rows_for
from screen_2024_2025_long_org_prose_v1 import RAW
from screen_2024_2025_two_paragraph_org_v1 import MANIFEST as PACKET_MANIFEST
from screen_2024_2025_two_paragraph_org_v1 import OUT as PACKET
from screen_2024_2025_two_paragraph_org_v1 import ROOT, main as verify_packet
from screen_2026_contact_expansion_v3 import shingles
from silver_dedupe import NearTextIndex
from source_identity import document_key


OUT = ROOT / "data/interim/silver/r25/us-two-paragraph-org-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
V4_CONFIG = ROOT / "configs/detector-shared-v4.toml"
REVIEWS = tuple(ROOT / f"data/interim/review/us-two-paragraph-org-review-{label}-v1.jsonl"
                for label in ("a", "b", "c"))
PRIOR_SILVER = (
    ROOT / "data/interim/silver/us-v5-reviewed-long-org-prose-v1.jsonl",
    ROOT / "data/interim/silver/us-reviewed-extra-long-org-prose-v2.jsonl",
)
POLICY_EXCLUSIONS = {
    "2025-07131": "CAS prefixes registry numbers; earlier registry-number notices were excluded on this policy.",
    "2025-11753": "Standards-body acronyms appear only inside technical specification identifiers.",
    "2025-23019": "The sanctions list is dominated by foreign postal addresses outside the US release scope.",
    "2025-23583": "The sanctions list is dominated by foreign postal addresses outside the US release scope.",
}


def candidate_rows():
    """Keep complete agreed labels that pass source, text, and contact isolation."""
    verify_packet()
    resolved, unresolved, decisions = reconciled_rows_for(PACKET, REVIEWS)
    if len(resolved) != 20 or unresolved:
        raise ValueError("two-paragraph review resolution changed")
    match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", V4_CONFIG.read_text())
    if not match:
        raise ValueError("V4 silver list is missing")
    active_paths = [ROOT / path for path in json.loads(match.group(1))] + list(PRIOR_SILVER)
    active = [row for path in active_paths for row in lines(path)]
    active_keys = {document_key(row) for row in active}
    active_groups = {row.get("source_group") for row in active}
    if len(active) != 1734 or len(active_keys) != 705 or None in active_keys:
        raise ValueError("prior US training data changed")
    old_eval, _ = exclusion_rows_v4()
    new_eval = verify_long_eval()
    evaluation = old_eval + new_eval
    eval_keys = {document_key(row) for row in evaluation} - {None}
    eval_groups = {row["source_group"] for row in new_eval}
    if active_keys & eval_keys:
        raise ValueError("prior training and evaluation sources overlap")
    eval_shingles = set().union(*(shingles(row["input"], 12)
                                  for row in evaluation))
    near = NearTextIndex()
    for row in active:
        near.add(row["id"], row["text"])
    selected = []
    excluded = set()
    for row in resolved:
        if row["source_id"] in POLICY_EXCLUSIONS:
            excluded.add(row["source_id"])
            continue
        key = document_key(row)
        if (key is None or key in active_keys or key in eval_keys or
                row["source_group"] in active_groups or
                row["source_group"] in eval_groups):
            raise ValueError(f"source crossed split: {row['name']}")
        if shingles(row["input"], 12) & eval_shingles or near.prior(row["input"]):
            raise ValueError(f"prose repeats training or evaluation: {row['name']}")
        raw_bytes = (RAW / f"{row['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != row["raw_sha256"] or
                raw["id"] != row["source_id"] or
                raw["url"] != row["source_url"] or
                raw["text"].encode()[row["source_start_byte"]:
                                     row["source_end_byte"]].decode() != row["input"]):
            raise ValueError(f"raw provenance changed: {row['name']}")
        labelled = collections.defaultdict(set)
        for span in row["expected"]:
            labelled[span["kind"]].add(span["text"])
        if (set(EMAIL.findall(row["input"])) - labelled["email"] or
                {phone_key(value) for value in PHONE.findall(row["input"])} -
                {phone_key(value) for value in labelled["phone"]}):
            raise ValueError(f"unlabeled contact in {row['name']}")
        sample = [(row["name"], row["input"], row["expected"])]
        if collisions(evaluation, sample) or contact_collisions(evaluation, sample):
            raise ValueError(f"labels overlap held-out evaluation: {row['name']}")
        active_keys.add(key)
        near.add(row["name"], row["input"])
        selected.append(row)
    if len(selected) != 16 or excluded != set(POLICY_EXCLUSIONS):
        raise ValueError("two-paragraph candidate membership changed")
    return selected, decisions, active_paths


def main():
    """Pin reviewed candidates without adding them to a training config."""
    rows, decisions, active_paths = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS, V4_EVAL, LONG_EVAL,
              *active_paths)
    manifest = {
        "kind": "us_two_paragraph_org_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({document_key(row) for row in rows}),
        "review_decisions": decisions,
        "policy_exclusions": POLICY_EXCLUSIONS,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["expected"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("two-paragraph candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("two-paragraph candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("two-paragraph candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed two-paragraph US candidates frozen")
    return rows


if __name__ == "__main__":
    main()
