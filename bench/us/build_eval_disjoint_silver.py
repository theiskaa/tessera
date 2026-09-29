"""Retain frozen silver sources while removing cases that expose evaluation bodies."""

import collections
import json
from pathlib import Path

from active_sources import BASE_SILVER, SAFE_REPLACEMENTS
from build_full_notice_gold import digest, read_rows
from check_holdout_overlap import collisions
from freeze_2025_ready_contact_train_candidates import EVALUATION
from org_aliases import active_aliases, normalized


ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/interim/silver"
MODEL_KINDS = {"person", "org", "address"}
EXCLUDED_IDS = {
    "us-reviewed-strict-v1": {
        "2024-28669", "2024-28604", "2024-28655",
    },
    "us-reviewed-contact-snippets-v1": {
        "2024-28647-contact-1", "2024-30968-contact-3",
        "2024-29542-contact-3", "2024-27256-contact-1",
    },
    "us-reviewed-org-paragraphs-v1": {
        "2024-27272-org-paragraph-4",
    },
    "us-r23-reviewed-org-v1": {
        "r23-us-org-2024-28677-3", "r23-us-org-2024-29140-6",
    },
    "us-federal-org-reviewed-v2": {
        "raw-us-org-2024-30239-1", "raw-us-org-2024-28701-1",
        "raw-us-org-2024-26466-4",
    },
    "us-2025-year-contact-expansion-v1": {
        "us-2025-contact-2025-01812-1",
    },
}


def filtered_rows(source_name):
    if source_name not in SAFE_REPLACEMENTS or source_name not in EXCLUDED_IDS:
        raise ValueError(f"unknown silver source to filter: {source_name}")
    source = SILVER / f"{source_name}.jsonl"
    source_manifest = source.with_suffix(".manifest.json")
    base_manifest = json.loads(source_manifest.read_text())
    if base_manifest["sha256"] != digest(source.read_bytes()):
        raise ValueError(f"frozen silver source changed: {source_name}")
    rows = read_rows(source)
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or not EXCLUDED_IDS[source_name] <= ids:
        raise ValueError(f"silver source exclusions changed: {source_name}")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    aliases = active_aliases(evaluation, include_roster=True)
    overlap = collisions(evaluation, ((row["id"], row["text"], row["entities"])
                                      for row in rows))
    leaking = {origin for kinds in overlap.values() for matches in kinds.values()
               for origin, _ in matches}
    for row in rows:
        source_bytes = row["text"].encode()
        if any(normalized(source_bytes[span["start"]:span["end"]].decode()) in aliases
               for span in row["entities"] if span["kind"] == "org"):
            leaking.add(row["id"])
    if leaking != EXCLUDED_IDS[source_name]:
        raise ValueError(f"silver evaluation alias exclusions changed: {source_name}: "
                         f"{sorted(leaking ^ EXCLUDED_IDS[source_name])}")
    return [row for row in rows if row["id"] not in leaking], base_manifest


def main():
    base_keys = dict(BASE_SILVER)
    if set(EXCLUDED_IDS) != set(SAFE_REPLACEMENTS):
        raise ValueError("filtered silver source list is incomplete")
    for source_name, safe_name in SAFE_REPLACEMENTS.items():
        rows, base_manifest = filtered_rows(source_name)
        source = SILVER / f"{source_name}.jsonl"
        source_manifest = source.with_suffix(".manifest.json")
        out = SILVER / f"{safe_name}.jsonl"
        manifest_path = out.with_suffix(".manifest.json")
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        counts = collections.Counter(span["kind"] for row in rows
                                     for span in row["entities"]
                                     if span["kind"] in MODEL_KINDS)
        key = base_keys[source_name]
        manifest = {
            "kind": "us_silver_eval_disjoint_v1",
            "source_name": source_name,
            "source_sha256": digest(source.read_bytes()),
            "source_manifest_sha256": digest(source_manifest.read_bytes()),
            "excluded_ids": sorted(EXCLUDED_IDS[source_name]),
            "filter_evaluation_sha256": {path.name: digest(path.read_bytes())
                                         for path in EVALUATION},
            key: base_manifest[key],
            "documents": len(rows),
            "labels": dict(sorted(counts.items())),
            "sha256": digest(data),
        }
        if out.exists() != manifest_path.exists():
            raise ValueError(f"filtered silver or manifest is missing: {safe_name}")
        if out.exists() and out.read_bytes() != data:
            raise ValueError(f"filtered silver changed: {safe_name}")
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"filtered silver manifest changed: {safe_name}")
        if not out.exists():
            out.write_bytes(data)
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"{safe_name}: {len(rows)} documents, {len(EXCLUDED_IDS[source_name])} excluded")


if __name__ == "__main__":
    main()
