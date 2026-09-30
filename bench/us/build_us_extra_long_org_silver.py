"""Promote the reviewed extra-long US notices without starting training."""

import collections
import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_extra_long_org_train_candidates_v2 import MANIFEST as CANDIDATE_MANIFEST
from freeze_2024_2025_extra_long_org_train_candidates_v2 import OUT as CANDIDATE
from freeze_2024_2025_extra_long_org_train_candidates_v2 import ROOT
from freeze_2024_2025_extra_long_org_train_candidates_v2 import main as verify_candidate
from source_identity import document_key


OUT = ROOT / "data/interim/silver/us-reviewed-extra-long-org-prose-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    """Freeze the source-backed, independently agreed passages as eligible silver."""
    candidates = verify_candidate()
    frozen = [json.loads(line) for line in CANDIDATE.read_text().splitlines()]
    if (len(candidates) != 29 or candidates != frozen or
            json.loads(CANDIDATE_MANIFEST.read_text())["training_eligible"] is not False):
        raise ValueError("extra-long reviewed candidate changed")
    rows = []
    for candidate in candidates:
        key = document_key(candidate)
        if key is None:
            raise ValueError(f"source identity missing: {candidate['name']}")
        rows.append({
            "id": candidate["name"],
            "country": "US",
            "source": "federal-register",
            "source_id": candidate["source_id"],
            "source_url": candidate["source_url"],
            "source_group": candidate["source_group"],
            "source_document_key": key,
            "raw_sha256": candidate["raw_sha256"],
            "text": candidate["input"],
            "entities": [{"kind": span["kind"], "start": span["start"],
                          "end": span["end"]} for span in candidate["expected"]],
        })
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_reviewed_extra_long_org_prose_v2",
        "training_eligible": True,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["entities"]).items())),
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "candidate_manifest_sha256": digest(CANDIDATE_MANIFEST.read_bytes()),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("extra-long silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("extra-long silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("extra-long silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} extra-long US passages are training eligible")


if __name__ == "__main__":
    main()
