"""Keep reviewed organization paragraphs absent from stricter silver documents."""

import json
import re
from pathlib import Path

from build_contact_snippets import lines
from build_us_full_eval_exclusions import digest
from silver_dedupe import NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "data/interim/silver/us-reviewed-strict-safe-v1.jsonl"
SOURCE = ROOT / "data/interim/silver/us-reviewed-org-paragraphs-safe-v1.jsonl"
OUT = ROOT / "data/interim/silver/us-reviewed-org-paragraphs-v4.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
DOCUMENT_ID = re.compile(r"/documents/\d{4}/\d{2}/\d{2}/(\d{4}-\d{5})/")


def filtered_rows():
    """Drop paragraphs already contained in a reviewed strict notice."""
    strict = lines(STRICT)
    source = lines(SOURCE)
    if len(strict) != 15 or len(source) != 66:
        raise ValueError("US reviewed organization source membership changed")
    prior = NearTextIndex()
    for row in strict:
        prior.add(row["id"], row["text"])
    rows = []
    dropped = []
    for row in source:
        match = DOCUMENT_ID.search(row["source_url"])
        if not match or not row["id"].startswith(match[1]):
            raise ValueError(f"US organization paragraph source changed: {row['id']}")
        if prior.prior(row["text"]):
            dropped.append(row["id"])
            continue
        rows.append({**row, "source_id": match[1], "source_group": match[1]})
    if len(rows) != 48 or len(dropped) != 18:
        raise ValueError(f"US organization paragraph overlap changed: {len(rows)}, {dropped}")
    return rows, dropped


def main():
    """Freeze source-distinct rows without changing the historical silver."""
    rows, dropped = filtered_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_reviewed_org_paragraphs_v4",
        "training_eligible": True,
        "cases": len(rows),
        "dropped_near_strict_ids": dropped,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (STRICT, SOURCE)
        },
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US v4 organization paragraphs or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US v4 organization paragraphs changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US v4 organization paragraph inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed US organization paragraphs remain after overlap screening")


if __name__ == "__main__":
    main()
