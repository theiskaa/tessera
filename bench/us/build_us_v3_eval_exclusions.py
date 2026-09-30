"""Include reserved Maine offices in the v3 generator exclusions."""

import json
from pathlib import Path

from build_us_full_eval_exclusions import OUT as BASE, combined_rows, digest
from build_us_nrcs_maine_reserved_gold import OUT as MAINE, gold_rows


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/review/us-eval-exclusions-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def exclusion_rows():
    """Rebuild both held-out sources and check their distinct membership."""
    base, _ = combined_rows()
    maine = gold_rows()
    if ([json.loads(line) for line in BASE.read_text().splitlines()] != base or
            [json.loads(line) for line in MAINE.read_text().splitlines()] != maine):
        raise ValueError("US v3 evaluation source changed")
    rows = base + maine
    if (len(rows) != 343 or len({row["name"] for row in rows}) != len(rows) or
            len({digest(row["input"].encode()) for row in rows}) != len(rows) or
            any(row["country"] != "US" for row in rows)):
        raise ValueError("US v3 evaluation membership changed")
    return rows


def main():
    """Pin all v3 exclusions before generating synthetic examples."""
    rows = exclusion_rows()
    data = BASE.read_bytes() + MAINE.read_bytes()
    manifest = {
        "kind": "us_v3_eval_exclusions_v1", "cases": len(rows),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                          for path in (BASE, MAINE)},
        "training_eligible": False, "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US v3 exclusion file or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US v3 exclusions changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US v3 exclusion inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} US v3 evaluation cases excluded from generation")


if __name__ == "__main__":
    main()
