"""Pin US evaluation exclusions with the reviewed development corrections."""

import json
from pathlib import Path

from build_corrected_dev_v3 import OUT as CORRECTED, corrected_rows_v3
from build_us_v3_eval_exclusions import OUT as V3, exclusion_rows
from build_us_full_eval_exclusions import digest


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/review/us-eval-exclusions-v4.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def exclusion_rows_v4():
    """Replace all development gold labels while preserving frozen case membership."""
    existing = exclusion_rows()
    corrected, _ = corrected_rows_v3()
    corrected_by_name = {row["name"]: row for row in corrected}
    if len(existing) != 343 or len(corrected_by_name) != 196:
        raise ValueError("US evaluation case membership changed")
    rows = []
    seen = set()
    changed = []
    for row in existing:
        name = row["name"]
        if name in corrected_by_name:
            revised = corrected_by_name[name]
            if (row["input"] != revised["input"] or
                    row["country"] != revised["country"]):
                raise ValueError(f"US development source changed: {name}")
            if row["expected"] != revised["expected"]:
                changed.append(name)
            row = revised
            seen.add(name)
        rows.append(row)
    if (seen != set(corrected_by_name) or len(changed) != 8 or
            len({row["name"] for row in rows}) != len(rows)):
        raise ValueError(f"US development corrections changed: {changed}")
    return rows, changed


def main():
    """Freeze the corrected exclusion labels for the next US data generation."""
    rows, changed = exclusion_rows_v4()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_v4_eval_exclusions_v1",
        "training_eligible": False,
        "cases": len(rows),
        "corrected_cases": changed,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (V3, CORRECTED)
        },
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("US v4 exclusion file or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("US v4 exclusions changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("US v4 exclusion inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} US v4 evaluation cases excluded; {len(changed)} corrected")


if __name__ == "__main__":
    main()
