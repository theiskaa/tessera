"""Freeze source-disjoint US environmental phrases for blind entity review."""

import hashlib
import json
import re
from pathlib import Path

from active_sources import PRIOR_DOL_REMAINING_SILVER
from build_contact_snippets import lines
from freeze_2025_ready_contact_train_candidates import EVALUATION
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/candidates/us-environmental-hard-negatives-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
SOURCE_DIRS = (
    ROOT / "data/raw/silver/federal-register",
    ROOT / "data/raw/candidates/federal-register-2026-contacts-v1/sources",
)
TARGETS = (
    ("2024-27844", "NEPA Compliance Specialist"),
    ("2026-00468", "Record of Decision"),
    ("2026-01367", "Record of Decision"),
    ("2026-02108", "Final EIS"),
    ("2026-03081", "Finding of No Significant Impact"),
    ("2026-05489", "final EIS"),
    ("2026-12215", "Finding of No Significant Impact"),
    ("2026-15270", "Record of Decision"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def excerpt(source, phrase):
    """Keep a complete source paragraph or a bounded sentence around the target."""
    match = re.search(re.escape(phrase), source, re.I)
    if match is None:
        raise ValueError(f"missing target phrase: {phrase}")
    start = source.rfind("\n\n", 0, match.start())
    start = 0 if start < 0 else start + 2
    end = source.find("\n\n", match.end())
    end = len(source) if end < 0 else end
    if end - start > 700:
        start = max(source.rfind(".", 0, match.start()) + 1,
                    source.rfind("\n", 0, match.start()) + 1)
        end = source.find(".", match.end())
        end = len(source) if end < 0 else end + 1
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    if end - start > 700 or end - start < len(phrase):
        raise ValueError(f"environmental excerpt is unclear: {phrase}")
    return start, end


def source_matches(row, source_id):
    """Check explicit provenance and document-number-bearing case identifiers."""
    return (row.get("source_id") == source_id or
            row.get("source_group") == source_id or
            source_id in row.get("source_url", "") or
            source_id in row.get("name", row.get("id", "")))


def packet_rows():
    """Reproduce snippets and reject evaluation or active source-document overlap."""
    evaluation_paths = [*EVALUATION, STRICT]
    evaluation = [row for path in evaluation_paths for row in lines(path)]
    if len(evaluation) != 339:
        raise ValueError("US evaluation membership changed")
    active = [row for name, _ in PRIOR_DOL_REMAINING_SILVER
              for row in lines(ROOT / f"data/interim/silver/{name}.jsonl")]
    rows = []
    for source_id, phrase in TARGETS:
        if any(source_matches(row, source_id) for row in (*evaluation, *active)):
            raise ValueError(f"environmental source repeats evaluation or training: {source_id}")
        paths = [directory / f"{source_id}.json" for directory in SOURCE_DIRS]
        found = [path for path in paths if path.exists()]
        if len(found) != 1:
            raise ValueError(f"environmental source membership changed: {source_id}")
        path = found[0]
        raw = path.read_bytes()
        source = json.loads(raw)
        if source["id"] != source_id or "federalregister.gov/documents/" not in source["url"]:
            raise ValueError(f"environmental source provenance changed: {source_id}")
        start, end = excerpt(source["text"], phrase)
        snippet = source["text"][start:end]
        if phrase.casefold() not in snippet.casefold() or source["text"].count(snippet) != 1:
            raise ValueError(f"environmental excerpt is not unique: {source_id}")
        rows.append({
            "name": f"us-environmental-negative-{source_id}", "country": "US",
            "input": snippet, "target_phrase": phrase,
            "source_id": source_id, "source_url": source["url"],
            "source_path": str(path.relative_to(ROOT)),
            "source_start": len(source["text"][:start].encode()),
            "source_end": len(source["text"][:end].encode()),
            "raw_sha256": digest(raw),
        })
    return rows


def main():
    """Pin source excerpts for two independent reviews without training use."""
    rows = packet_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_environmental_hard_negative_blind_v1", "cases": len(rows),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "active_silver_sha256": {
            f"data/interim/silver/{name}.jsonl": digest((ROOT / f"data/interim/silver/{name}.jsonl").read_bytes())
            for name, _ in PRIOR_DOL_REMAINING_SILVER},
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("environmental packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("environmental packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("environmental packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-disjoint environmental snippets frozen for blind review")


if __name__ == "__main__":
    main()
