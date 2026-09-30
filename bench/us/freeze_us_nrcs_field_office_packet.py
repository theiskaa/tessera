"""Freeze unlabeled field-office mailing blocks from a 2026 USDA directory."""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from freeze_2025_ready_contact_train_candidates import EVALUATION
from screen_org_rich_contact_candidates import STRICT


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/us-nrcs-pa-directory-2026-v1"
PDF = RAW_DIR / "source.pdf"
PAGES = RAW_DIR / "pages.jsonl"
SOURCE_MANIFEST = RAW_DIR / "manifest.json"
SOURCE = "https://www.nrcs.usda.gov/sites/default/files/2026-02/Personnel_Directory%20Feb%202026-508.pdf"
EXTRACTOR = ROOT / "bench/us/extract_nrcs_pa_directory.swift"
OUT = ROOT / "data/raw/candidates/us-nrcs-pa-field-offices-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
BLOCK = re.compile(
    r"(?m)^([^\n]{4,75}Field Office[^\n]*)\n"
    r"Natural Resources Conservation Service\n([^\n]+)\n([^\n]+)\nFAX:"
)
ZIP = re.compile(r"PA \d{5}(?:-\d{4})?\Z")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_pages():
    """Verify the captured PDF and its page-by-page text extraction."""
    raw = PDF.read_bytes()
    pages_data = PAGES.read_bytes()
    pages = [json.loads(line) for line in pages_data.splitlines()]
    if (len(raw) < 1_000_000 or not raw.startswith(b"%PDF-") or
            len(pages) != 49 or [page["page"] for page in pages] != list(range(1, 50))):
        raise ValueError("incomplete USDA directory capture")
    if sys.platform == "darwin":
        env = os.environ.copy()
        env["SWIFT_MODULE_CACHE_PATH"] = "/private/tmp/tessera-swift-cache"
        env["CLANG_MODULE_CACHE_PATH"] = "/private/tmp/tessera-clang-cache"
        extracted = subprocess.check_output(
            ["swift", str(EXTRACTOR), str(PDF)], env=env, cwd=ROOT,
        )
        if extracted != pages_data:
            raise ValueError("USDA directory page extraction changed")
    return pages, digest(raw), digest(pages_data)


def packet_rows():
    """Keep exact two-line mailing blocks with their source page and offsets."""
    pages, raw_hash, pages_hash = source_pages()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (source_manifest["kind"] != "us_nrcs_pa_directory_2026_v1" or
            source_manifest["source_url"] != SOURCE or
            source_manifest["raw_sha256"] != raw_hash or
            source_manifest["pages_sha256"] != pages_hash or
            source_manifest["pages"] != 49 or
            source_manifest["training_eligible"]):
        raise ValueError("USDA directory source manifest changed")
    rows = []
    for page in pages:
        for ordinal, match in enumerate(BLOCK.finditer(page["text"]), 1):
            end = match.end() - len("\nFAX:")
            text = page["text"][match.start():end]
            if len(text.splitlines()) != 4 or not ZIP.search(text):
                raise ValueError(f"USDA field-office address changed on page {page['page']}")
            rows.append({
                "name": f"us-nrcs-pa-field-office-{page['page']:02d}-{ordinal:02d}",
                "country": "US", "input": text,
                "source_url": SOURCE, "raw_sha256": raw_hash,
                "pages_sha256": pages_hash,
                "source_block": {"page": page["page"],
                                 "start": match.start(), "end": end},
            })
    if len(rows) != 42 or len({row["input"] for row in rows}) != 42:
        raise ValueError(f"USDA field-office packet membership changed: {len(rows)}")
    return rows


def main():
    """Pin the source and blind packet before label review."""
    pages, raw_hash, pages_hash = source_pages()
    source_manifest = {
        "kind": "us_nrcs_pa_directory_2026_v1",
        "source_url": SOURCE,
        "captured_utc": (json.loads(SOURCE_MANIFEST.read_text())["captured_utc"]
                         if SOURCE_MANIFEST.exists() else datetime.now(timezone.utc).isoformat()),
        "raw_sha256": raw_hash, "pages_sha256": pages_hash,
        "pages": len(pages), "extractor": str(EXTRACTOR.relative_to(ROOT)),
        "training_eligible": False,
    }
    if SOURCE_MANIFEST.exists() and json.loads(SOURCE_MANIFEST.read_text()) != source_manifest:
        raise ValueError("USDA directory source manifest changed")
    if not SOURCE_MANIFEST.exists():
        SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    rows = packet_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_nrcs_pa_field_office_blind_v1", "cases": len(rows),
        "source_manifest_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "evaluation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                              for path in (*EVALUATION, STRICT)},
        "training_eligible": False, "label_status": "unlabeled",
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("USDA field-office packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("USDA field-office packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("USDA field-office packet inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} USDA field-office passages frozen for blind review")


if __name__ == "__main__":
    main()
