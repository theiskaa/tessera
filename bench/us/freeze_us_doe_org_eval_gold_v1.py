"""Freeze agreed DOE organization prose unseen in prepared US training."""

import ast
import collections
import hashlib
import json
import re
from pathlib import Path

from build_contact_snippets import lines
from build_us_v4_eval_exclusions import MANIFEST as V4_EVAL_MANIFEST
from build_us_v4_eval_exclusions import OUT as V4_EVAL
from build_us_v4_eval_exclusions import exclusion_rows_v4
from check_holdout_overlap import collisions, synthetic_sources
from reconcile_extra_long_org_reviews import reconciled_rows_for
from screen_2026_contact_expansion_v3 import shingles
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/raw/candidates/us-doe-org-overviews-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
SOURCE = ROOT / "data/raw/us-doe-organization-overviews-2024-v1/source.pdf"
SOURCE_MANIFEST = SOURCE.with_name("manifest.json")
REVIEWS = tuple(ROOT / f"data/interim/review/us-doe-org-overviews-review-{label}-v1.jsonl"
                for label in "abc")
REVIEW_UNCERTAINTIES = tuple(
    ROOT / f"data/interim/review/us-doe-org-overviews-review-{label}-uncertain-v1.json"
    for label in "abc"
)
OUT = ROOT / "data/interim/review/us-doe-org-overviews-strict-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
CONFIGS = (ROOT / "configs/detector-shared-v4.toml",
           ROOT / "runs/detector-us-finetune-v5/config.toml")
PROCESSED = (ROOT / "data/processed/detector-us-v4",
             ROOT / "data/processed/detector-us-v1")
GENERATOR_MANIFESTS = (ROOT / "data/manifests/v4/detector-synthetic.json",
                       ROOT / "data/manifests/detector-synthetic.json")
NEW_SILVER = (
    ROOT / "data/interim/silver/us-v5-reviewed-long-org-prose-v1.jsonl",
    ROOT / "data/interim/silver/us-reviewed-extra-long-org-prose-v2.jsonl",
    ROOT / "data/interim/silver/us-reviewed-two-paragraph-org-prose-v3.jsonl",
)
EXPECTED_UNRESOLVED = {15, 106}
EXPECTED_OVERLAP = {6, 37, 40, 42, 45, 86, 128, 135, 142}


def digest(data):
    """Hash a frozen input or output."""
    return hashlib.sha256(data).hexdigest()


def silver_paths(config):
    """Read the configured real sources without loading an application stack."""
    match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", config.read_text())
    if match is None:
        raise ValueError(f"silver list missing from {config}")
    return tuple(ROOT / path for path in ast.literal_eval(match.group(1)))


def checked_synthetic(processed, prose):
    """Screen shared twelve-word text while scanning synthetic labels."""
    for origin, text, spans in synthetic_sources(processed):
        overlap = shingles(text, 12) & prose
        if overlap:
            raise ValueError(f"DOE evaluation prose overlaps {processed}: {len(overlap)}")
        yield origin, text, spans


def gold_rows():
    """Keep whole agreed passages without training source, label, or prose reuse."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (packet_manifest["kind"] != "us_doe_org_overviews_blind_v1" or
            packet_manifest["cases"] != 24 or
            packet_manifest["training_eligible"] or
            packet_manifest["evaluation_eligible"] or
            packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["source_pdf_sha256"] != digest(SOURCE.read_bytes()) or
            source_manifest["pdf_sha256"] != digest(SOURCE.read_bytes())):
        raise ValueError("DOE source or blind packet changed")
    resolved, unresolved, decisions = reconciled_rows_for(PACKET, REVIEWS)
    packet = lines(PACKET)
    pages = {row["name"]: row["source_page"] for row in packet}
    if (len(packet) != 24 or len(resolved) != 22 or
            {pages[name] for name in unresolved} != EXPECTED_UNRESOLVED):
        raise ValueError("DOE independent review resolution changed")
    paths = tuple(dict.fromkeys((*NEW_SILVER, *(path for config in CONFIGS
                                                    for path in silver_paths(config)))))
    real = [row for path in paths for row in lines(path)]
    if any(document_key(row) == document_key(packet[0]) for row in real):
        raise ValueError("DOE evaluation source entered real training")
    prior_eval, _ = exclusion_rows_v4()
    prior_eval += lines(ROOT / "data/interim/review/us-long-org-eval-gold-v4.jsonl")
    near = NearTextIndex()
    for row in real:
        near.add(row["id"], row["text"])
    for row in prior_eval:
        near.add(row["name"], row["input"])
    for row in resolved:
        if near.prior(row["input"]):
            raise ValueError(f"DOE prose repeats prior data: {row['name']}")
    sources = ((row["id"], row["text"], row["entities"]) for row in real)
    overlap = set(collisions(resolved, sources))
    prose = set().union(*(shingles(row["input"], 12) for row in resolved))
    for processed in PROCESSED:
        overlap.update(collisions(resolved, checked_synthetic(processed, prose)))
    overlap_pages = {pages[name] for name in overlap}
    if overlap_pages != EXPECTED_OVERLAP:
        raise ValueError(f"DOE training-name overlap changed: {sorted(overlap_pages)}")
    selected = [row for row in resolved if row["name"] not in overlap]
    if (len(selected) != 13 or sum(len(row["expected"]) for row in selected) != 106 or
            {span["kind"] for row in selected for span in row["expected"]} != {"org"}):
        raise ValueError("DOE strict evaluation membership changed")
    return selected, decisions, paths


def main():
    """Write immutable strict gold and pin all inputs used to choose it."""
    rows, decisions, paths = gold_rows()
    gold = [{**row, "doc_type": "pdf_prose", "source_host": "www.energy.gov"}
            for row in rows]
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in gold).encode()
    inputs = (PACKET, PACKET_MANIFEST, SOURCE, SOURCE_MANIFEST, *REVIEWS,
              *REVIEW_UNCERTAINTIES, V4_EVAL, V4_EVAL_MANIFEST,
              *CONFIGS, *GENERATOR_MANIFESTS,
              *(processed / f"{split}.parquet" for processed in PROCESSED
                for split in ("train", "valid")), *paths,
              ROOT / "data/interim/review/us-long-org-eval-gold-v4.jsonl")
    manifest = {
        "kind": "us_doe_org_overviews_strict_gold_v1",
        "training_eligible": False,
        "intended_use": "source_distinct_label_surface_screened_diagnostic",
        "cases": len(rows),
        "source_documents": 1,
        "org_spans": sum(len(row["expected"]) for row in rows),
        "doe_acronym_spans": sum(span["text"] == "DOE" for row in rows
                                 for span in row["expected"]),
        "common_alias_limit": "DOE can refer to Department of Energy already seen under its full name",
        "unresolved_pages": sorted(EXPECTED_UNRESOLVED),
        "training_overlap_pages": sorted(EXPECTED_OVERLAP),
        "review_decisions": decisions,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("DOE gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("DOE strict gold changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("DOE strict gold inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} DOE passages frozen with {manifest['org_spans']} organization labels")


if __name__ == "__main__":
    main()
