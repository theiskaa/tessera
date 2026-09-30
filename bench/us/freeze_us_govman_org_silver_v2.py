"""Keep independently agreed Government Manual labels outside US evaluation overlap."""

import collections
import hashlib
import json
import re
from pathlib import Path

from check_holdout_overlap import collisions


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/raw/candidates/us-govman-org-prose-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
REVIEWS = tuple(ROOT / f"data/interim/review/us-govman-org-prose-review-{pass_id}-v1.jsonl"
                for pass_id in "abc")
UNCERTAINTIES = tuple(ROOT / f"data/interim/review/us-govman-org-prose-review-{pass_id}-uncertain-v1.json"
                      for pass_id in "abc")
EVALUATION = tuple(ROOT / "data/interim/review" / name for name in (
    "us-eval-exclusions-v4.jsonl",
    "us-long-org-eval-gold-v4.jsonl",
    "us-doe-org-overviews-strict-gold-v1.jsonl",
))
OUT = ROOT / "data/interim/silver/us-govman-reviewed-org-prose-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PACKAGE_KEY = "govinfo.gov:GOVMAN-2025-12-31"
DOE_REFERENT = re.compile(r"\b(?:DOE|Department of Energy)\b")


def digest(data):
    """Hash exact frozen source, review, and output bytes."""
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    """Read a JSONL artifact without changing row order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def labels(row, source):
    """Validate complete review fields and organization byte spans."""
    if {key: value for key, value in row.items() if key != "expected"} != source:
        raise ValueError(f"review changed blind input: {source['name']}")
    data = source["input"].encode()
    spans = row["expected"]
    previous = 0
    for span in sorted(spans, key=lambda item: item["start"]):
        if (span["kind"] != "org" or not previous <= span["start"] < span["end"] <= len(data)
                or data[span["start"]:span["end"]].decode() != span["text"]):
            raise ValueError(f"invalid review span: {source['name']}")
        previous = span["end"]
    return tuple(sorted((span["start"], span["end"], span["kind"])
                        for span in spans))


def silver_rows():
    """Resolve unanimous rows and remove held-out labels and DOE referents."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["cases"] != 30 or packet_manifest["training_eligible"] or
            packet_manifest["evaluation_eligible"]):
        raise ValueError("blind Government Manual packet changed")
    packet = read_rows(PACKET)
    by_name = {row["name"]: row for row in packet}
    if len(by_name) != len(packet):
        raise ValueError("duplicate blind Government Manual case")
    reviews = []
    for path in REVIEWS[:2]:
        rows = read_rows(path)
        mapping = {row["name"]: row for row in rows}
        if len(mapping) != len(rows) or set(mapping) != set(by_name):
            raise ValueError(f"incomplete independent review: {path.name}")
        for name, row in mapping.items():
            labels(row, by_name[name])
        reviews.append(mapping)
    agreed = []
    disputed = []
    for source in packet:
        name = source["name"]
        a, b = (labels(review[name], source) for review in reviews)
        if a == b:
            agreed.append(reviews[0][name])
        else:
            disputed.append(name)
    third_rows = read_rows(REVIEWS[2])
    third = {row["name"]: row for row in third_rows}
    if len(third) != len(third_rows) or set(third) != set(disputed):
        raise ValueError("third review did not cover exactly the disputed cases")
    for name, row in third.items():
        third_labels = labels(row, by_name[name])
        if third_labels in {labels(review[name], by_name[name]) for review in reviews}:
            raise ValueError(f"third review now resolves a disputed case: {name}")
    evaluation = [row for path in EVALUATION for row in read_rows(path)]
    evaluation_sources = ((row["name"], row["input"], row["expected"])
                          for row in evaluation)
    overlap = collisions(agreed, evaluation_sources, strict_aliases=True)
    kept = [row for row in agreed if row["name"] not in overlap and
            not DOE_REFERENT.search(row["input"])]
    blocked_referents = [row["name"] for row in agreed
                         if row["name"] not in overlap and DOE_REFERENT.search(row["input"])]
    if (len(agreed) != 25 or len(disputed) != 5 or len(overlap) != 13 or
            blocked_referents or len(kept) != 12 or
            sum(len(row["expected"]) for row in kept) != 50):
        raise ValueError("Government Manual training screening changed")
    silver = []
    for row in kept:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected", "source_group",
                                   "source_document_key"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "source_group": PACKAGE_KEY,
                       "source_document_key": PACKAGE_KEY,
                       "entities": [{key: span[key] for key in ("start", "end", "kind")}
                                    for span in row["expected"]]})
    return silver, disputed, overlap, blocked_referents


def main():
    """Freeze evaluated-safe labels without activating a new training run."""
    rows, disputed, overlap, blocked_referents = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS, *UNCERTAINTIES, *EVALUATION,
              ROOT / "data/raw/us-govman-2025-v1/source.xml",
              ROOT / "data/raw/us-govman-2025-v1/manifest.json",
              ROOT / "bench/us/org_aliases.py",
              ROOT / "bench/us/check_holdout_overlap.py")
    manifest = {
        "kind": "us_govman_reviewed_org_prose_v2",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(rows),
        "source_packages": 1,
        "source_entities": len({row["source_entity_id"] for row in rows}),
        "org_spans": sum(len(row["entities"]) for row in rows),
        "negative_passages": sum(not row["entities"] for row in rows),
        "disputed_excluded": sorted(disputed),
        "heldout_overlap_excluded": {
            name: sorted({value for _, value in kinds.get("org", set())})
            for name, kinds in sorted(overlap.items())},
        "doe_referent_excluded": sorted(blocked_referents),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(data),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("Government Manual silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("Government Manual silver changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("Government Manual silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(rows)} Government Manual passages, {manifest['org_spans']} org spans frozen")


if __name__ == "__main__":
    main()
