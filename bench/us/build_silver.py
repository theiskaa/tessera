"""Build a US silver candidate from rounds with two completed label passes."""

import collections
import glob
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

from address_keys import address_keys
from silver_dedupe import NearDuplicateIndex


ROOT = Path(__file__).resolve().parents[2]
ROUNDS = (1, 11, 13, 15, 17)
OUT = ROOT / "data/interim/silver/us-reviewed-candidate-v1.jsonl"
MANIFEST = ROOT / "data/interim/silver/us-reviewed-candidate-v1.manifest.json"
STRICT_OUT = ROOT / "data/interim/silver/us-reviewed-strict-v1.jsonl"
STRICT_MANIFEST = ROOT / "data/interim/silver/us-reviewed-strict-v1.manifest.json"
STRICT_ERRATA = ROOT / "bench/us/fixtures/us-strict-label-errata-v1.json"
GOLD = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
PAGE_MARKER = re.compile(rb"\n\n\[\[Page [0-9]+\]\]\n\n")
MAX_FAMILY_DOCS = 12
SOURCE_ARTIFACT = re.compile(r'">|</?[A-Za-z][^>]*>|&(?:amp|quot|nbsp|lt|gt);')
LEGACY_CUTOFF_REASON = "legacy source cut at 2000 supplementary characters"
REVIEW_EXCLUDED = {
    "2024-27630": "unlabeled Fedeli Group Inc. organization occurrence",
    "2024-26636": "USDA alias within a longer labeled organization",
    "2024-27386": "unlabeled Commerce organization mention",
    "2024-27673": "unlabeled Food and Drug Administration mention",
    "2024-29047": "unlabeled NMFS organization mention",
    "2024-29703": "Congress alias within a longer labeled organization",
    "2024-26655": "Commerce alias overlaps evaluation",
    "2024-27098": "Department of Labor alias overlaps evaluation",
    "2024-28233": "National Park Service alias overlaps evaluation",
    "2024-29349": "Department of Defense alias overlaps evaluation",
    "2024-29605": "Commerce alias overlaps evaluation",
    "2024-29640": "Department of Defense alias overlaps evaluation",
    "2024-30667": "National Park Service alias overlaps evaluation",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize_surface(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def pass_labels(directory):
    labels = {}
    for path in sorted(directory.glob("*.json")):
        for name, spans in json.loads(path.read_text()).items():
            if name in labels:
                raise ValueError(f"duplicate label pass id: {name}")
            labels[name] = {(item["kind"], item["text"]) for item in spans}
    return labels


def remove_page_markers(row):
    source = row["text"].encode()
    markers = []
    for match in PAGE_MARKER.finditer(source):
        inside = any(
            span["start"] < match.start() and match.end() < span["end"]
            for span in row["entities"]
        )
        if any(
            match.start() < span["start"] < match.end()
            or match.start() < span["end"] < match.end()
            for span in row["entities"]
        ):
            raise ValueError(f"page marker touches label boundary: {row['id']}")
        markers.append((match.start(), match.end(), b" " if inside else b"\n\n"))
    if not markers:
        return row, 0

    def mapped(offset):
        shift = 0
        for start, end, replacement in markers:
            if offset >= end:
                shift += end - start - len(replacement)
            elif start < offset < end:
                raise ValueError(f"label boundary inside page marker: {row['id']}")
        return offset - shift

    pieces = []
    previous = 0
    for start, end, replacement in markers:
        pieces.extend((source[previous:start], replacement))
        previous = end
    pieces.append(source[previous:])
    cleaned = dict(row)
    cleaned["text"] = b"".join(pieces).decode()
    cleaned["entities"] = [
        {**span, "start": mapped(span["start"]), "end": mapped(span["end"])}
        for span in row["entities"]
    ]
    return cleaned, len(markers)


def family(url):
    slug = urlparse(url).path.rsplit("/", 1)[-1]
    return "-".join(slug.split("-")[:8])


def label_surfaces(row, kind):
    text = row["text"].encode()
    return {
        text[span["start"]:span["end"]]
        for span in row["entities"]
        if span["kind"] == kind
    }


def legacy_cutoff(row):
    marker = "SUPPLEMENTARY INFORMATION:\n"
    return marker in row["text"] and len(row["text"].split(marker, 1)[1]) == 2000


def apply_strict_errata(rows, allowed_missing=frozenset()):
    corrections = json.loads(STRICT_ERRATA.read_text())["corrections"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate strict silver source ID")
    revised = set()
    for correction in corrections:
        name = correction["id"]
        if name in revised or (name not in by_id and name not in allowed_missing):
            raise ValueError(f"duplicate or excluded strict errata source: {name}")
        revised.add(name)
        if name not in by_id:
            continue
        row = by_id[name]
        source = row["text"].encode()
        if digest(source) != correction["text_sha256"]:
            raise ValueError(f"strict errata source changed: {name}")
        for span in correction["remove"]:
            key = {field: span[field] for field in ("kind", "start", "end")}
            if (source[span["start"]:span["end"]].decode() != span["text"] or
                    key not in row["entities"]):
                raise ValueError(f"strict errata removal absent: {name} {span}")
            row["entities"].remove(key)
        for span in correction["add"]:
            if source[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"strict errata addition differs from source: {name} {span}")
            row["entities"].append({field: span[field] for field in ("kind", "start", "end")})
        row["entities"].sort(key=lambda span: (span["start"], span["end"]))
    for row in rows:
        source = row["text"].encode()
        previous_end = 0
        for span in row["entities"]:
            if (span["start"] < previous_end or span["start"] >= span["end"] or
                    span["end"] > len(source)):
                raise ValueError(f"invalid strict label after errata: {row['id']} {span}")
            previous_end = span["end"]
    return rows


def main():
    gold = [json.loads(line) for line in GOLD.read_text().splitlines()]
    gold_names = {row["name"] for row in gold}
    gold_texts = {digest(row["input"].encode()) for row in gold}
    gold_surfaces = {
        kind: {
            normalize_surface(span["text"])
            for row in gold
            for span in row["expected"]
            if span["kind"] == kind
        }
        for kind in ("person", "org", "address")
    }
    gold_address_keys = set().union(*(
        address_keys(span["text"])
        for row in gold for span in row["expected"] if span["kind"] == "address"
    ))
    raw = {}
    for path in (ROOT / "data/raw/silver/federal-register").glob("*.json"):
        row = json.loads(path.read_text())
        if row["id"] in raw:
            raise ValueError(f"duplicate raw silver id: {row['id']}")
        raw[row["id"]] = row
    selected = []
    rounds = {}
    seen = set()
    seen_text = {}
    duplicate_text_ids = []
    disagreement_ids = []
    page_markers_removed = 0
    legacy_cutoff_ids = set()
    for number in ROUNDS:
        directory = ROOT / f"data/interim/silver/r{number}"
        pass1 = pass_labels(directory / "pass1")
        pass2 = pass_labels(directory / "pass2")
        source = directory / "train.jsonl"
        before = len(selected)
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row["country"] != "US":
                continue
            name = row["id"]
            if name in seen or name in gold_names or name not in pass1 or name not in pass2:
                raise ValueError(f"duplicate, evaluation, or missing pass id: {name}")
            seen.add(name)
            if pass1[name] != pass2[name]:
                disagreement_ids.append(name)
                continue
            if name not in raw or row["text"] not in raw[name]["text"] or not raw[name].get("url"):
                raise ValueError(f"missing source lineage: {name}")
            if digest(row["text"].encode()) in gold_texts:
                raise ValueError(f"evaluation text overlap: {name}")
            exported = {
                (span["kind"], row["text"].encode()[span["start"]:span["end"]].decode())
                for span in row["entities"]
            }
            if exported != pass2[name]:
                raise ValueError(f"second-pass labels differ from export: {name}")
            if legacy_cutoff(row):
                legacy_cutoff_ids.add(name)
            row, removed = remove_page_markers(row)
            page_markers_removed += removed
            text_hash = digest(row["text"].encode())
            if text_hash in gold_texts:
                raise ValueError(f"cleaned evaluation text overlap: {name}")
            if text_hash in seen_text:
                if row["entities"] != seen_text[text_hash]["entities"]:
                    raise ValueError(f"conflicting labels for duplicate text: {name}")
                duplicate_text_ids.append(name)
                continue
            seen_text[text_hash] = row
            selected.append(row)
        rounds[f"r{number}"] = {
            "source_sha256": digest(source.read_bytes()),
            "us_documents": len(selected) - before,
        }
    if not selected:
        raise ValueError("no US silver documents")
    ranked = sorted(
        selected,
        key=lambda row: (
            -len(label_surfaces(row, "address")),
            -len(label_surfaces(row, "person")),
            row["id"],
        ),
    )
    family_counts = collections.Counter()
    kept_ids = set()
    for row in ranked:
        group = family(raw[row["id"]]["url"])
        if family_counts[group] < MAX_FAMILY_DOCS:
            kept_ids.add(row["id"])
            family_counts[group] += 1
    family_exclusions = [row["id"] for row in selected if row["id"] not in kept_ids]
    selected = [row for row in selected if row["id"] in kept_ids]
    overlap_documents = collections.Counter()
    quality_exclusions = {}
    strict = []
    near = NearDuplicateIndex()
    for row in selected:
        raw_text = row["text"].encode()
        overlaps = {
            span["kind"]
            for span in row["entities"]
            if span["kind"] in gold_surfaces
            and normalize_surface(raw_text[span["start"]:span["end"]].decode())
            in gold_surfaces[span["kind"]]
        }
        if any(address_keys(raw_text[span["start"]:span["end"]].decode()) & gold_address_keys
               for span in row["entities"] if span["kind"] == "address"):
            overlaps.add("address")
        if overlaps:
            overlap_documents[",".join(sorted(overlaps))] += 1
        elif SOURCE_ARTIFACT.search(row["text"]):
            quality_exclusions[row["id"]] = "source HTML artifact in plain text"
        elif row["id"] in REVIEW_EXCLUDED:
            quality_exclusions[row["id"]] = REVIEW_EXCLUDED[row["id"]]
        elif row["id"] in legacy_cutoff_ids:
            quality_exclusions[row["id"]] = LEGACY_CUTOFF_REASON
        elif near.prior(row):
            quality_exclusions[row["id"]] = "near-duplicate reviewed document"
        else:
            near.add(row)
            strict.append(row)
    counts = collections.Counter(
        span["kind"] for row in selected for span in row["entities"]
    )
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    manifest = {
        "kind": "us_silver_candidate",
        "status": "awaiting_label_quality_audit",
        "documents": len(selected),
        "labels": dict(counts),
        "rounds": rounds,
        "source": "federal-register",
        "source_lineage": "raw source text contains each excerpt and has a URL",
        "passes": "every included document has agreeing pass1 and pass2 labels; exported surfaces equal both",
        "disagreement_ids_excluded": disagreement_ids,
        "evaluation_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_id_or_text_overlap": 0,
        "evaluation_surface_overlap_documents": dict(overlap_documents),
        "page_markers_removed": page_markers_removed,
        "duplicate_text_ids_removed": duplicate_text_ids,
        "family_key": "first eight words of source URL slug",
        "family_document_cap": MAX_FAMILY_DOCS,
        "family_excluded_ids": family_exclusions,
        "sha256": digest(data),
    }
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cutoff_exclusions = {name for name, reason in quality_exclusions.items()
                         if reason == LEGACY_CUTOFF_REASON}
    strict = apply_strict_errata(strict, cutoff_exclusions)
    strict_counts = collections.Counter(
        span["kind"] for row in strict for span in row["entities"]
    )
    strict_data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in strict).encode()
    STRICT_OUT.write_bytes(strict_data)
    STRICT_MANIFEST.write_text(json.dumps({
        "kind": "us_silver_strict_candidate",
        "status": "two agreeing label passes with reviewed errata; source and repeat-label screen passed",
        "documents": len(strict),
        "labels": dict(strict_counts),
        "reviewed_errata_sha256": digest(STRICT_ERRATA.read_bytes()),
        "quality_exclusions": quality_exclusions,
        "legacy_cutoff_excluded_ids": sorted(cutoff_exclusions),
        "evaluation_gold_sha256": digest(GOLD.read_bytes()),
        "evaluation_id_text_entity_or_physical_overlap": 0,
        "sha256": digest(strict_data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"built {len(selected)} US silver candidates; {len(strict)} have no evaluation entity overlap")


if __name__ == "__main__":
    main()
