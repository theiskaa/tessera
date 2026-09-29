"""Recover source-backed US contact paragraphs from reviewed Federal Register pages."""

import collections
import hashlib
import json
import re
from pathlib import Path

from address_keys import address_keys
from build_silver import REVIEW_EXCLUDED, SOURCE_ARTIFACT, normalize_surface
from silver_dedupe import NearDuplicateIndex


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data/interim/silver/us-reviewed-candidate-v1.jsonl"
CANDIDATE_MANIFEST = CANDIDATE.with_suffix(".manifest.json")
STRICT = ROOT / "data/interim/silver/us-reviewed-strict-v1.jsonl"
GOLD = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
RAW = ROOT / "data/raw/silver/federal-register"
OUT = ROOT / "data/interim/silver/us-reviewed-contact-snippets-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
BREAK = re.compile(rb"\n[ \t]*\n")
SOURCE_NOISE = re.compile(r"_{5,}|\ufffd")
US_POSTCODE = re.compile(
    r"\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|"
    r"MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|"
    r"UT|VT|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?\b"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def excluded_surface_pattern(gold):
    surfaces = {
        normalize_surface(span["text"])
        for row in gold
        for span in row["expected"]
        if span["kind"] in {"person", "org", "address"}
    }
    expressions = [r"(?<!\w)" + re.escape(surface) + r"(?!\w)" for surface in surfaces]
    expressions.extend(
        r"(?<!\w)" + r"\.?".join(surface) + r"\.?(?!\w)"
        for surface in surfaces
        if re.fullmatch(r"[a-z]{2,8}", surface)
    )
    return re.compile("|".join(sorted(expressions, key=len, reverse=True)))


def paragraphs(source):
    start = 0
    for match in BREAK.finditer(source):
        yield start, match.start()
        start = match.end()
    yield start, len(source)


def contact_heading(text):
    return text.lstrip().startswith((
        "FOR FURTHER INFORMATION CONTACT:",
        "ADDRESSES:",
    ))


def main():
    candidate_bytes = CANDIDATE.read_bytes()
    candidate_manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    if digest(candidate_bytes) != candidate_manifest["sha256"]:
        raise ValueError("reviewed candidate changed since its two-pass manifest")
    gold_bytes = GOLD.read_bytes()
    gold = lines(GOLD)
    forbidden = excluded_surface_pattern(gold)
    gold_addresses = set().union(*(
        address_keys(span["text"])
        for row in gold
        for span in row["expected"]
        if span["kind"] == "address"
    ))
    strict_ids = {row["id"] for row in lines(STRICT)}
    near = NearDuplicateIndex()
    for row in lines(STRICT):
        near.add(row)
    kept = []
    skipped = collections.Counter()
    seen_texts = set()
    for row in lines(CANDIDATE):
        if row["id"] in strict_ids or row["id"] in REVIEW_EXCLUDED:
            continue
        raw = json.loads((RAW / f"{row['id']}.json").read_text())
        if not raw.get("url"):
            raise ValueError(f"{row['id']}: missing source URL")
        source = row["text"].encode()
        for number, (start, end) in enumerate(paragraphs(source), 1):
            piece = source[start:end]
            text = piece.decode()
            if not contact_heading(text) or len(piece) < 80 or len(piece) > 5000:
                continue
            if not text.rstrip().endswith((".", "!", "?")):
                skipped["incomplete_paragraph"] += 1
                continue
            spans = [
                span for span in row["entities"]
                if start <= span["start"] and span["end"] <= end
            ]
            if not spans:
                continue
            if any(
                span["start"] < end and start < span["end"] and span not in spans
                for span in row["entities"]
            ):
                skipped["crosses_paragraph"] += 1
                continue
            if text not in raw["text"]:
                skipped["not_in_raw_source"] += 1
                continue
            if SOURCE_ARTIFACT.search(text) or SOURCE_NOISE.search(text):
                skipped["source_artifact"] += 1
                continue
            if forbidden.search(normalize_surface(text)):
                skipped["evaluation_surface"] += 1
                continue
            if any(
                address_keys(source[span["start"]:span["end"]].decode()) & gold_addresses
                for span in spans if span["kind"] == "address"
            ):
                skipped["evaluation_address"] += 1
                continue
            if any(
                not US_POSTCODE.search(source[span["start"]:span["end"]].decode())
                for span in spans if span["kind"] == "address"
            ):
                skipped["non_us_address"] += 1
                continue
            if digest(piece) in seen_texts:
                skipped["duplicate_text"] += 1
                continue
            adjusted = [
                {**span, "start": span["start"] - start, "end": span["end"] - start}
                for span in spans
            ]
            for span in adjusted:
                if not piece[span["start"]:span["end"]]:
                    raise ValueError(f"{row['id']}: empty label after slicing")
            result = {
                "id": f"{row['id']}-contact-{number}",
                "source": "federal-register",
                "source_url": raw["url"],
                "country": "US",
                "text": text,
                "entities": adjusted,
            }
            if near.prior(result):
                skipped["near_duplicate"] += 1
                continue
            seen_texts.add(digest(piece))
            near.add(result)
            kept.append(result)
    counts = collections.Counter(span["kind"] for row in kept for span in row["entities"])
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept).encode()
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_reviewed_contact_snippets",
        "source_candidate_sha256": digest(candidate_bytes),
        "evaluation_gold_sha256": digest(gold_bytes),
        "documents": len(kept),
        "labels": dict(counts),
        "skipped": dict(skipped),
        "sha256": digest(data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"built {len(kept)} contact snippets with {dict(counts)} labels")


if __name__ == "__main__":
    main()
