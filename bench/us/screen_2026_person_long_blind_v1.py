"""Freeze source-distinct long notice passages for blind person review."""

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/silver"))
from fetch_federal_register_2026_candidates import verify_manifest


CAPTURE = ROOT / "data/raw/candidates/federal-register-2026-contacts-v1"
EVALUATION = ROOT / "data/interim/review/us-eval-exclusions-v4.jsonl"
SILVER = ROOT / "data/interim/silver"
OUT = ROOT / "data/raw/candidates/us-2026-person-long-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
CONTACT = "FOR FURTHER INFORMATION CONTACT:\n"
PERSON_CUE = re.compile(r"\b[A-Z][a-zÀ-ÿ’\-]+\s+(?:[A-Z]\.\s+)?[A-Z][a-zÀ-ÿ’\-]+\b")
REPLY_CUE = re.compile(r"@|\d{3}[-.) ]\d{3}")
HARD_NEGATIVES = (
    "Docket", "Notice", "Federal Register", "Affected Public", "OMB",
    "Executive Summary", "Expiration Date", "Hand Delivery",
    "Agency Website", "Comments", "Information Collection", "Regulation",
)


def digest(data):
    """Return the SHA-256 digest of captured bytes."""
    return hashlib.sha256(data).hexdigest()


def rows(path):
    """Read a JSONL input without changing row order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def shingles(text):
    """Find exact twelve-word fragments across source formats."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[index:index + 12])
            for index in range(len(words) - 11)}


def source_family(url):
    """Group Federal Register notices by their leading title words."""
    slug = url.rsplit("/", 1)[-1]
    return "-".join(slug.split("-")[:8])


def contact_passage(text):
    """Keep whole paragraphs from the first contact through long surrounding prose."""
    start = text.find(CONTACT)
    if start < 0:
        return None
    parts = text[start:].split("\n\n")
    selected = []
    words = 0
    for part in parts:
        next_words = words + len(part.split())
        if next_words > 500:
            break
        selected.append(part)
        words = next_words
        if words >= 300:
            break
    if not 200 <= words <= 500:
        return None
    passage = "\n\n".join(selected)
    end = start + len(passage)
    if text[start:end] != passage:
        raise ValueError("notice passage is not an exact source window")
    return passage, len(text[:start].encode()), len(text[:end].encode())


def selected_rows():
    """Screen unlabeled source passages against all frozen US split inputs."""
    capture = verify_manifest()
    evaluation = rows(EVALUATION)
    silver_paths = sorted(SILVER.glob("us-*.jsonl"))
    prior = evaluation + [row for path in silver_paths for row in rows(path)]
    prior_keys = {document_key(row) for row in prior}
    prior_families = {
        source_family(row["source_url"]) for row in prior
        if row.get("source_url", "").startswith(
            "https://www.federalregister.gov/documents/")
    }
    prior_shingles = set().union(*(
        shingles(row.get("input", row.get("text", ""))) for row in prior
    ))
    people = {
        span["text"] for row in evaluation for span in row["expected"]
        if span["kind"] == "person"
    }
    person_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(name)
                                       for name in sorted(people, key=len, reverse=True))
        + r")(?!\w)", re.IGNORECASE
    )
    counts = collections.Counter()
    candidates = []
    for source in capture["sources"]:
        source_id = source["source_id"]
        if not re.fullmatch(r"2026-\d{5}", source_id) or source["template"]:
            counts["other_year_or_template"] += 1
            continue
        if (f"fr:{source_id}" in prior_keys or
                source["source_group"] in prior_families):
            counts["prior_source_or_family"] += 1
            continue
        path = CAPTURE / "sources" / f"{source_id}.json"
        raw_bytes = path.read_bytes()
        if digest(raw_bytes) != source["raw_sha256"]:
            raise ValueError(f"captured source changed: {source_id}")
        raw = json.loads(raw_bytes)
        if (raw["id"] != source_id or raw["url"] != source["source_url"] or
                not raw["date"].startswith(source["month"])):
            raise ValueError(f"captured source identity changed: {source_id}")
        source_text = raw["text"]
        window = contact_passage(source_text)
        if window is None:
            counts["length_or_encoding"] += 1
            continue
        text, start_byte, end_byte = window
        words = len(text.split())
        if "\ufffd" in text:
            counts["length_or_encoding"] += 1
            continue
        if CONTACT not in text or "SUPPLEMENTARY INFORMATION:" not in text:
            counts["missing_sections"] += 1
            continue
        contact = text.split(CONTACT, 1)[1].split("SUPPLEMENTARY INFORMATION:", 1)[0]
        if not (5 <= len(contact.split()) <= 120 and
                PERSON_CUE.search(contact) and REPLY_CUE.search(contact)):
            counts["contact_without_person_cue"] += 1
            continue
        cues = tuple(cue for cue in HARD_NEGATIVES if cue in text)
        if not cues:
            counts["few_negative_cues"] += 1
            continue
        if person_pattern.search(text):
            counts["evaluation_person_surface"] += 1
            continue
        if shingles(text) & prior_shingles:
            counts["prior_twelve_word_fragment"] += 1
            continue
        candidates.append((source, text, words, cues, start_byte, end_byte,
                           digest(source_text.encode()), raw["text_url"]))
    candidates.sort(key=lambda item: (-len(item[3]), -item[2], item[0]["source_id"]))
    selected = []
    used_groups = set()
    selected_shingles = set()
    months = collections.Counter()
    agencies = collections.Counter()
    for source, text, words, cues, start_byte, end_byte, source_text_hash, text_url in candidates:
        if len(selected) == 20:
            break
        if (source["source_group"] in used_groups or
                months[source["month"]] >= 3 or
                agencies[source["agency"]] >= 3 or
                shingles(text) & selected_shingles):
            continue
        selected.append({
            "name": f"us-2026-person-long-{source['source_id']}",
            "country": "US", "input": text,
            "source": "federal-register", "source_id": source["source_id"],
            "source_url": source["source_url"],
            "source_document_key": f"fr:{source['source_id']}",
            "source_group": source["source_group"],
            "source_text_url": text_url,
            "raw_sha256": source["raw_sha256"],
            "source_text_sha256": source_text_hash,
            "source_start_byte": start_byte,
            "source_end_byte": end_byte,
            "month": source["month"], "agency": source["agency"],
        })
        used_groups.add(source["source_group"])
        selected_shingles.update(shingles(text))
        months[source["month"]] += 1
        agencies[source["agency"]] += 1
    selected.sort(key=lambda row: row["name"])
    if len(selected) != 20 or len(months) < 7 or len(agencies) < 15:
        raise ValueError(f"person packet lacks source diversity: {len(selected)}, "
                         f"{len(months)} months, {len(agencies)} agencies; "
                         f"{len(candidates)} candidates, {dict(counts)}")
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in (EVALUATION, *silver_paths)}
    return selected, counts, inputs, months, agencies


def main():
    """Freeze the blind packet and fail if any screened input changes."""
    selected, counts, inputs, months, agencies = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "us_2026_person_long_blind_v1",
        "training_eligible": False,
        "evaluation_eligible": False,
        "label_status": "unlabeled",
        "cases": len(selected),
        "source_documents": len({row["source_document_key"] for row in selected}),
        "source_groups": len({row["source_group"] for row in selected}),
        "source_agencies": len(agencies),
        "cases_by_month": dict(sorted(months.items())),
        "cases_by_agency": dict(sorted(agencies.items())),
        "screening": dict(sorted(counts.items())),
        "capture_manifest_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "reserved_input_sha256": inputs,
        "sha256": digest(data),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("person packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen person packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("person screening inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind long PERSON passages from {len(agencies)} agencies")


if __name__ == "__main__":
    main()
