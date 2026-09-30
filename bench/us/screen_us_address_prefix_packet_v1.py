"""Freeze source-distinct Federal Register address passages for blind review."""

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from address_keys import POSTCODE, address_keys_in_text
from build_contact_snippets import excluded_surface_pattern, lines
from build_silver import normalize_surface
from check_us_next_data_split import ADDITIONS, EVALUATION, REPLACEMENTS
from freeze_2025_contact_packet import ARTIFACT, EMAIL, PHONE
from freeze_full_notice_packet import has_person_alias, person_alias_patterns
from screen_2026_contact_expansion_v3 import CAPTURE, PRIOR_PACKETS, shingles
from screen_us_error_target_contacts import MALFORMED_MARKUP
from silver_dedupe import NearTextIndex
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/candidates/federal-register-us-address-prefix-v1/blind-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
PERSON_PACKET = ROOT / "data/raw/candidates/us-2026-person-long-v1/blind-v1.jsonl"
PREFIX = re.compile(
    r"\b(?:Room\s+[A-Z0-9/-]*\d[A-Z0-9/-]*|Rm\.?\s+[A-Z0-9/-]*\d[A-Z0-9/-]*|"
    r"(?:Suite|Ste\.?)\s+[A-Z0-9/-]*\d[A-Z0-9/-]*|Mail\s+(?:Code|Stop)\s+[A-Z0-9/-]+|"
    r"MS\s*:\s*[A-Z0-9/()-]+|Bldg\.?\s+[A-Z0-9/-]+|"
    r"(?:[A-Z][A-Za-z.&'-]*\s+){1,5}(?:Building|Center|Courthouse|Complex|Hall)|"
    r"(?:Ground|\d+(?:st|nd|rd|th))\s+Floor)", re.I
)
CATEGORIES = ("room_suite_mail", "building_venue")
MAX_CASES = 28
MAX_PER_MONTH = 5
MAX_PER_AGENCY = 3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def address_excerpts(paragraph):
    """Keep a contiguous source clause ending at its physical address."""
    for match in PREFIX.finditer(paragraph):
        address_end = None
        for postcode in POSTCODE.finditer(paragraph, match.end()):
            if postcode.end() > match.end() + 220:
                break
            if address_keys_in_text(paragraph[match.start():postcode.end()]):
                address_end = postcode.end()
            break
        if address_end is None:
            continue
        boundaries = [point.end() for point in re.finditer(
            r"(?:[.;:]\s+|\n)", paragraph[max(0, match.start() - 220):match.start()])]
        start = max(0, match.start() - 220) + max(boundaries, default=0)
        if start and paragraph[start - 1:start] == " ":
            while start < match.start() and paragraph[start].isspace():
                start += 1
        if start > match.start() or (start == 0 and match.start() > 220):
            continue
        end = address_end + (paragraph[address_end:address_end + 1] == ".")
        excerpt = paragraph[start:end]
        if not 10 <= len(excerpt.split()) <= 100:
            continue
        category = ("room_suite_mail" if re.match(
            r"(?i)(?:room|rm\.?|suite|ste\.?|mail\s+(?:code|stop)|ms\s*:|bldg\.?)", match[0])
                    else "building_venue")
        yield start, end, excerpt, category


def selected_rows():
    """Replay all pinned source, split, and overlap checks."""
    sys.path.insert(0, str(ROOT / "bench/silver"))
    from fetch_federal_register_2026_candidates import source_family, verify_manifest

    capture = verify_manifest()
    if capture["training_eligible"] or len(capture["sources"]) != 2000:
        raise ValueError("2026 Federal Register capture changed")
    config = (ROOT / "configs/detector-shared-v4.toml").read_text()
    match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", config)
    if match is None:
        raise ValueError("V4 silver list is missing")
    configured = json.loads(match[1])
    active_paths = [ROOT / f"data/interim/silver/"
                    f"{REPLACEMENTS.get(Path(path).stem, Path(path).stem)}.jsonl"
                    for path in configured]
    active_paths += [ROOT / f"data/interim/silver/{name}.jsonl" for name in ADDITIONS]
    active = [row for path in active_paths for row in lines(path)]
    evaluation = [row for path in EVALUATION for row in lines(path)]
    if len(active) != 1749 or len(evaluation) != 371:
        raise ValueError("proposed training or evaluation membership changed")
    prior_paths = [*PRIOR_PACKETS, PERSON_PACKET]
    prior_paths += sorted((ROOT / "data/raw/candidates").glob("federal-register-*/blind-*.jsonl"))
    prior_paths = sorted(set(prior_paths) - {OUT})
    prior = [row for path in prior_paths for row in lines(path)]
    blocked_keys = {document_key(row) for row in evaluation + active + prior}
    blocked_groups = {row.get("source_group") for row in active + prior}
    blocked_groups.discard(None)
    forbidden = excluded_surface_pattern(evaluation)
    people = person_alias_patterns(evaluation)
    addresses = set().union(*(address_keys_in_text(span["text"])
                              for row in evaluation for span in row["expected"]
                              if span["kind"] == "address"))
    streets = {key[1:] for key in addresses}
    eval_prose = set().union(*(shingles(row["input"], 12) for row in evaluation))
    earlier = [(row["name"], row["input"]) for row in evaluation + prior]
    earlier += [(row["id"], row["text"]) for row in active]
    near = NearTextIndex()
    for name, text in earlier:
        near.add(name, text)
    earlier_emails = {match.casefold() for _, text in earlier for match in EMAIL.findall(text)}
    earlier_phones = {re.sub(r"\D", "", match) for _, text in earlier
                      for match in PHONE.findall(text)}
    counts = collections.Counter()
    candidates = []
    historical = ROOT / "data/raw/silver/federal-register"
    sources = list(capture["sources"])
    for path in sorted(historical.glob("*.json")):
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        sources.append({
            "source_id": raw["id"], "source_url": raw["url"],
            "source_group": source_family(raw["url"]),
            "raw_sha256": digest(raw_bytes), "month": raw["date"][:7],
            "agency": source_family(raw["url"]),
            "raw_path": str(path.relative_to(ROOT)),
        })
    for source in sources:
        source_id = source["source_id"]
        raw_path = ROOT / source.get(
            "raw_path", f"data/raw/candidates/federal-register-2026-contacts-v1/sources/{source_id}.json")
        raw_bytes = raw_path.read_bytes()
        if digest(raw_bytes) != source["raw_sha256"]:
            raise ValueError(f"Federal Register raw source changed: {source_id}")
        raw = json.loads(raw_bytes)
        if (raw.get("id") != source_id or raw.get("url") != source["source_url"] or
                raw.get("date", "")[:7] != source["month"] or
                source_family(raw["url"]) != source["source_group"]):
            raise ValueError(f"Federal Register source metadata changed: {source_id}")
        if (document_key({"source_id": source_id}) in blocked_keys or
                source["source_group"] in blocked_groups):
            counts["reserved_source"] += 1
            continue
        for number, paragraph in enumerate(raw["text"].split("\n\n"), 1):
            if not paragraph.startswith(("ADDRESSES:", "FOR FURTHER INFORMATION CONTACT:")):
                continue
            counts["contact_paragraph"] += 1
            if (not 20 <= len(paragraph.split()) <= 220 or
                    not paragraph.rstrip().endswith((".", "!", "?")) or
                    ARTIFACT.search(paragraph) or MALFORMED_MARKUP.search(paragraph) or
                    raw["text"].count(paragraph) != 1):
                counts["boundary_or_artifact"] += 1
                continue
            excerpts = list(address_excerpts(paragraph))
            if not excerpts:
                counts["no_prefix_address"] += 1
                continue
            paragraph_start = raw["text"].find(paragraph)
            if paragraph_start < 0 or raw["text"].count(paragraph) != 1:
                raise ValueError(f"Federal Register paragraph moved: {source_id}")
            for start, end, excerpt, category in excerpts:
                if (forbidden.search(normalize_surface(excerpt)) or
                        has_person_alias(excerpt, people)):
                    counts["evaluation_name"] += 1
                    continue
                keys = address_keys_in_text(excerpt)
                if keys & addresses or {key[1:] for key in keys} & streets:
                    counts["evaluation_street"] += 1
                    continue
                if shingles(excerpt, 12) & eval_prose:
                    counts["evaluation_prose"] += 1
                    continue
                emails = {match.casefold() for match in EMAIL.findall(excerpt)}
                phones = {re.sub(r"\D", "", match) for match in PHONE.findall(excerpt)}
                if emails & earlier_emails or phones & earlier_phones:
                    counts["earlier_contact"] += 1
                    continue
                if near.prior(excerpt):
                    counts["earlier_near_text"] += 1
                    continue
                source_start = len(raw["text"][:paragraph_start + start].encode())
                source_end = len(raw["text"][:paragraph_start + end].encode())
                if raw["text"].encode()[source_start:source_end].decode() != excerpt:
                    raise ValueError(f"Federal Register byte window moved: {source_id}")
                candidates.append({
                    "name": f"us-address-prefix-{source_id}-{number}-{start}",
                    "country": "US", "source_id": source_id,
                    "source_url": source["source_url"],
                    "source_group": source["source_group"],
                    "raw_sha256": source["raw_sha256"], "input": excerpt,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "raw_text_start": source_start, "raw_text_end": source_end,
                    "month": source["month"], "agency": source["agency"],
                    "categories": [category],
                })
    ranked = sorted(candidates, key=lambda row: (
        len(row["categories"]) != 1,
        digest(f"tessera-us-address-prefix-v1:{row['name']}".encode()),
    ))
    selected = []
    groups = set()
    selected_addresses = set()
    months = collections.Counter()
    agencies = collections.Counter()
    category_counts = collections.Counter()
    for category in CATEGORIES:
        for row in ranked:
            if len(selected) >= MAX_CASES or category_counts[category] >= MAX_CASES // 2:
                break
            if (category not in row["categories"] or row["source_group"] in groups or
                    months[row["month"]] >= MAX_PER_MONTH or
                    agencies[row["agency"]] >= MAX_PER_AGENCY or near.prior(row["input"])):
                continue
            keys = address_keys_in_text(row["input"])
            if keys & selected_addresses:
                continue
            selected.append(row)
            groups.add(row["source_group"])
            selected_addresses.update(keys)
            months[row["month"]] += 1
            agencies[row["agency"]] += 1
            category_counts[category] += 1
            near.add(row["name"], row["input"])
    if len(selected) < 20 or len(months) < 5 or min(category_counts.values()) < 7:
        raise ValueError(f"too few independent address passages: {len(selected)}, "
                         f"{dict(category_counts)}, {dict(months)}, {dict(counts)}")
    selected.sort(key=lambda row: row["name"])
    inputs = {str(path.relative_to(ROOT)): digest(path.read_bytes())
              for path in [*EVALUATION, *active_paths, *prior_paths]}
    return selected, counts, category_counts, inputs


def main():
    """Write once, then require byte-for-byte replay."""
    selected, counts, categories, inputs = selected_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in selected).encode()
    manifest = {
        "kind": "federal_register_us_address_prefix_blind_v1",
        "training_eligible": False,
        "evaluation_eligible": False,
        "label_status": "unlabeled",
        "cases": len(selected),
        "source_group_key": "source_group",
        "capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "input_sha256": inputs,
        "selection": dict(sorted(counts.items())),
        "categories": dict(sorted(categories.items())),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("address packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("address packet changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("address packet inputs changed after freezing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind US address passages: {dict(categories)}, "
          f"{len({row['source_group'] for row in selected})} source groups")


if __name__ == "__main__":
    main()
