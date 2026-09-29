"""Capture a source-disjoint January–August 2026 US contact candidate pool."""

import argparse
import calendar
import collections
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urlparse

from fetch_federal_register import API, fetch, get


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/us"))
OUT = ROOT / "data/raw/candidates/federal-register-2026-contacts-v1"
LISTING = OUT / "listing"
SOURCES = OUT / "sources"
MANIFEST = OUT / "manifest.json"
MONTHS = tuple(range(1, 9))
PER_MONTH = 250
PER_PAGE = 100
AGENCY_MONTH_CAP = 20
AGENCY_YEAR_CAP = 60
FAMILY_MONTH_CAP = 4
FAMILY_YEAR_CAP = 16
TEMPLATE_MONTH_CAP = 20
TEMPLATE_YEAR_CAP = 100
SEED = "tessera-us-contact-2026-v1"
SAFE_NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
ORDINARY_NUMBER = re.compile(r"20\d{2}-\d{5}$")
FIELDS = ("document_number", "publication_date", "html_url", "agencies")


def digest(data):
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    """Atomically write a frozen artifact, or verify its existing bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"frozen artifact differs: {path}")
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value):
    """Serialize a manifest deterministically."""
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def api_url(month):
    """Build the complete monthly NOTICE listing query."""
    params = {
        "conditions[type][]": "NOTICE",
        "conditions[publication_date][gte]": f"2026-{month:02d}-01",
        "conditions[publication_date][lte]": (
            f"2026-{month:02d}-{calendar.monthrange(2026, month)[1]:02d}"
        ),
        "per_page": PER_PAGE,
        "fields[]": FIELDS,
    }
    return f"{API}?{urlencode(params, doseq=True)}"


def validate_document(row, month):
    """Check the listing's source identity and publication month."""
    source_id = row.get("document_number")
    date = row.get("publication_date")
    url = row.get("html_url")
    if (not isinstance(source_id, str) or not SAFE_NUMBER.fullmatch(source_id) or
            not isinstance(date, str) or not date.startswith(f"2026-{month:02d}-") or
            not isinstance(url, str) or
            not url.startswith(
                f"https://www.federalregister.gov/documents/{date.replace('-', '/')}/"
                f"{source_id}/") or
            not isinstance(row.get("agencies"), list)):
        raise ValueError(f"2026-{month:02d} listing contains invalid notice: {source_id}")


def read_listing(month, allow_network):
    """Replay a frozen monthly listing or save every official API page."""
    directory = LISTING / f"2026-{month:02d}"
    monthly_manifest = directory / "manifest.json"
    if monthly_manifest.exists():
        metadata = json.loads(monthly_manifest.read_text())
        pages = []
        for page in metadata["pages"]:
            path = directory / page["file"]
            body = path.read_bytes()
            if digest(body) != page["sha256"]:
                raise ValueError(f"frozen API page changed: {path}")
            pages.append(json.loads(body))
        rows = [row for page in pages for row in page["results"]]
        for row in rows:
            validate_document(row, month)
        ids = [row["document_number"] for row in rows]
        if (len(rows) != metadata["documents"] or len(ids) != len(set(ids)) or
                digest(json_bytes(ids)) != metadata["sha256"]):
            raise ValueError(f"frozen listing count changed: 2026-{month:02d}")
        return rows, metadata
    if not allow_network:
        raise ValueError(f"2026-{month:02d} API listing has not been captured")
    next_url = api_url(month)
    records = []
    pages = []
    expected_count = None
    while next_url:
        parsed = urlparse(next_url)
        if (parsed.scheme != "https" or parsed.netloc != "www.federalregister.gov" or
                parsed.path not in {"/api/v1/documents.json", "/api/v1/documents"}):
            raise ValueError(f"API pagination changed endpoint: {next_url}")
        filename = f"page-{len(pages) + 1:03d}.json"
        page_path = directory / filename
        cached = page_path.exists()
        body = page_path.read_bytes() if cached else get(next_url)
        page = json.loads(body)
        if not isinstance(page.get("results"), list):
            raise ValueError(f"API page lacks results: {next_url}")
        if expected_count is None:
            expected_count = page.get("count")
        elif page.get("count") != expected_count:
            raise ValueError(f"API listing count changed during 2026-{month:02d} capture")
        write_once(page_path, body)
        pages.append({"file": filename, "url": next_url,
                      "sha256": digest(body), "results": len(page["results"])})
        records.extend(page["results"])
        next_url = page.get("next_page_url")
        if next_url and not cached:
            time.sleep(1)
    ids = []
    for row in records:
        validate_document(row, month)
        ids.append(row["document_number"])
    if (len(ids) != len(set(ids)) or
            isinstance(expected_count, int) and len(ids) != expected_count):
        raise ValueError(f"2026-{month:02d} API listing is incomplete or duplicated")
    metadata = {"month": f"2026-{month:02d}", "query": api_url(month),
                "expected_count": expected_count, "documents": len(records),
                "pages": pages, "sha256": digest(json_bytes(ids))}
    write_once(monthly_manifest, json_bytes(metadata))
    return records, metadata


def source_family(url):
    """Group notices with the same leading title words."""
    slug = urlparse(url).path.rsplit("/", 1)[-1]
    return "-".join(slug.split("-")[:8])


def agency(row):
    """Use the most specific issuing agency in the API listing."""
    agencies = row["agencies"]
    if not agencies:
        return None
    item = agencies[-1]
    if not isinstance(item, dict):
        return None
    return item.get("slug") or item.get("raw_name")


def template(row):
    """Identify the repeated repatriation notice layout."""
    slug = row["html_url"].rsplit("/", 1)[-1]
    return slug.startswith(("notice-of-inventory-completion",
                            "notice-of-intended-repatriation"))


def historical_sources(*, frozen_paths=None):
    """Pin old review and silver sources before selecting new notices."""
    from active_sources import ACTIVE_SILVER, BASE_SILVER
    from build_contact_snippets import lines

    review_raw = ROOT / "data/raw/review/federal-register"
    if frozen_paths is None:
        paths = [*sorted(review_raw.glob("2026-*.json")),
                 *sorted((ROOT / "data/interim/review").rglob("*.jsonl")),
                 *sorted((ROOT / "data/interim/silver/r23").glob("*.jsonl")),
                 *sorted((ROOT / "data/interim/silver/r24").glob("*.jsonl"))]
        names = {name for name, _ in (*BASE_SILVER, *ACTIVE_SILVER)}
        paths.extend(ROOT / f"data/interim/silver/{name}.jsonl" for name in sorted(names))
    else:
        paths = []
        for relative in frozen_paths:
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or path.suffix not in {".json", ".jsonl"}:
                raise ValueError(f"invalid frozen source-use path: {relative}")
            paths.append(path)
    ids = set()
    families = set()
    hashes = {}
    for path in paths:
        if not path.exists():
            raise ValueError(f"source-use input missing: {path}")
        hashes[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        rows = [json.loads(path.read_text())] if path.parent == review_raw else lines(path)
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("source_id", "name", "id", "source_url", "url"):
                value = row.get(key)
                if isinstance(value, str):
                    ids.update(re.findall(r"20\d{2}-\d{5}", value))
            url = row.get("source_url") or row.get("url")
            if isinstance(url, str) and url.startswith("https://www.federalregister.gov/documents/"):
                families.add(source_family(url))
    if sum(path.parent == review_raw for path in paths) != 150:
        raise ValueError("reserved 2026 Federal Register review capture changed")
    return ids, families, dict(sorted(hashes.items()))


def ranked_month(rows, month):
    """Return a stable hash order independent of API page ordering."""
    return sorted(rows, key=lambda row: (
        digest(f"{SEED}:{month:02d}:{row['document_number']}".encode()),
        row["document_number"],
    ))


def selected_sources(allow_network):
    """Capture 250 source-disjoint notices for each completed month."""
    ids, families, input_hashes = historical_sources()
    captured = []
    counts = collections.Counter()
    agency_year = collections.Counter()
    family_year = collections.Counter()
    template_year = 0
    listing_hashes = {}
    SOURCES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for month in MONTHS:
            rows, listing = read_listing(month, allow_network)
            print(f"2026-{month:02d}: listing verified ({len(rows)} notices)", flush=True)
            listing_hashes[listing["month"]] = digest(
                (LISTING / listing["month"] / "manifest.json").read_bytes())
            counts["special_notices"] += sum(
                not ORDINARY_NUMBER.fullmatch(row["document_number"]) for row in rows)
            month_agencies = collections.Counter()
            month_families = collections.Counter()
            month_templates = 0
            kept = 0
            remaining = [row for row in ranked_month(rows, month)
                         if ORDINARY_NUMBER.fullmatch(row["document_number"]) and
                         row["document_number"] not in ids and
                         source_family(row["html_url"]) not in families]
            position = 0
            last_progress = 0
            while kept < PER_MONTH and position < len(remaining):
                batch = []
                provisional_agencies = month_agencies.copy()
                provisional_families = month_families.copy()
                provisional_templates = month_templates
                for row in remaining[position:]:
                    if len(batch) >= min(30, PER_MONTH - kept):
                        break
                    position += 1
                    group = source_family(row["html_url"])
                    owner = agency(row)
                    if not owner:
                        counts["unknown_agency"] += 1
                        continue
                    if (provisional_agencies[owner] >= AGENCY_MONTH_CAP or
                            agency_year[owner] + provisional_agencies[owner] - month_agencies[owner]
                            >= AGENCY_YEAR_CAP):
                        counts["agency_cap"] += 1
                        continue
                    if (provisional_families[group] >= FAMILY_MONTH_CAP or
                            family_year[group] + provisional_families[group] - month_families[group]
                            >= FAMILY_YEAR_CAP):
                        counts["family_cap"] += 1
                        continue
                    if template(row) and (provisional_templates >= TEMPLATE_MONTH_CAP or
                                          template_year + provisional_templates - month_templates
                                          >= TEMPLATE_YEAR_CAP):
                        counts["template_cap"] += 1
                        continue
                    batch.append(row)
                    provisional_agencies[owner] += 1
                    provisional_families[group] += 1
                    provisional_templates += template(row)
                if not batch:
                    break
                results = list(pool.map(lambda row: fetch(row, out=SOURCES), batch))
                for row, ok in zip(batch, results):
                    if not ok:
                        counts["missing_or_short_source"] += 1
                        continue
                    source_id = row["document_number"]
                    group = source_family(row["html_url"])
                    owner = agency(row)
                    raw_bytes = (SOURCES / f"{source_id}.json").read_bytes()
                    raw = json.loads(raw_bytes)
                    if (raw["id"] != source_id or raw["date"] != row["publication_date"] or
                            raw["url"] != row["html_url"] or len(raw["text"]) < 200):
                        raise ValueError(f"captured notice differs from API listing: {source_id}")
                    captured.append({"source_id": source_id, "month": listing["month"],
                                     "source_url": row["html_url"],
                                     "agency": owner, "source_group": group,
                                     "template": template(row), "raw_sha256": digest(raw_bytes)})
                    ids.add(source_id)
                    month_agencies[owner] += 1
                    agency_year[owner] += 1
                    month_families[group] += 1
                    family_year[group] += 1
                    month_templates += template(row)
                    template_year += template(row)
                    kept += 1
                progress = kept // 50
                if progress > last_progress:
                    print(f"2026-{month:02d}: {kept}/{PER_MONTH} sources captured", flush=True)
                    last_progress = progress
            if kept != PER_MONTH:
                raise ValueError(f"2026-{month:02d}: only {kept}/{PER_MONTH} notices passed capture caps")
            counts[f"kept_2026_{month:02d}"] = kept
    return captured, counts, input_hashes, listing_hashes


def verify_manifest():
    """Verify an existing candidate capture without querying the API."""
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["sources"]
    if (manifest["training_eligible"] or len(rows) != PER_MONTH * len(MONTHS) or
            len({row["source_id"] for row in rows}) != len(rows)):
        raise ValueError("candidate capture manifest is incomplete")
    for row in rows:
        path = SOURCES / f"{row['source_id']}.json"
        if digest(path.read_bytes()) != row["raw_sha256"]:
            raise ValueError(f"candidate source changed: {path}")
    for month, expected in manifest["listing_manifest_sha256"].items():
        if digest((LISTING / month / "manifest.json").read_bytes()) != expected:
            raise ValueError(f"candidate listing changed: {month}")
    for relative, expected in manifest["reserved_input_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"candidate source-use input changed: {relative}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="read one January API page only")
    parser.add_argument("--verify", action="store_true", help="verify frozen capture offline")
    args = parser.parse_args()
    if args.probe:
        body = get(api_url(1))
        page = json.loads(body)
        print(f"2026-01 NOTICE listing probe: {page.get('count')} results, "
              f"{len(page.get('results', []))} on first page")
        return
    if args.verify or MANIFEST.exists():
        manifest = verify_manifest()
        print(f"verified {len(manifest['sources'])} frozen 2026 candidate notices")
        return
    sources, counts, inputs, listings = selected_sources(allow_network=True)
    manifest = {
        "kind": "federal_register_2026_contact_source_candidates_v1",
        "intended_use": "blind_contact_screening_only",
        "training_eligible": False,
        "label_status": "unlabeled",
        "months": [f"2026-{month:02d}" for month in MONTHS],
        "per_month": PER_MONTH,
        "selection_seed": SEED,
        "agency_month_cap": AGENCY_MONTH_CAP,
        "agency_year_cap": AGENCY_YEAR_CAP,
        "family_month_cap": FAMILY_MONTH_CAP,
        "family_year_cap": FAMILY_YEAR_CAP,
        "template_month_cap": TEMPLATE_MONTH_CAP,
        "template_year_cap": TEMPLATE_YEAR_CAP,
        "reserved_input_sha256": inputs,
        "listing_manifest_sha256": listings,
        "selection": dict(sorted(counts.items())),
        "sources": sources,
    }
    write_once(MANIFEST, json_bytes(manifest))
    verify_manifest()
    print(f"captured {len(sources)} frozen 2026 candidate notices")


if __name__ == "__main__":
    main()
