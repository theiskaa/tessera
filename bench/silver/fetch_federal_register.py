"""Fetch Federal Register notices and keep their contact-bearing sections.

The Federal Register API lists the notices; the text comes from govinfo.gov, the Government
Publishing Office's copy, because federalregister.gov answers automated full-text requests
with a bot check. Federal Register documents are US government works in the public domain.
Run from the repository root:

    python bench/silver/fetch_federal_register.py [limit] --year 2025
    python bench/silver/fetch_federal_register.py 250 --year 2025 --all-months
    python bench/silver/fetch_federal_register.py [limit] --year 2024 \
        --out data/raw/silver/federal-register-full-2024

With --all-months, the limit applies separately to each month.
"""

import argparse
import calendar
import html
import json
import os
import pathlib
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://www.federalregister.gov/api/v1/documents.json"
GOVINFO = "https://www.govinfo.gov/content/pkg/FR-{date}/html/{number}.htm"
OUT = pathlib.Path("data/raw/silver/federal-register")
SECTIONS = ("ADDRESSES", "FOR FURTHER INFORMATION CONTACT", "SUPPLEMENTARY INFORMATION")
HEADERS = {"User-Agent": "tessera-bench/0.1 (https://github.com/theiskaa/tessera)"}


def get(url):
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=60) as response:
        return response.read()


def pages(limit, year, month=None):
    got = 0
    params = {
        "conditions[type][]": "NOTICE",
        "per_page": 100,
        "fields[]": ["document_number", "publication_date", "html_url"],
    }
    if month is None:
        params["conditions[publication_date][year]"] = str(year)
    else:
        last_day = calendar.monthrange(year, month)[1]
        params["conditions[publication_date][gte]"] = f"{year:04d}-{month:02d}-01"
        params["conditions[publication_date][lte]"] = f"{year:04d}-{month:02d}-{last_day:02d}"
    expected_date = f"{year:04d}-{month:02d}-" if month else f"{year:04d}-"
    url = f"{API}?{urlencode(params, doseq=True)}"
    while url and got < limit:
        body = json.loads(get(url))
        for doc in body.get("results", []):
            if not doc["publication_date"].startswith(expected_date):
                raise ValueError(f"document outside requested date range: {doc['document_number']}")
            yield doc
            got += 1
            if got >= limit:
                return
        url = body.get("next_page_url")
        time.sleep(1)


def decode_cfemail(hexed):
    """Cloudflare's email obfuscation: the first byte is a key XORed into the rest."""
    key = int(hexed[:2], 16)
    return "".join(chr(int(hexed[i:i + 2], 16) ^ key) for i in range(2, len(hexed), 2))


# GPO locator codes that are not HTML entity names; `[revaps]` is the Hawaiian ʻokina in names
# such as Kauaʻi. Editorial brackets ([sic], [reserved]) are not in either table and stay.
GPO_CHARS = {"supreg": "®", "revaps": "ʻ", "cir": "○", "ssquf": "▪", "ibreve": "ĭ", "hyphen": "-"}


def gpo_entities(text):
    """Replaces GPO's bracketed character codes (`Mu[ntilde]oz`) with the characters."""
    def char(m):
        name = m.group(1)
        if name in GPO_CHARS:
            return GPO_CHARS[name]
        c = html.unescape(f"&{name};")
        return c if len(c) == 1 else m.group(0)
    return re.sub(r"\[([a-z]{2,8})\]", char, text)


def page_text(raw_html):
    """The notice's plain text, with obfuscated emails restored and paragraphs unwrapped."""
    raw_html = re.sub(
        r'<(a|span)[^>]*data-cfemail="([0-9a-f]+)"[^>]*>.*?</\1>',
        lambda m: decode_cfemail(m.group(2)),
        raw_html,
        flags=re.S,
    )
    text = gpo_entities(html.unescape(re.sub(r"<[^>]+>", "", raw_html)).replace("\xa0", " "))
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        joined = lines[0]
        for line in lines[1:]:
            # GPO wraps at a hyphen without splitting words, so the hyphen belongs to the word.
            joined += line if joined.endswith("-") else " " + line
        paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def extract(text):
    out = []
    for name in SECTIONS:
        m = re.search(rf"{name}:\s*(.*?)(?=\n\n[A-Z][A-Z ]{{3,}}:|\Z)", text, re.S)
        if m:
            body = m.group(1).strip()
            out.append(f"{name}:\n{body}")
    return "\n\n".join(out)


def fetch(doc, out=OUT):
    """Writes one notice's contact sections; returns whether it was kept."""
    path = out / f"{doc['document_number']}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if (row["id"] != doc["document_number"] or
                row["url"] != doc["html_url"] or
                row["date"] != doc["publication_date"] or
                row["source"] != "federal-register" or
                row["text_url"] != GOVINFO.format(
                    date=doc["publication_date"], number=doc["document_number"]) or
                len(row["text"]) < 200):
            raise ValueError(f"existing source metadata differs: {path}")
        return True
    url = GOVINFO.format(date=doc["publication_date"], number=doc["document_number"])
    try:
        raw_html = get(url).decode("utf-8")
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    time.sleep(0.5)
    text = extract(page_text(raw_html))
    if len(text) < 200:
        return False
    row = {
        "id": doc["document_number"], "source": "federal-register", "url": doc["html_url"],
        "text_url": url, "date": doc["publication_date"], "text": text,
    }
    descriptor, temporary = tempfile.mkstemp(
        dir=out, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def main(limit=3000, year=2024, out=OUT, month=None):
    out.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    # Many notices (FERC filing lists) have no contact sections, so pages are listed until
    # `limit` are kept; the API stops at 10,000 results per query. Three requests at a time,
    # each followed by a half-second pause, keep the load on govinfo.gov modest.
    listed = pages(10_000, year, month)
    with ThreadPoolExecutor(max_workers=3) as pool:
        while kept < limit:
            batch = [d for _, d in zip(range(min(30, limit - kept)), listed)]
            if not batch:
                break
            for ok in pool.map(partial(fetch, out=out), batch):
                kept += ok
                skipped += not ok
            print(f"federal register {year}-{month or 'all'}: "
                  f"{kept} kept, {skipped} skipped", flush=True)
    print(f"federal register {year}-{month or 'all'}: {kept} kept, {skipped} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limit", type=int, nargs="?", default=3000)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    dates = parser.add_mutually_exclusive_group()
    dates.add_argument("--month", type=int)
    dates.add_argument("--all-months", action="store_true")
    args = parser.parse_args()
    if (args.limit < 1 or not 1995 <= args.year <= 2100 or
            args.month is not None and not 1 <= args.month <= 12):
        parser.error("limit must be positive, year 1995-2100, and month 1-12")
    if args.all_months:
        for month in range(1, 13):
            main(args.limit, args.year, args.out, month)
    else:
        main(args.limit, args.year, args.out, args.month)
