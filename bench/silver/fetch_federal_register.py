"""Fetch 2024 Federal Register notices and keep their contact-bearing sections.

The Federal Register API lists the notices; the text comes from govinfo.gov, the Government
Publishing Office's copy, because federalregister.gov answers automated full-text requests
with a bot check. Federal Register documents are US government works in the public domain.
Run from the repository root:

    python bench/silver/fetch_federal_register.py [limit]
"""

import html
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://www.federalregister.gov/api/v1/documents.json"
GOVINFO = "https://www.govinfo.gov/content/pkg/FR-{date}/html/{number}.htm"
OUT = pathlib.Path("data/raw/silver/federal-register")
SECTIONS = ("ADDRESSES", "FOR FURTHER INFORMATION CONTACT", "SUPPLEMENTARY INFORMATION")
HEADERS = {"User-Agent": "tessera-bench/0.1 (https://github.com/theiskaa/tessera)"}


def pages(limit):
    got = 0
    url, params = API, {
        "conditions[type][]": "NOTICE",
        "conditions[publication_date][year]": "2024",
        "per_page": 100,
        "fields[]": ["document_number", "publication_date", "html_url"],
    }
    while url and got < limit:
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        body = r.json()
        for doc in body.get("results", []):
            yield doc
            got += 1
            if got >= limit:
                return
        url, params = body.get("next_page_url"), None
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
            if name == "SUPPLEMENTARY INFORMATION":
                body = body[:2000]
            out.append(f"{name}:\n{body}")
    return "\n\n".join(out)


def fetch(doc):
    """Writes one notice's contact sections; returns whether it was kept."""
    path = OUT / f"{doc['document_number']}.json"
    if path.exists():
        return True
    url = GOVINFO.format(date=doc["publication_date"], number=doc["document_number"])
    r = requests.get(url, headers=HEADERS, timeout=60)
    time.sleep(0.5)
    if r.status_code != 200:
        return False
    text = extract(page_text(r.text))
    if len(text) < 200:
        return False
    path.write_text(json.dumps({
        "id": doc["document_number"], "source": "federal-register", "url": doc["html_url"],
        "text_url": url, "date": doc["publication_date"], "text": text,
    }, ensure_ascii=False))
    return True


def main(limit=3000):
    OUT.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    # Many notices (FERC filing lists) have no contact sections, so pages are listed until
    # `limit` are kept; the API stops at 10,000 results per query. Three requests at a time,
    # each followed by a half-second pause, keep the load on govinfo.gov modest.
    listed = pages(10_000)
    with ThreadPoolExecutor(max_workers=3) as pool:
        while kept < limit:
            batch = [d for _, d in zip(range(30), listed)]
            if not batch:
                break
            for ok in pool.map(fetch, batch):
                kept += ok
                skipped += not ok
            print(f"federal register: {kept} kept, {skipped} skipped", flush=True)
    print(f"federal register: {kept} kept, {skipped} skipped")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
