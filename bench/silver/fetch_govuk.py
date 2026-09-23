"""Fetch 2024 GOV.UK press releases: body text including the press office block.

Contains public sector information licensed under the Open Government Licence v3.0.
Run from the repository root:

    python bench/silver/fetch_govuk.py [limit]
"""

import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

SEARCH = "https://www.gov.uk/api/search.json"
CONTENT = "https://www.gov.uk/api/content{link}"
OUT = pathlib.Path("data/raw/silver/govuk")
HEADERS = {"User-Agent": "tessera-bench/0.1 (https://github.com/theiskaa/tessera)"}


def links(limit):
    start = 0
    while start < limit:
        r = requests.get(SEARCH, params={
            "filter_format": "press_release",
            "filter_public_timestamp": "from:2024-01-01,to:2024-12-31",
            "count": 100, "start": start, "fields": "link", "order": "public_timestamp",
        }, headers=HEADERS, timeout=60)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return
        for item in results:
            yield item["link"]
        start += len(results)
        time.sleep(1)


BLOCKS = ["p", "li", "h2", "h3", "h4", "td", "th", "pre", "address"]


def body_text(body_html):
    """Block elements become paragraphs; inline elements such as `abbr` stay in their line."""
    soup = BeautifulSoup(body_html, "html.parser")
    out = []
    for block in soup.find_all(BLOCKS):
        if block.find(BLOCKS):
            continue
        text = re.sub(r"[ \t]+", " ", block.get_text()).strip()
        text = "\n".join(l.strip() for l in text.splitlines() if l.strip())
        if text:
            out.append(text)
    return "\n\n".join(out)


def fetch(link):
    """Writes one press release; returns whether it was kept."""
    name = link.strip("/").replace("/", "_")
    path = OUT / f"{name}.json"
    if path.exists():
        return True
    r = requests.get(CONTENT.format(link=link), headers=HEADERS, timeout=60)
    time.sleep(1)
    if r.status_code != 200:
        return False
    item = r.json()
    text = body_text(item.get("details", {}).get("body", ""))
    if len(text) < 200:
        return False
    path.write_text(json.dumps({
        "id": name, "source": "govuk", "url": f"https://www.gov.uk{link}",
        "date": (item.get("first_published_at") or "")[:10], "text": text,
    }, ensure_ascii=False))
    return True


def main(limit=2000):
    OUT.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    listed = links(limit * 2)
    # Three requests at a time, each followed by a one-second pause.
    with ThreadPoolExecutor(max_workers=3) as pool:
        while kept < limit:
            batch = [link for _, link in zip(range(min(30, limit - kept)), listed)]
            if not batch:
                break
            for ok in pool.map(fetch, batch):
                kept += ok
                skipped += not ok
            print(f"govuk: {kept} kept, {skipped} skipped", flush=True)
    print(f"govuk: {kept} kept, {skipped} skipped")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
