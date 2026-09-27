"""Shared plumbing for the review-set fetchers: output layout, text clipping, HTML-to-text,
contact heuristics, and the polite three-worker fetch loop.

Documents land in data/raw/review/<source>/<id>.json with the fields id, source, url, date,
country, doc_type, text, and licence.
"""

import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench" / "silver"))

from fetch_federal_register import HEADERS, decode_cfemail, gpo_entities, page_text  # noqa: E402,F401

import requests  # noqa: E402

ROOT = REPO / "data" / "raw" / "review"
MAX_CHARS = 6000
WORKERS = 3

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone is a run of at least nine digits in the usual separators, optionally with a leading
# +country code; bare years and short reference numbers stay below that length.
PHONE = re.compile(r"(?<![\w/])(?:\+|00)?\(?\d[\d ()./-]{7,}\d(?![\w/])")
UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b")
PO_BOX = re.compile(r"\b(?:PO|P\.O\.) ?Box\b", re.I)


def banner(source, licence, terms):
    print(f"{source}: licence: {licence}", flush=True)
    print(f"{source}: terms: {terms}", flush=True)


def get(url, **kw):
    """One GET with the bench User-Agent, followed by the polite pause."""
    kw.setdefault("timeout", 60)
    kw.setdefault("headers", HEADERS)
    try:
        r = requests.get(url, **kw)
    except requests.RequestException:
        r = None
    time.sleep(0.75)
    return r


def safe_id(raw):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")[:150]


def path_for(source, doc_id):
    return ROOT / source / f"{safe_id(doc_id)}.json"


def exists(source, doc_id):
    return path_for(source, doc_id).exists()


def write(doc):
    path = path_for(doc["source"], doc["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {k: doc[k] for k in ("id", "source", "url", "date", "country", "doc_type", "text",
                                "licence")} | {k: v for k, v in doc.items() if k.startswith("x_")}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))


def normalize(text):
    """Blank-line-separated paragraphs, trailing spaces trimmed, no runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clip(text, limit=MAX_CHARS):
    """At most `limit` characters, cut at the last paragraph boundary that fits."""
    if len(text) <= limit:
        return text
    out = ""
    for para in text.split("\n\n"):
        nxt = f"{out}\n\n{para}" if out else para
        if len(nxt) > limit:
            break
        out = nxt
    return out or text[:limit].rsplit("\n", 1)[0]


def has_contact(text):
    return bool(EMAIL.search(text) or has_phone(text) or UK_POSTCODE.search(text)
                or PO_BOX.search(text))


# Dashes and digits Japanese text writes phones with: `03−5253−5111`, `０３－５２５３`.
WIDE = str.maketrans({**{c: "-" for c in "\u2212\uff0d\u2015\u30fc\u2010\u2013\u2014"},
                      **{chr(0xFF10 + d): str(d) for d in range(10)}, "（": "(", "）": ")"})


def has_phone(text):
    return any(len(re.sub(r"\D", "", m)) >= 9 for m in PHONE.findall(text.translate(WIDE)))


BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "tr", "dt", "dd",
              "address", "div", "section", "article", "header", "blockquote", "pre", "ul",
              "ol", "dl", "table", "figure", "figcaption", "summary", "details", "aside"}


def block_text(element):
    """Visible text of a BeautifulSoup element: block elements become paragraphs, `<br>`
    becomes a line break, and source-code whitespace inside a paragraph collapses to one
    space. The element is modified in place."""
    from bs4 import NavigableString

    for tag in element.find_all(["script", "style", "noscript", "template", "svg", "button",
                                 "input", "select", "form"]):
        tag.decompose()
    for node in element.find_all(string=True):
        if node.find_parent("pre") is None and type(node) is NavigableString:
            node.replace_with(re.sub(r"\s+", " ", str(node)))
    for br in element.find_all("br"):
        br.replace_with("\n")
    for tag in element.find_all(BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    text = element.get_text("")
    paras = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            paras.append("\n".join(lines))
    return "\n\n".join(paras)


def run(label, candidates, fetch, limit):
    """Feeds `candidates` to `fetch` three at a time until `limit` documents are kept.

    `fetch` returns True when it wrote a document, False when it skipped the candidate, and
    None when the document was already written by an earlier run; those count from the start,
    so a resumed run stops at `limit` documents in total."""
    kept = sum(1 for _ in (ROOT / label).glob("*.json"))
    skipped = 0
    it = iter(candidates)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while kept < limit:
            batch = [c for _, c in zip(range(min(WORKERS * 4, limit - kept)), it)]
            if not batch:
                break
            for ok in pool.map(fetch, batch):
                kept += ok is True
                skipped += ok is False
            print(f"{label}: {kept} kept, {skipped} skipped", flush=True)
    print(f"{label}: done, {kept} kept, {skipped} skipped", flush=True)
    return kept


BOILERPLATE = re.compile(r"(^|[-_ ])(menu|nav|navbar|header|footer|sidebar|breadcrumbs?|share|social|"
                         r"cookie|banner|pager|pagination|skip)([-_ ]|$)", re.I)


def content_text(soup):
    """The main content of a page without a known container: header, footer, navigation, and
    boilerplate-classed elements are dropped, then the deepest element holding at least 60% of
    the remaining text outside links is taken."""
    for tag in soup.find_all(["script", "style", "noscript", "template", "header", "footer",
                              "nav", "aside"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        names = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if BOILERPLATE.search(names) and tag.name not in ("body", "html", "main"):
            tag.decompose()
    body = soup.body or soup

    def own(el):
        total = len(el.get_text(" ", strip=True))
        links = sum(len(a.get_text(" ", strip=True)) for a in el.find_all("a"))
        return total - links

    whole = own(body) or 1
    best = body
    while True:
        child = max((c for c in best.find_all(True, recursive=False)), key=own, default=None)
        if child is None or own(child) < 0.6 * whole:
            break
        best = child
    return block_text(best)


def tail_clip(text, head=1500, limit=MAX_CHARS):
    """At most `limit` characters: the paragraphs of the first `head` characters and as many
    of the last ones as fit, so a contact block at the end of a long release survives."""
    if len(text) <= limit:
        return text
    paras = text.split("\n\n")
    first, used = [], 0
    for p in paras:
        if used + len(p) > head:
            break
        first.append(p)
        used += len(p) + 2
    last = []
    for p in reversed(paras[len(first):]):
        if used + len(p) + 2 > limit:
            break
        last.insert(0, p)
        used += len(p) + 2
    return "\n\n".join(first + last)
