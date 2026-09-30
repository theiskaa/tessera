"""Freeze source-pinned, evaluation-disjoint long US person passages."""

import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import ADDITIONS, CONFIG, EVALUATION, REPLACEMENTS, ROOT, rows
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key


CAPTURE = ROOT / "data/raw/candidates/us-person-long-v2"
INDEX = CAPTURE / "capture-index.json"
OUT = CAPTURE / "blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
BLOCK_WINDOWS = {
    "fsu-research-professors": (2, 8),
    "kingcounty-taylor": (5, 10),
    "kstate-hitzler": (2, 8),
    "maryland-distinguished-professors": (4, 10),
    "minnesota-archaeologist": (1, 6),
    "nasa-ehsan-gharib-nezhad": (0, 13),
    "nasa-jennifer-eigenbrode": (11, 12),
    "psu-early-career": (5, 9),
    "rice-kavraki": (1, 6),
}
VIRGINIA_WINDOW = (2, 6)
EXCLUDED = {
    "kentucky-research-professors": "article prose is under 200 words",
    "nasa-insoo-jun": "biography is under 200 words",
    "osu-green": "all 200-500 word person windows repeat reserved ORG referents",
}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}


def digest(data):
    """Hash source and packet bytes without changing their encoding."""
    return hashlib.sha256(data).hexdigest()


class ArticleBlocks(HTMLParser):
    """Extract visible headings and paragraphs with their source order preserved."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tag = None
        self.depth = 0
        self.parts = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if self.tag is None and tag in {"h1", "h2", "h3", "p"}:
            self.tag = tag
            self.depth = 1
            self.parts = []
        elif self.tag is not None and tag not in VOID:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.tag is None:
            return
        self.depth -= 1
        if self.depth <= 0:
            value = " ".join("".join(self.parts).split())
            if value:
                self.blocks.append((self.tag, value))
            self.tag = None
            self.depth = 0

    def handle_data(self, data):
        if self.tag is not None:
            self.parts.append(data)


class VirginiaArticle(HTMLParser):
    """Read the DHR press-release content field, including its plain-text paragraphs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.depth:
            if tag not in VOID:
                self.depth += 1
        elif (tag == "div" and "jet-listing-dynamic-field__content" in
              dict(attrs).get("class", "").split()):
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1
            if tag == "p":
                self.parts.append("\n\n")

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def passage(source_id, raw):
    """Return an exact, deterministic article-block window and its locator."""
    html = raw.decode("utf-8")
    if source_id == "virginia-archaeologist":
        parser = VirginiaArticle()
        parser.feed(html)
        article = "".join(parser.parts)
        anchor = article.index("RICHMOND –")
        paragraphs = [" ".join(value.split()) for value in
                      article[anchor:].split("\n\n") if value.split()]
        start, end = VIRGINIA_WINDOW
        return "\n\n".join(paragraphs[start:end + 1]), {
            "method": "dhr_content_paragraphs_v1", "start": start, "end": end,
        }
    parser = ArticleBlocks()
    parser.feed(html)
    first_h1 = next(index for index, (tag, _) in enumerate(parser.blocks)
                    if tag == "h1")
    start, end = BLOCK_WINDOWS[source_id]
    selected = parser.blocks[first_h1 + start:first_h1 + end + 1]
    if len(selected) != end - start + 1:
        raise ValueError(f"article window changed: {source_id}")
    return "\n\n".join(value for _, value in selected), {
        "method": "heading_paragraph_blocks_v1", "start": start, "end": end,
    }


def proposed_real_silver():
    """Replay the captured silver baseline after later sources are promoted."""
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text())
        if frozen["kind"] != "us_person_long_blind_v2":
            raise ValueError("person baseline manifest kind changed")
        paths = [ROOT / name for name in frozen["input_sha256"]
                 if name.startswith("data/interim/silver/")]
    else:
        match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", CONFIG.read_text())
        if match is None:
            raise ValueError("proposed silver configuration is missing")
        configured = ast.literal_eval(match.group(1))
        paths = [ROOT / f"data/interim/silver/"
                 f"{REPLACEMENTS.get(Path(path).stem, Path(path).stem)}.jsonl"
                 for path in configured]
        paths.extend(ROOT / f"data/interim/silver/{name}.jsonl" for name in ADDITIONS)
    if len(paths) != len(set(paths)):
        raise ValueError("proposed silver has duplicate files")
    return paths, [row for path in paths for row in rows(path)]


def selected_rows():
    """Validate raw captures and screen every frozen passage before review."""
    capture = json.loads(INDEX.read_text())
    if capture["kind"] != "us_person_long_v2_official_page_capture":
        raise ValueError("person source capture kind changed")
    sources = {source["source_id"]: source for source in capture["sources"]}
    if set(sources) != set(BLOCK_WINDOWS) | {"virginia-archaeologist"} | set(EXCLUDED):
        raise ValueError("person source capture membership changed")
    selected = []
    for source_id in sorted(set(BLOCK_WINDOWS) | {"virginia-archaeologist"}):
        source = sources[source_id]
        path = ROOT / source["raw_path"]
        raw = path.read_bytes()
        if digest(raw) != source["raw_sha256"] or len(raw) != source["bytes"]:
            raise ValueError(f"person raw source changed: {source_id}")
        body, locator = passage(source_id, raw)
        words = len(body.split())
        if not 200 <= words <= 500 or "\ufffd" in body:
            raise ValueError(f"person article window invalid: {source_id}")
        selected.append({
            "name": f"us-person-long-v2-{source_id}",
            "country": "US", "input": body,
            "source": "official-page", "source_id": source_id,
            "source_url": source["source_url"],
            "source_group": f"official-page:{source_id}",
            "source_document_key": f"url:{source['source_url']}",
            "source_file": source["raw_path"],
            "raw_sha256": source["raw_sha256"],
            "source_window": locator,
            "input_sha256": digest(body.encode()),
            "words": words,
        })
    gold = [row for path in EVALUATION for row in rows(path)]
    if len(gold) != 371:
        raise ValueError("reserved evaluation membership changed")
    hits = heldout_reasons(selected, gold)
    if hits:
        raise ValueError(f"person packet overlaps reserved evaluation: {hits}")
    silver_paths, silver = proposed_real_silver()
    silver_keys = {document_key(row) for row in silver}
    silver_urls = {row.get("source_url", "").rstrip("/") for row in silver}
    silver_fragments = set().union(*(shingles(row["text"], 12) for row in silver))
    for row in selected:
        if (document_key(row) in silver_keys or
                row["source_url"].rstrip("/") in silver_urls or
                shingles(row["input"], 12) & silver_fragments):
            raise ValueError(f"person packet overlaps proposed real silver: {row['name']}")
    source_keys = [document_key(row) for row in selected]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("person packet repeats a source document")
    return selected, sources, silver_paths, len(silver)


def main():
    """Write an immutable unlabeled packet with all screen inputs pinned."""
    selected, sources, silver_paths, silver_count = selected_rows()
    packet = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                     for row in selected).encode()
    inputs = (INDEX, ROOT / "bench/us/check_us_blind_packet_holdout.py", CONFIG,
              *EVALUATION, *silver_paths)
    manifest = {
        "kind": "us_person_long_blind_v2",
        "label_status": "unlabeled",
        "training_eligible": False,
        "evaluation_eligible": False,
        "cases": len(selected),
        "source_documents": len(selected),
        "source_organizations": len({urlsplit(row["source_url"]).netloc
                                     for row in selected}),
        "reserved_evaluation_cases": 371,
        "proposed_real_silver_cases": silver_count,
        "screening": "strict person/org/address/contact/source/12-word eval overlap; "
                     "source and 12-word proposed real silver overlap",
        "excluded_captures": EXCLUDED,
        "capture_sources": len(sources),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                         for path in inputs},
        "sha256": digest(packet),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("person packet or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != packet:
        raise ValueError("frozen person packet changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("person packet screening inputs changed")
    if not OUT.exists():
        OUT.write_bytes(packet)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(selected)} blind person passages from "
          f"{manifest['source_organizations']} official organizations")


if __name__ == "__main__":
    main()
