"""Extract split-safe person supervision from the official House staff directory."""

import collections
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-staff"
PDF = RAW / "house-phonebook-2023.pdf"
GOLD = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUTPUT = ROOT / "data/interim/silver/us-house-staff-v1.jsonl"
MANIFEST = ROOT / "data/interim/silver/us-house-staff-v1.manifest.json"
SOURCE_URL = "https://directory.house.gov/App/pdfs/Phonebook-2023.pdf"
PDF_SHA256 = "e9e993ee57417d03d872a030b7ae139d17426eac7a1cf58db846e57f434c1c3b"
TEXT_SHA256 = "1358767910ca3f394902f482cc6d97681365108906e95c4abfa75e53582cc43c"
FIRST_PAGE = 20
LAST_PAGE = 135
ROWS_PER_DOCUMENT = 15
NAME = re.compile(
    r"^[A-Z][A-Za-zÀ-ÿ'’\-]*(?:[ -][A-Z][A-Za-zÀ-ÿ'’\-]*)*, "
    r"[A-Z][A-Za-zÀ-ÿ'’\-]*(?:[ .-][A-Z][A-Za-zÀ-ÿ'’\-]*)*"
    r"(?:,? (?:Jr\.?|Sr\.?|II|III|IV))?\.?$"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def name_from_line(part):
    if ", " not in part or not part[:1].isupper():
        return None
    name = part.split(" (", 1)[0].strip()
    if " (" not in part and not re.fullmatch(r"[\w .,'’\-]+", name):
        return None
    name = re.split(r"\.{3,}|\d{3}|  ", name)[0].strip()
    return name if NAME.fullmatch(name) else None


def clean_line(part):
    without_affiliation = re.sub(r"\([^)]*(?:\)|$)", "", part)
    return re.sub(r"\s+", " ", re.sub(r"\.{2,}", " ", without_affiliation)).strip()


def right_column(lines):
    positions = []
    for line in lines[1:]:
        positions.extend(
            match.start()
            for match in re.finditer(r"[A-Z][a-zA-Z'’\-]+, [A-Z][a-zA-Z'’\-]+", line)
            if match.start() > 100
        )
    if not positions:
        raise ValueError("no right-column names in extracted page")
    return collections.Counter(positions).most_common(1)[0][0]


def evaluation_people():
    names = {
        span["text"].casefold()
        for line in GOLD.read_text().splitlines()
        for span in json.loads(line)["expected"]
        if span["kind"] == "person"
    }
    return names, {person_key(name) for name in names}


def person_key(value):
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    if "," in normalized:
        surname, given = normalized.split(",", 1)
        given_parts = re.findall(r"[a-z]+", given)
        surname_parts = re.findall(r"[a-z]+", surname)
        return (given_parts[0], surname_parts[-1]) if given_parts and surname_parts else None
    parts = re.findall(r"[a-z]+", normalized)
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        parts.pop()
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    pdf_hash = digest(PDF.read_bytes())
    if pdf_hash != PDF_SHA256:
        raise ValueError(f"House directory PDF changed: {pdf_hash}")
    subprocess.run([
        "gs", "-q", "-dBATCH", "-dNOPAUSE", f"-dFirstPage={FIRST_PAGE}",
        f"-dLastPage={LAST_PAGE}", "-sDEVICE=txtwrite",
        f"-sOutputFile={RAW / 'page-%03d.txt'}", str(PDF),
    ], check=True, capture_output=True)
    source_pages = [RAW / f"page-{page - FIRST_PAGE + 1:03}.txt"
                    for page in range(FIRST_PAGE, LAST_PAGE + 1)]
    text_hash = digest(b"".join(path.read_bytes() for path in source_pages))
    if text_hash != TEXT_SHA256:
        raise ValueError(f"House directory text extraction changed: {text_hash}")
    excluded, excluded_keys = evaluation_people()
    rows = []
    counts = collections.Counter()
    distinct = set()
    for page in range(FIRST_PAGE, LAST_PAGE + 1):
        path = source_pages[page - FIRST_PAGE]
        lines = path.read_text().splitlines()
        if len(lines) < 3 or "ALPHABETICAL STAFF LISTING" not in lines[0] and "U.S. HOUSE OF REPRESENTATIVES" not in lines[0]:
            raise ValueError(f"unexpected text extraction: {path}")
        split = right_column(lines)
        for column, label in ((0, "left"), (1, "right")):
            entries = []
            for line in lines[1:]:
                part = (line[:split] if column == 0 else line[split:]).strip()
                name = name_from_line(part)
                if name is None:
                    continue
                given = " ".join(reversed(name.split(", ", 1)))
                if (name.casefold() in excluded or given.casefold() in excluded or
                        person_key(name) in excluded_keys):
                    counts["evaluation_names_removed"] += 1
                    continue
                cleaned = clean_line(part)
                if not cleaned.startswith(name) or "\ufffd" in cleaned:
                    raise ValueError(f"bad extracted staff line on page {page}: {part!r}")
                entries.append((cleaned, name))
                distinct.add(name.casefold())
            for batch_index, batch in enumerate(chunks(entries, ROWS_PER_DOCUMENT), 1):
                if len(batch) < 3:
                    counts["short_batch_rows_removed"] += len(batch)
                    continue
                text = "\n".join(line for line, _ in batch)
                entities = []
                offset = 0
                for line, name in batch:
                    entities.append({"kind": "person", "start": offset,
                                     "end": offset + len(name.encode())})
                    offset += len(line.encode()) + 1
                rows.append({
                    "id": f"house-phonebook-2023-p{page:04}-{label}-{batch_index:02}",
                    "country": "US",
                    "source": "us-house-staff-directory-2023",
                    "source_url": SOURCE_URL,
                    "source_page": page,
                    "text": text,
                    "entities": entities,
                })
                counts["person_labels"] += len(entities)
    for row in rows:
        raw = row["text"].encode()
        for entity in row["entities"]:
            name = raw[entity["start"]:entity["end"]].decode()
            if (not NAME.fullmatch(name) or name.casefold() in excluded or
                    person_key(name) in excluded_keys):
                raise ValueError(f"invalid person span in {row['id']}: {name!r}")
    if counts["person_labels"] < 8000 or len(distinct) < 7900:
        raise ValueError("House staff extraction lost too many names")
    data = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_house_staff_derived_silver",
        "status": "all names match pinned source text and strict person pattern; visual page sample passed",
        "source_url": SOURCE_URL,
        "source_sha256": PDF_SHA256,
        "source_text_sha256": TEXT_SHA256,
        "extraction": "Ghostscript txtwrite, PDF pages 20-135, two columns; only strict name-pattern rows retained",
        "evaluation_sha256": digest(GOLD.read_bytes()),
        "documents": len(rows),
        "distinct_people": len(distinct),
        "counts": dict(counts),
        "sha256": digest(data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} documents, {counts['person_labels']} labels, {len(distinct)} distinct names")


if __name__ == "__main__":
    main()
