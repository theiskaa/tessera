"""Extract source-checked US district office addresses from the House directory."""

import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

from address_keys import address_keys


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-staff"
PDF = RAW / "house-phonebook-2023.pdf"
GOLD = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUTPUT = ROOT / "data/interim/silver/us-house-district-offices-v1.jsonl"
MANIFEST = ROOT / "data/interim/silver/us-house-district-offices-v1.manifest.json"
SOURCE_URL = "https://directory.house.gov/App/pdfs/Phonebook-2023.pdf"
PDF_SHA256 = "e9e993ee57417d03d872a030b7ae139d17426eac7a1cf58db846e57f434c1c3b"
TEXT_SHA256 = "469e5fba5577280811e27ba4d79760591c733530e83473c35586ca78d7de7bb0"
FIRST_PAGE = 136
LAST_PAGE = 224
OFFICE = re.compile(r"(?<![A-Z])([A-Z][A-Z .'/\-]{2,50})—([^\n]*?\b[A-Z]{2} \d{5}(?:-\d{4})?)")
ZIP = re.compile(r", ([A-Z]{2}) (\d{5}(?:-\d{4})?)$")
STREET = re.compile(r"\b(?:ST|AVE|RD|DR|BLVD|HWY|WAY|LN|PL|PKWY|CIR|CT|PIKE|ROAD|STREET|BROADWAY)\b")
MAX_PER_PAGE = 4
TARGET = 200


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return " ".join(value.casefold().split())


def evaluation_addresses():
    labels = [span["text"] for line in GOLD.read_text().splitlines()
              for span in json.loads(line)["expected"] if span["kind"] == "address"]
    return ({normalized(label) for label in labels},
            set().union(*(address_keys(label) for label in labels)))


def main():
    if digest(PDF.read_bytes()) != PDF_SHA256:
        raise ValueError("House directory PDF changed")
    subprocess.run([
        "gs", "-q", "-dBATCH", "-dNOPAUSE", f"-dFirstPage={FIRST_PAGE}",
        f"-dLastPage={LAST_PAGE}", "-sDEVICE=txtwrite",
        f"-sOutputFile={RAW / 'house-reps-page-%03d.txt'}", str(PDF),
    ], check=True, capture_output=True)
    pages = [RAW / f"house-reps-page-{index:03}.txt"
             for index in range(1, LAST_PAGE - FIRST_PAGE + 2)]
    if digest(b"".join(path.read_bytes() for path in pages)) != TEXT_SHA256:
        raise ValueError("House representative-page text extraction changed")
    excluded, excluded_places = evaluation_addresses()
    by_state = collections.defaultdict(list)
    seen = set()
    counts = collections.Counter()
    for index, path in enumerate(pages, FIRST_PAGE):
        lines = path.read_text().splitlines()
        if not any("REPRESENTATIVES AND STAFF" in line or "U.S. HOUSE OF REPRESENTATIVES" in line
                   for line in lines[:3]):
            raise ValueError(f"unexpected House directory page {index}")
        for line_no, line in enumerate(lines, 1):
            for match in OFFICE.finditer(line):
                address = match[2].strip()
                postcode = ZIP.search(address)
                if (postcode is None or not re.match(r"^\d{1,6}[A-Z]? ", address)
                        or len(address) > 100 or "  " in address or "—" in address
                        or STREET.search(address) is None):
                    counts["format_rejected"] += 1
                    continue
                if normalized(address) in excluded or address_keys(address) & excluded_places:
                    counts["evaluation_rejected"] += 1
                    continue
                if normalized(address) in seen:
                    counts["duplicate_rejected"] += 1
                    continue
                seen.add(normalized(address))
                by_state[postcode[1]].append((index, line_no, match[1].strip(), address))
                counts["candidates"] += 1
    if counts["candidates"] < 300 or len(by_state) < 45:
        raise ValueError("too few source-checked House office addresses")
    for state in by_state:
        by_state[state].sort(key=lambda row: digest(row[3].encode()))
    selected = []
    per_page = collections.Counter()
    for offset in range(max(map(len, by_state.values()))):
        for state in sorted(by_state):
            if offset >= len(by_state[state]):
                continue
            page, line_no, locality, address = by_state[state][offset]
            if per_page[page] >= MAX_PER_PAGE:
                continue
            selected.append((page, line_no, locality, address, state))
            per_page[page] += 1
            if len(selected) == TARGET:
                break
        if len(selected) == TARGET:
            break
    if len(selected) != TARGET:
        raise ValueError("House office selection could not fill the target")
    rows = []
    for page, line_no, locality, address, state in selected:
        text = "District office address:\n" + address
        start = len("District office address:\n".encode())
        rows.append({
            "id": f"house-district-office-p{page:04}-l{line_no:03}",
            "country": "US", "source": "us-house-district-directory-2023",
            "source_url": SOURCE_URL, "source_page": page,
            "source_line": line_no, "source_locality": locality, "state": state,
            "text": text, "entities": [{"kind": "address", "start": start,
                                     "end": start + len(address.encode())}],
        })
    if len({row["id"] for row in rows}) != TARGET:
        raise ValueError("duplicate House office source line")
    for row in rows:
        span = row["entities"][0]
        address = row["text"].encode()[span["start"]:span["end"]].decode()
        source_lines = pages[row["source_page"] - FIRST_PAGE].read_text().splitlines()
        if address not in source_lines[row["source_line"] - 1]:
            raise ValueError(f"House address differs from source: {row['id']}")
    data = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_house_district_office_silver",
        "status": "strict single-line source addresses; source and offset verified",
        "source_url": SOURCE_URL, "source_sha256": PDF_SHA256,
        "source_text_sha256": TEXT_SHA256,
        "evaluation_sha256": digest(GOLD.read_bytes()),
        "documents": len(rows), "labels": {"address": len(rows)},
        "states": len({row["state"] for row in rows}),
        "pages": len(per_page), "selection": "state round-robin; at most four offices per PDF page",
        "counts": dict(counts), "sha256": digest(data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} House district office addresses across {len({row['state'] for row in rows})} states")


if __name__ == "__main__":
    main()
