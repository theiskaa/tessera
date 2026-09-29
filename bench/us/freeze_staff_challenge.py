"""Freeze source-verified US staff directory cases before model training."""

import collections
import hashlib
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-release-eval"
OUTPUT = ROOT / "data/interim/review/us-staff-challenge-v1.jsonl"
MANIFEST = ROOT / "data/interim/review/us-staff-challenge-v1.manifest.json"
EXISTING = ROOT / "data/interim/review/us-dev-office-exclusions-v1.jsonl"
PHONE = re.compile(r"(?:\(\d{3}\)|\d{3})[ -.]*\d{3}[-. ]\d{4}")
MAX_PEOPLE_PER_CASE = 23
BASE_SHA256 = "685898c5dbd4d200ad4205933046e3afa99e0e3fd294c0bf0a9a1d38f14de095"
FROZEN_SHA256 = "7f840f4649ec310184f56e6bf8af4ed3fa645890a96de61db502df23dfad1624"
SOURCE_SHA256 = {
    "annapolis-law": "f62b02a12e1d8229dcd4b143cca882a13b0c22ce97d78d2bfb309fa7e3564b20",
    "annapolis-mayor": "9647908920a76a69a416e1fa36d9a0a41fc3dba0051f5868fb2e949e82c062f4",
    "augusta-admin": "bc9af00bfc6cc5d9048879caaa8f885d1f8931a4d024e202bfc2894b17d488b1",
    "dra": "0a126c5e9b1bb9390f33eccb97fc8e2eaf758ca8fd400f47cfad15807e24ded7",
    "framingham-mayor": "ab4d10aad20ae6441a141315f8d857ba0709535978336f699ce599e85b6a5229",
    "opc-dc": "93894325e33e5f845128bd07867c1e2c90647050a7b6e967df1fdb9b3acd3b1e",
    "oregon-real-estate": "8f81705550879ca1ed74cd4671a1dfb1bc636960ce8aff52a0cd27f1ded87ba4",
    "rome-manager": "10997b43657f6b40935fbf1486a29599bd14a1fadb150cffacc281d8b5551edf",
}
PAGES = {
    "annapolis-law": (
        "https://www.annapolis.gov/directory.aspx?did=28",
        "Office of Law", "160 Duke of Gloucester\nAnnapolis, MD 21401", "civicplus", 10,
    ),
    "annapolis-mayor": (
        "https://www.annapolis.gov/Directory.aspx?DID=27",
        "Mayor's Office", "160 Duke of Gloucester Street\nAnnapolis, MD 21401", "civicplus", 12,
    ),
    "augusta-admin": (
        "https://www.augustaga.gov/3289/Staff",
        "Administrator's Office", "535 Telfair St.\nSuite 910\nAugusta, GA 30901", "civicplus", 6,
    ),
    "framingham-mayor": (
        "https://framinghamma.gov/Directory.aspx?DID=89",
        "Office of the Mayor", "150 Concord Street\nRoom 213\nFramingham, MA 01702", "civicplus", 7,
    ),
    "rome-manager": (
        "https://www.romega.gov/Directory.aspx?did=51",
        "City Manager's Office", "601 Broad Street\nRome, GA 30161", "civicplus", 5,
    ),
    "oregon-real-estate": (
        "https://www.oregon.gov/rea/about-us/Pages/staff-directory.aspx",
        "Oregon Real Estate Agency", "775 Summer St NE, Suite 330\nSalem, OR 97301", "oregon", 28,
    ),
    "dra": (
        "https://dra.gov/about/staff-directory/",
        "Delta Regional Authority", "236 Sharkey Avenue, Suite 400\nClarksdale, MS 38614", "dra", 27,
    ),
    "opc-dc": (
        "https://opc-dc.gov/about-opc/directory/",
        "Office of the People's Counsel", "655 15th Street NW\nSuite 200\nWashington DC, 20005-5701", "opc", 46,
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean(fragment):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def first(pattern, text, required=True):
    match = re.search(pattern, text, re.S)
    if match is None:
        if required:
            raise ValueError(f"source field not found: {pattern}")
        return ""
    return clean(match[1])


def civicplus(source):
    rows = []
    blocks = re.findall(r'<li class="list-group-item[^>]*>(.*?)</li>', source, re.S)
    for block in blocks:
        name = first(r'<a href="/m/directory/employee[^\"]*" class="fw-bold no-underline-link">(.*?)</a>', block)
        role = first(r'<div class="d-sm-block d-none">(.*?)</div>', block)
        email = first(r'<a href="mailto:[^\"]*"[^>]*>(.*?)</a>', block, False)
        if "@" not in email:
            email = ""
        phone = first(r'<a href="tel:[^\"]*"[^>]*>(.*?)</a>', block, False)
        match = PHONE.search(phone)
        rows.append((name, role, email, match.group() if match else ""))
    return rows


def oregon(source):
    return [(clean(name), clean(role), "", "") for name, role in re.findall(
        r'<p class="panel-title">(.*?)</p>\s*<p class="panel-subtitle">(.*?)</p>', source, re.S
    )]


def dra(source):
    return [(clean(name), clean(role), "", "") for name, role in re.findall(
        r'<h3 class="profile-name">([^<]+)</h3>\s*<h4 class="profile-position">([^<]+)</h4>', source
    )]


def opc(source):
    names = list(re.finditer(r'<div class="person-name">(.*?)</div>', source, re.S))
    rows = []
    for index, match in enumerate(names):
        block = source[match.end():names[index + 1].start() if index + 1 < len(names) else len(source)]
        name = clean(match[1])
        role = first(r'<div class="position">(.*?)</div>', block)
        email = first(r'<a href="mailto:[^\"]*"[^>]*class="email">(.*?)</a>', block, False)
        phone = first(r'<a href="tel:[^\"]*"[^>]*class="phone">(.*?)</a>', block, False)
        match_phone = PHONE.search(phone)
        rows.append((name, role, email, match_phone.group() if match_phone else ""))
    return rows


READERS = {"civicplus": civicplus, "oregon": oregon, "dra": dra, "opc": opc}


class Case:
    def __init__(self):
        self.text = ""
        self.expected = []

    def add(self, value, kind=None):
        start = len(self.text.encode())
        self.text += value
        if kind:
            self.expected.append({"kind": kind, "start": start,
                                  "end": start + len(value.encode()), "text": value})


def main():
    if digest(EXISTING.read_bytes()) != BASE_SHA256:
        raise ValueError("development evaluation changed after staff challenge was frozen")
    existing = [json.loads(line) for line in EXISTING.read_text().splitlines()]
    existing_texts = {digest(row["input"].encode()) for row in existing}
    existing_surfaces = {(span["kind"], span["text"].casefold())
                         for row in existing for span in row["expected"]}
    cases = []
    source_hashes = {}
    counts = collections.Counter()
    for slug, (url, org, address, reader, expected_people) in PAGES.items():
        path = RAW / f"{slug}.html"
        raw = path.read_bytes()
        source_hashes[slug] = digest(raw)
        if source_hashes[slug] != SOURCE_SHA256[slug]:
            raise ValueError(f"staff source changed after freeze: {slug}")
        source = raw.decode(errors="strict")
        visible = clean(source)
        for value in (org, *address.split("\n")):
            if value not in visible:
                raise ValueError(f"{slug}: source field missing: {value}")
        rows = READERS[reader](source)
        if len(rows) != expected_people:
            raise ValueError(f"{slug}: expected {expected_people} people, got {len(rows)}")
        for name, role, email, phone in rows:
            if (len(name.split()) < 2 or len(name) > 80 or
                    any(char.isdigit() or char in '<>"' for char in name) or not role):
                raise ValueError(f"{slug}: incomplete staff row: {name!r}")
            for field in (name, role, email, phone):
                if field and field not in visible:
                    raise ValueError(f"{slug}: staff field missing in source: {field}")
        for offset in range(0, len(rows), MAX_PEOPLE_PER_CASE):
            case = Case()
            case.add(org, "org")
            case.add("\n")
            case.add(address, "address")
            case.add("\n\n")
            for name, role, email, phone in rows[offset:offset + MAX_PEOPLE_PER_CASE]:
                case.add(name, "person")
                case.add(" | " + role)
                if email:
                    case.add(" | ")
                    case.add(email, "email")
                if phone:
                    case.add(" | ")
                    case.add(phone, "phone")
                case.add("\n")
            case.text = case.text.rstrip("\n")
            if digest(case.text.encode()) in existing_texts:
                raise ValueError(f"{slug}: duplicate development text")
            raw_text = case.text.encode()
            for span in case.expected:
                if raw_text[span["start"]:span["end"]].decode() != span["text"]:
                    raise ValueError(f"{slug}: bad byte span")
                if (span["kind"], span["text"].casefold()) in existing_surfaces:
                    raise ValueError(f"{slug}: evaluation entity overlap: {span['text']}")
                counts[span["kind"]] += 1
            cases.append({"name": f"us-staff-challenge-{slug}-{offset // MAX_PEOPLE_PER_CASE + 1:02}",
                          "country": "US", "input": case.text, "expected": case.expected,
                          "source_url": url, "source_sha256": source_hashes[slug],
                          "doc_type": "staff_directory"})
    data = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in cases)
    if digest(data) != FROZEN_SHA256:
        raise ValueError("staff challenge labels changed after freeze")
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_staff_challenge",
        "status": "exposed in us-v1 evaluation; these 11 rows remain excluded from training",
        "representation": "staff fields transcribed into consistent rows; source names and contact values retained",
        "cases": len(cases),
        "labels": dict(counts),
        "sources_sha256": source_hashes,
        "existing_evaluation_sha256": digest(EXISTING.read_bytes()),
        "sha256": digest(data),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(PAGES)} sources, {len(cases)} cases, {sum(counts.values())} labels: {dict(counts)}")


if __name__ == "__main__":
    main()
