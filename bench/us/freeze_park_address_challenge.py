"""Freeze source-checked park office contacts for US evaluation."""

import collections
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/us-release-eval"
REVIEW = ROOT / "data/interim/review"
OUTPUT = REVIEW / "us-park-address-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-park-address-challenge-v1.manifest.json"
SOURCE_URL = "https://www.nps.gov/aboutus/contactinformation.htm"
DATA_URL = "https://www.nps.gov/common/uploads/sortable_dataset/aboutus/E1AC0F2F-C312-E9B6-02E997F74239807C/aboutus-NPSParkandSuperintendentList2023.csv"
FROZEN_SHA256 = "2978521f96555ffe25960c80967dd97e5e3b8cc7cb136e2b92746634787b04c7"
BASE_SHA256 = {
    "us-dev-office-exclusions-v1.jsonl": "685898c5dbd4d200ad4205933046e3afa99e0e3fd294c0bf0a9a1d38f14de095",
    "us-staff-challenge-v1.jsonl": "7f840f4649ec310184f56e6bf8af4ed3fa645890a96de61db502df23dfad1624",
}
HASHES = {
    "nps-contact-information.html": "25732f0809290dc1f1e6688ca57bf43469e81e29d22384a7af96b30ebda5c308",
    "nps-parks.csv": "1665fc2e0d0bb7834b3ab26522d935d1214397e2a1a5bbe569ea7a1865c12604",
    "nps-parks.json": "ca4d815bd1a3f4fa80986cf48e758b7d8d1a69bdfed8cf6b0b2e2aa9f45d847e",
}
REVIEW_EXCLUDED_SITES = {
    "HONOULIULI": "source street has an apparent spelling error",
    "CAMP NELSON": "source repeats a road fragment in the address",
    "MEDGAR AND MYRLIE EVERS HOME": "source address includes an unlabeled site name",
    "MINUTEMAN MISSILE": "source splits the street name across lines",
    "PATERSON GREAT FALLS": "source address misspells the city name",
}
ZIP = re.compile(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?$")
PHONE = re.compile(r"(?:\(?\d{3}\)?[ .-]*)\d{3}[ .-]*\d{4}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    return "\n".join(" ".join(line.split()) for line in value.strip().splitlines() if line.strip())


def flat(value):
    return " ".join(value.split())


def validate_sources():
    for filename, expected in HASHES.items():
        if digest((RAW / filename).read_bytes()) != expected:
            raise ValueError(f"NPS source changed: {filename}")
    page = (RAW / "nps-contact-information.html").read_text()
    if "aboutus-NPSParkandSuperintendentList2023.csv" not in page:
        raise ValueError("captured NPS page does not link the source dataset")
    with (RAW / "nps-parks.csv").open(encoding="cp1252", newline="") as handle:
        source_rows = list(csv.reader(handle))
    if source_rows[0][:4] != ["Park or Site", "Current or Acting Superintendent", "Address", "Phone"]:
        raise ValueError("NPS CSV columns changed")
    csv_rows = {flat(row[0]): row for row in source_rows[1:] if row and flat(row[0])}
    json_rows = json.loads((RAW / "nps-parks.json").read_text())["DATA"]
    if len(json_rows) < 400:
        raise ValueError("NPS dataset too small")
    return csv_rows, json_rows


def existing_surfaces():
    paths = [REVIEW / filename for filename in BASE_SHA256]
    for path in paths:
        if digest(path.read_bytes()) != BASE_SHA256[path.name]:
            raise ValueError(f"prior evaluation changed after NPS freeze: {path.name}")
    return {
        (span["kind"], flat(span["text"]).casefold())
        for path in paths
        for line in path.read_text().splitlines()
        for span in json.loads(line)["expected"]
    }


def case_for(index, row):
    park, raw_name, raw_address, raw_phone = row[:4]
    if flat(park) in REVIEW_EXCLUDED_SITES:
        return None
    name = flat(raw_name)
    address = clean(raw_address)
    match = ZIP.search(address)
    if (match is None or not re.match(r"^\d{1,6}[A-Za-z-]*\s", address)
            or any(value in address.casefold() for value in ("p.o. box", "po box", "c/o", "http", "@"))
            or len(address) > 130 or "\ufffd" in address or "\n" in raw_name
            or not 2 <= len(name.split()) <= 4 or len(name) > 60
            or name.casefold().startswith("see ")
            or any(char.isdigit() for char in name)):
        return None
    phone = PHONE.search(raw_phone)
    if phone is None:
        return None
    phone = phone.group()
    text = "Superintendent: " + name + "\nAddress: " + address + "\nPhone: " + phone
    expected = []
    for kind, value, prefix in (("person", name, "Superintendent: "),
                                ("address", address, "Address: "),
                                ("phone", phone, "Phone: ")):
        start = text.index(prefix) + len(prefix)
        start = len(text[:start].encode())
        expected.append({"kind": kind, "start": start,
                         "end": start + len(value.encode()), "text": value})
    return {"name": f"us-park-address-{index:03}", "country": "US", "input": text,
            "expected": expected, "doc_type": "park_office_contact",
            "source_url": SOURCE_URL, "source_data_url": DATA_URL,
            "source_json_index": index, "source_site": flat(park), "state": match[1]}


def main():
    csv_rows, json_rows = validate_sources()
    excluded = existing_surfaces()
    by_state = collections.defaultdict(list)
    for index, row in enumerate(json_rows):
        case = case_for(index, row)
        if case is None:
            continue
        csv_row = csv_rows.get(case["source_site"])
        if csv_row is None or any(flat(left) != flat(right) for left, right in zip(csv_row[:4], row[:4])):
            continue
        if any((span["kind"], flat(span["text"]).casefold()) in excluded
               for span in case["expected"]):
            continue
        by_state[case["state"]].append(case)
    if len(by_state) < 40:
        raise ValueError("NPS challenge lost geographic coverage")
    for state in by_state:
        by_state[state].sort(key=lambda case: digest(case["input"].encode()))
    cases = []
    seen_texts = set()
    for offset in range(max(map(len, by_state.values()))):
        for state in sorted(by_state):
            if offset < len(by_state[state]):
                case = by_state[state][offset]
                if case["input"] in seen_texts:
                    continue
                cases.append(case)
                seen_texts.add(case["input"])
                if len(cases) == 50:
                    break
        if len(cases) == 50:
            break
    if len(cases) != 50 or len({case["input"] for case in cases}) != 50:
        raise ValueError("NPS challenge does not have 50 distinct cases")
    counts = collections.Counter(span["kind"] for case in cases for span in case["expected"])
    for case in cases:
        data = case["input"].encode()
        for span in case["expected"]:
            if data[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"bad NPS label: {case['name']}")
    output = b"".join(json.dumps(case, ensure_ascii=False).encode() + b"\n" for case in cases)
    if digest(output) != FROZEN_SHA256:
        raise ValueError("NPS challenge labels changed after freeze")
    OUTPUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "us_park_address_challenge",
        "status": "exposed in us-v1 evaluation; these 50 rows remain excluded from training",
        "representation": "source fields transcribed into superintendent, address, phone rows",
        "source_sha256": HASHES,
        "cases": len(cases), "states": len({case["state"] for case in cases}),
        "labels": dict(counts), "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(cases)} NPS cases, {len({case['state'] for case in cases})} states, {dict(counts)}")


if __name__ == "__main__":
    main()
