"""Freeze independent Kentucky superintendent contacts before role-template changes."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
OUTPUT = REVIEW / "us-ky-superintendent-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-ky-superintendent-challenge-v1.manifest.json"
SOURCE_URL = "https://openhouse.education.ky.gov/Superintendents"
FROZEN_SHA256 = "aad64112489863b64b543ca4e71c8593bb6165e1843d38b8782c22bf44c6d9d8"
ROWS = (
    ("Adair County", "Jason Faulkner", "1204 Greensburg St.", "Columbia", "42728", "(270) 384-2476"),
    ("Allen County", "Travis Hamby", "570 Oliver St", "Scottsville", "42164", "(270) 618-3181"),
    ("Anchorage Independent", "Sharla Six", "11400 Ridge Rd", "Anchorage", "40223", "(502) 245-8927"),
    ("Anderson County", "Sheila Mitchell", "1160 By-pass North", "Lawrenceburg", "40342", "(502) 839-3406"),
    ("Ashland Independent", "Derek Howard", "1820 Hickman St", "Ashland", "41101", "(606) 327-2706"),
    ("Augusta Independent", "Lisa Mccane", "307 Bracken St", "Augusta", "41002", "(606) 756-2545"),
    ("Ballard County", "Casey Allen", "11 Vocational School Road", "Barlow", "42024", "(270) 665-8400"),
    ("Barbourville Independent", "Dennis Messer", "140 School St", "Barbourville", "40906", "(606) 546-3120"),
    ("Bardstown Independent", "Ryan Clark", "308 N. Fifth Street", "Bardstown", "40004", "(502) 331-8800"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def case_for(index, row):
    district, person, street, city, postcode, phone = row
    address = f"{street}\n{city}, KY {postcode}"
    text = f"District: {district}\nSuperintendent: {person}\nAddress: {address}\nPhone: {phone}"
    expected = []
    for kind, value, prefix in (
        ("org", district, "District: "),
        ("person", person, "Superintendent: "),
        ("address", address, "Address: "),
        ("phone", phone, "Phone: "),
    ):
        start = text.index(prefix) + len(prefix)
        if kind == "address":
            start = text.index(address)
        start = len(text[:start].encode())
        expected.append({"kind": kind, "start": start,
                         "end": start + len(value.encode()), "text": value})
    return {"name": f"us-ky-superintendent-{index:02}", "country": "US", "input": text,
            "expected": expected, "doc_type": "school_district_contact",
            "source_url": SOURCE_URL, "source_district": district,
            "source_representation": "official directory fields transcribed into labeled contact lines"}


def main():
    cases = [case_for(index, row) for index, row in enumerate(ROWS, 1)]
    prior = [json.loads(line) for filename in (
        "us-dev-office-exclusions-v1.jsonl",
        "us-staff-challenge-v1.jsonl",
        "us-park-address-challenge-v1.jsonl",
    ) for line in (REVIEW / filename).read_text().splitlines()]
    prior_surfaces = {(span["kind"], span["text"].casefold())
                      for case in prior for span in case["expected"]}
    for case in cases:
        data = case["input"].encode()
        for span in case["expected"]:
            if data[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"bad label: {case['name']}")
            if (span["kind"], span["text"].casefold()) in prior_surfaces:
                raise ValueError(f"prior evaluation overlap: {span['text']}")
    output = b"".join(json.dumps(case, ensure_ascii=False).encode() + b"\n" for case in cases)
    if digest(output) != FROZEN_SHA256:
        raise ValueError("Kentucky holdout changed after freeze")
    OUTPUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "us_independent_superintendent_challenge",
        "status": "exposed in role-v2 evaluation; rows excluded from later training",
        "source_url": SOURCE_URL,
        "source_capture": "manual transcription of public fields shown on the official directory",
        "cases": len(cases), "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(cases)} Kentucky superintendent cases: {digest(output)}")


if __name__ == "__main__":
    main()
