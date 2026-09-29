"""Freeze Oregon district contacts from the state education department's directory."""

import hashlib
import json
from pathlib import Path

from check_ready import normalized, person_key
from freeze_ca_superintendent_challenge import training_surfaces


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
OUT = REVIEW / "us-or-district-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-or-district-challenge-v1.manifest.json"
SOURCE_URL = "https://www.oregon.gov/ode/students-and-family/SpecialEducation/GeneralSupervision/Documents/districtsupportcontacts.pdf"
FROZEN_SHA256 = "34db9119c19211232f30d0b53476d3c8817a15e2754079b33eb85c122030b2d8"
ROWS = (
    ("Adel SD 21", "Cherisse Gordon", "Cherisse.Gordon@ode.oregon.gov"),
    ("Arlington SD 3", "Claire Skelly", "Claire.Skelly@ode.oregon.gov"),
    ("Astoria SD 1", "Kristin Irwin", "Kristin.Irwin@ode.oregon.gov"),
    ("Bethel SD 52", "Stacy Matthews", "Stacy.Matthews@ode.oregon.gov"),
    ("Centennial SD 28J", "Laura Petschauer", "Laura.Petschauer@ode.oregon.gov"),
    ("Alsea SD 7J", "Cherisse Gordon", "Cherisse.Gordon@ode.oregon.gov"),
    ("Baker SD 5J", "Claire Skelly", "Claire.Skelly@ode.oregon.gov"),
    ("Banks SD 13", "Kristin Irwin", "Kristin.Irwin@ode.oregon.gov"),
    ("Blachly SD 90", "Stacy Matthews", "Stacy.Matthews@ode.oregon.gov"),
    ("Corbett SD 39", "Laura Petschauer", "Laura.Petschauer@ode.oregon.gov"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def case_for(index, district, person, email):
    text = f"District: {district}\nCompliance specialist: {person}\nEmail: {email}"
    expected = []
    for kind, value in (("org", district), ("person", person), ("email", email)):
        start = len(text[:text.index(value)].encode())
        expected.append({"kind": kind, "start": start,
                         "end": start + len(value.encode()), "text": value})
    return {"name": f"us-or-district-{index:02}", "country": "US",
            "input": text, "expected": expected,
            "doc_type": "school_district_contact", "source_url": SOURCE_URL,
            "source_representation": "official directory table row transcribed into labeled contact lines"}


def main():
    values, people, _ = training_surfaces()
    prior = [json.loads(line) for line in
             (REVIEW / "us-eval-exclusions-v1.jsonl").read_text().splitlines()]
    prior = [row for row in prior if not row["name"].startswith("us-or-district-")]
    prior_people = {person_key(span["text"]) for row in prior
                    for span in row["expected"] if span["kind"] == "person"}
    prior_surfaces = {(span["kind"], normalized(span["text"]))
                      for row in prior for span in row["expected"]}
    cases = [case_for(index, *row) for index, row in enumerate(ROWS, 1)]
    for case in cases:
        data = case["input"].encode()
        for span in case["expected"]:
            value = span["text"]
            kind = span["kind"]
            if data[span["start"]:span["end"]].decode() != value:
                raise ValueError(f"bad Oregon label: {case['name']}")
            if (kind, normalized(value)) in prior_surfaces:
                raise ValueError(f"evaluation overlap: {value}")
            if kind in values and normalized(value) in values[kind]:
                raise ValueError(f"training overlap: {value}")
            if kind == "person" and person_key(value) in people | prior_people:
                raise ValueError(f"person alias overlap: {value}")
    output = b"".join(json.dumps(case, ensure_ascii=False).encode() + b"\n" for case in cases)
    if digest(output) != FROZEN_SHA256:
        raise ValueError("Oregon challenge changed after freeze")
    OUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "us_independent_district_contact_challenge",
        "status": "exposed in district-v3 evaluation; rows excluded from later training",
        "source_url": SOURCE_URL,
        "source_capture": "manual transcription of ten rows in the official July 2026 directory",
        "cases": len(cases), "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(cases)} Oregon district cases: {digest(output)}")


if __name__ == "__main__":
    main()
