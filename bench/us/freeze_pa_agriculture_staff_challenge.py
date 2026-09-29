"""Freeze unseen role-first staff lines from the Pennsylvania agriculture directory."""

import hashlib
import json
from pathlib import Path

from check_ready import person_key
from freeze_ca_superintendent_challenge import training_surfaces


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
OUTPUT = REVIEW / "us-pa-agriculture-staff-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-pa-agriculture-staff-challenge-v1.manifest.json"
FROZEN_SHA256 = "dfb6ff8a6ff47b3e29a3996cb6d4f9b051c94df3cdfd88cac52c0ff66fab5863"
SOURCE_URL = "https://www.pa.gov/agencies/pda/about-pda/leadership"
ROWS = (
    ("Secretary", "Russell Redding"),
    ("Deputy Secretary for Animal Health and Food Safety", "Lisa Graybeal"),
    ("Deputy Secretary for Farm, Food and Market Access", "Heidi Secord"),
    ("Acting Deputy Secretary for Plant Industry and Consumer Protection", "Chris Davis"),
    ("Central Regional Director, Special Assistant for Workforce Development", "Sara Gligora"),
    ("Executive Assistant to the Secretary", "Grace Dunigan"),
    ("Legislative Director", "Stephen Rudman"),
    ("Deputy Director of Legislative Affairs", "Sarah Hammond"),
    ("Acting Policy Director", "Lauren Shaffer"),
    ("Executive Policy Specialist", "Eve Adrian"),
    ("Communications Director", "Ashley Fehr"),
    ("Press Secretary", "Shannon Powers"),
    ("Deputy Communications Director", "Daniel Blottenberger"),
    ("Digital Director", "Zachary Newby"),
    ("Deputy Digital Director", "Caroline Estey"),
    ("Executive Director", "Marty Qually"),
    ("Director of Innovation", "Michael Roth"),
    ("Food Policy Council Director", "Dawn Plummer"),
    ("Director of Human Resources", "Christine Haertsch"),
    ("Director of Administrative Services", "Walt Remmert"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    _, training_people, _ = training_surfaces()
    prior = [json.loads(line) for line in
             (REVIEW / "us-eval-exclusions-v1.jsonl").read_text().splitlines()]
    prior_people = {person_key(span["text"]) for row in prior
                    for span in row["expected"] if span["kind"] == "person"}
    rows = []
    seen = set()
    for number, (role, person) in enumerate(ROWS, 1):
        key = person_key(person)
        if key is None or key in training_people | prior_people | seen:
            raise ValueError(f"person overlap: {person}")
        seen.add(key)
        text = f"{role}, {person}"
        start = len(f"{role}, ".encode())
        rows.append({"name": f"us-pa-agriculture-staff-{number:02}",
                     "country": "US", "input": text,
                     "expected": [{"kind": "person", "start": start,
                                   "end": start + len(person.encode()), "text": person}],
                     "doc_type": "state_agency_staff", "source_url": SOURCE_URL})
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in rows) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError("Pennsylvania staff challenge differs from its frozen version")
    if MANIFEST.exists() and not OUTPUT.exists():
        raise ValueError("Pennsylvania staff challenge data is missing")
    status = "unseen"
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != data or not MANIFEST.exists():
            raise ValueError("frozen Pennsylvania staff challenge changed")
        prior = json.loads(MANIFEST.read_text())
        if prior["sha256"] != FROZEN_SHA256 or prior["status"] not in {"unseen", "exposed"}:
            raise ValueError("invalid Pennsylvania staff challenge manifest")
        status = prior["status"]
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_pa_agriculture_staff_challenge",
                                    "cases": len(rows), "sha256": digest(data),
                                    "status": status}, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} untouched Pennsylvania staff cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
