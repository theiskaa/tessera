"""Freeze unseen state agency and program acronym examples from official pages."""

import hashlib
import json
from pathlib import Path

from check_ready import normalized
from freeze_ca_superintendent_challenge import training_surfaces


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
OUTPUT = REVIEW / "us-pa-acronym-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-pa-acronym-challenge-v1.manifest.json"
FROZEN_SHA256 = "db181864d479f21002db00b8dac8f4a8f9c338be6faad2ff712e74b028aea4c9"
ROWS = (
    ("history", "https://www.pa.gov/agencies/phmc/about-phmc",
     "Pennsylvania Historical and Museum Commission (PHMC)",
     ("Pennsylvania Historical and Museum Commission", "PHMC")),
    ("justice", "https://www.pa.gov/agencies/pccd/about",
     "Pennsylvania Commission on Crime and Delinquency (PCCD)",
     ("Pennsylvania Commission on Crime and Delinquency", "PCCD")),
    ("gaming", "https://gamingcontrolboard.pa.gov/gaming/gaming-overview",
     "The Pennsylvania Gaming Control Board(PGCB) is an independent state agency.",
     ("Pennsylvania Gaming Control Board", "PGCB")),
    ("emergency", "https://www.pa.gov/agencies/pema/contact",
     "Contact PEMA", ("PEMA",)),
    ("retirement", "https://sers.pa.gov/",
     "Contact SERS with questions about your specific plan.", ("SERS",)),
    ("research_program", "https://www.pa.gov/agencies/health/research/research/cure",
     "Commonwealth Universal Research Enhancement program (CURE)", ()),
    ("open_records_law", "https://gamingcontrolboard.pa.gov/news-and-transparency/right-know",
     "RTKL Requests", ()),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    values, _, _ = training_surfaces()
    prior = [json.loads(line) for line in
             (REVIEW / "us-eval-exclusions-v1.jsonl").read_text().splitlines()]
    prior_orgs = {normalized(span["text"]) for row in prior
                  for span in row["expected"] if span["kind"] == "org"}
    rows = []
    seen = set()
    for slug, source_url, text, orgs in ROWS:
        labels = []
        for org in orgs:
            key = normalized(org)
            if key in values["org"] | prior_orgs | seen:
                raise ValueError(f"organization overlap: {org}")
            seen.add(key)
            start = len(text[:text.index(org)].encode())
            labels.append({"kind": "org", "start": start,
                           "end": start + len(org.encode()), "text": org})
        rows.append({"name": f"us-pa-acronym-{slug}", "country": "US",
                     "input": text, "expected": labels,
                     "doc_type": "state_agency_or_program", "source_url": source_url})
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in rows) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError("Pennsylvania acronym challenge differs from its frozen version")
    if MANIFEST.exists() and not OUTPUT.exists():
        raise ValueError("Pennsylvania acronym challenge data is missing")
    status = "unseen"
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != data or not MANIFEST.exists():
            raise ValueError("frozen Pennsylvania acronym challenge changed")
        prior = json.loads(MANIFEST.read_text())
        if prior["sha256"] != FROZEN_SHA256 or prior["status"] not in {"unseen", "exposed"}:
            raise ValueError("invalid Pennsylvania acronym challenge manifest")
        status = prior["status"]
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_pa_acronym_challenge",
                                    "cases": len(rows), "sha256": digest(data),
                                    "status": status}, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} untouched Pennsylvania acronym cases frozen: {digest(data)}")


if __name__ == "__main__":
    main()
