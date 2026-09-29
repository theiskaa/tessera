"""Freeze source-backed Pennsylvania address boundaries before training on room prefixes."""

import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from freeze_ca_superintendent_challenge import training_surfaces


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
OUTPUT = REVIEW / "us-pa-room-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-pa-room-challenge-v1.manifest.json"
FROZEN_SHA256 = "c38839db80c05c1eb5e16742e3fb4454f5f06a3a72670042d55f090f3a7845e7"
ROWS = (
    ("dgs", "https://www.pa.gov/agencies/dgs/about/dgs-directory",
     "Room 610-611 North Office Building\n6th Floor, 401 North Street\nHarrisburg, PA 17120"),
    ("health", "https://www.pa.gov/agencies/health/bureaus-and-offices",
     "Room 1023\n625 Forster St.\nHarrisburg, PA 17120-0701"),
    ("dhs", "https://www.pa.gov/agencies/dhs/resources/licensing/pch-alr-licensing/pch-alr-contact",
     "Norristown State Hospital Building #2, Room 256\n1001 Sterigere Street\nNorristown, PA 19401"),
    ("ethics", "https://www.pa.gov/agencies/ethics/contact-us",
     "Finance Building\n613 North Street, Room 304\nHarrisburg, PA 17120-0400"),
    ("labor", "https://www.dliweb.pa.gov/prevwage/Pages/DetermRequest.aspx?ID=&PageType=",
     "Labor & Industry Building\nRoom 1301\n651 Boas Street\nHarrisburg PA 17121"),
    ("state", "https://www.pa.gov/agencies/dos/contact-us",
     "321 Biden Street, Room 525\nScranton, PA 18503"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    values, _, training_addresses = training_surfaces()
    prior = [json.loads(line) for line in
             (REVIEW / "us-eval-exclusions-v1.jsonl").read_text().splitlines()]
    prior = [row for row in prior if not row["name"].startswith("us-pa-room-")]
    prior_values = {" ".join(span["text"].casefold().split()) for row in prior
                    for span in row["expected"] if span["kind"] == "address"}
    prior_addresses = set().union(*(address_keys(span["text"]) for row in prior
                                    for span in row["expected"] if span["kind"] == "address"))
    rows = []
    used_addresses = set()
    for slug, source_url, address in ROWS:
        key = " ".join(address.casefold().split())
        physical = address_keys(address)
        if (not physical or key in values["address"] | prior_values
                or physical & (training_addresses | prior_addresses | used_addresses)):
            raise ValueError(f"address overlap: {slug}")
        used_addresses.update(physical)
        text = "Mailing address:\n" + address
        start = len("Mailing address:\n".encode())
        rows.append({"name": f"us-pa-room-{slug}", "country": "US", "input": text,
                     "expected": [{"kind": "address", "start": start,
                                   "end": start + len(address.encode()), "text": address}],
                     "doc_type": "state_office_contact", "source_url": source_url})
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in rows) + "\n").encode()
    if digest(data) != FROZEN_SHA256:
        raise ValueError("Pennsylvania room challenge differs from its frozen version")
    if MANIFEST.exists() and not OUTPUT.exists():
        raise ValueError("Pennsylvania room challenge data is missing")
    status = "unseen"
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != data or not MANIFEST.exists():
            raise ValueError("frozen Pennsylvania room challenge changed")
        prior = json.loads(MANIFEST.read_text())
        if prior["sha256"] != FROZEN_SHA256 or prior["status"] not in {"unseen", "exposed"}:
            raise ValueError("invalid Pennsylvania room challenge manifest")
        status = prior["status"]
    OUTPUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({"kind": "us_pa_room_challenge", "cases": len(rows),
                                    "sha256": digest(data), "status": status},
                                   indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} untouched Pennsylvania room addresses frozen: {digest(data)}")


if __name__ == "__main__":
    main()
