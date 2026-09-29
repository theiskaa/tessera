"""Freeze independent California directory contacts and check training overlap."""

import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from check_ready import normalized, person_key


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
SILVER = ROOT / "data/interim/silver"
OUT = REVIEW / "us-ca-superintendent-challenge-v1.jsonl"
MANIFEST = REVIEW / "us-ca-superintendent-challenge-v1.manifest.json"
BASE_URL = "https://sd.cde.ca.gov/schooldirectory/details?cdscode="
FROZEN_SHA256 = "1bbe1e70652dd2b244e591f5a8717374e02968e8487ed3cd85f8bce74974177d"
ROWS = (
    ("37103710000000", "San Diego County Office of Education", "Gloria Ciriza", "6401 Linda Vista Rd.\nSan Diego, CA 92111-7319", "(858) 292-3500"),
    ("38684780000000", "San Francisco Unified", "Maria Su", "555 Franklin St.\nSan Francisco, CA 94102-5207", "(415) 241-6000"),
    ("37679830000000", "Borrego Springs Unified", "Mark Stevens", "2281 Diegueno Rd.\nBorrego Springs, CA 92004-0235", "(760) 767-5357"),
    ("24753660000000", "Delhi Unified", "Eric Griffin", "9716 Hinton Ave.\nDelhi, CA 95315-9455", "(209) 656-2000"),
    ("37682210000000", "National Elementary", "Laura Philyaw", "1500 N Ave.\nNational City, CA 91950-4827", "(619) 336-7500"),
    ("19645680000000", "Glendale Unified", "Darneika Watson", "223 North Jackson St.\nGlendale, CA 91206-4334", "(818) 241-3111"),
    ("42767860000000", "Santa Barbara Unified", "Hilda Maldonado", "720 Santa Barbara St.\nSanta Barbara, CA 93101-2232", "(805) 963-4338"),
    ("49402530000000", "Santa Rosa City Schools", "Monica Thomas", "110 Stony Point Rd.\nSte. 210\nSanta Rosa, CA 95401-4118", "(707) 890-3800"),
    ("48705320000000", "Dixon Unified", "Brett Barley", "180 South First St., Ste. 6\nDixon, CA 95620-3447", "(707) 693-6300"),
    ("01612910000000", "San Leandro Unified", "Mike McLaughlin", "2600 Teagarden St.\nSan Leandro, CA 94577-3767", "(510) 667-3500"),
    ("20652430000000", "Madera Unified", "Todd Lile", "1902 Howard Rd.\nMadera, CA 93637-5123", "(559) 675-4500"),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def labels(text, values):
    result = []
    for kind, value in values:
        start = text.index(value)
        start = len(text[:start].encode())
        result.append({"kind": kind, "start": start,
                       "end": start + len(value.encode()), "text": value})
    return result


def training_surfaces():
    import pyarrow.parquet as parquet

    values = {"person": set(), "org": set(), "address": set()}
    people = set()
    addresses = set()

    def add(text, spans):
        data = text.encode()
        for span in spans:
            kind = span["kind"]
            if kind not in values:
                continue
            value = data[span["start"]:span["end"]].decode()
            values[kind].add(normalized(value))
            if kind == "person":
                people.add(person_key(value))
            if kind == "address":
                addresses.update(address_keys(value))

    for split in ("train", "valid", "test"):
        path = ROOT / f"data/processed/detector-us-v1/{split}.parquet"
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=10000,
                                                            columns=["text", "entities_json"]):
            for row in batch.to_pylist():
                add(row["text"], json.loads(row["entities_json"]))
    for path in sorted(SILVER.glob("us-*-v1.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            add(row["text"], row["entities"])
    for split in ("train", "valid", "test"):
        path = ROOT / f"data/processed/parser-us-v1/{split}.parquet"
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=10000, columns=["text"]):
            for row in batch.to_pylist():
                values["address"].add(normalized(row["text"]))
                addresses.update(address_keys(row["text"]))
    return values, people, addresses


def main():
    values, people, addresses = training_surfaces()
    prior = [json.loads(line) for line in
             (REVIEW / "us-eval-exclusions-v1.jsonl").read_text().splitlines()]
    prior = [row for row in prior if not row["name"].startswith("us-ca-superintendent-")]
    prior_texts = {row["input"] for row in prior}
    prior_surfaces = {(span["kind"], normalized(span["text"]))
                      for row in prior for span in row["expected"]}
    prior_people = {person_key(span["text"]) for row in prior
                    for span in row["expected"] if span["kind"] == "person"}
    prior_addresses = set().union(*(address_keys(span["text"]) for row in prior
                                    for span in row["expected"] if span["kind"] == "address"))
    rows = []
    dropped = []
    for code, district, person, address, phone in ROWS:
        text = f"District: {district}\nSuperintendent: {person}\nAddress: {address}\nPhone: {phone}"
        expected = labels(text, (("org", district), ("person", person),
                                 ("address", address), ("phone", phone)))
        if (text in prior_texts or person_key(person) in people | prior_people
                or address_keys(address) & (addresses | prior_addresses)
                or any(normalized(value) in values[kind]
                       for kind, value in (("org", district), ("person", person),
                                           ("address", address)))
                or any((kind, normalized(value)) in prior_surfaces
                       for kind, value in (("org", district), ("person", person),
                                           ("address", address)))):
            dropped.append(code)
            continue
        for span in expected:
            if text.encode()[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"bad California label: {code}")
        rows.append({"name": f"us-ca-superintendent-{code}", "country": "US",
                     "input": text, "expected": expected,
                     "doc_type": "school_district_contact",
                     "source_url": BASE_URL + code,
                     "source_representation": "official directory fields transcribed into labeled contact lines"})
    if len(rows) < 8:
        raise ValueError(f"too few disjoint California cases: {len(rows)}; dropped {dropped}")
    output = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    if digest(output) != FROZEN_SHA256:
        raise ValueError("California holdout changed after freeze")
    OUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "us_post_start_independent_school_directory_challenge",
        "status": "exposed in role-v2 evaluation; rows excluded from later training",
        "source_capture": "manual transcription of official directory fields; self-reported by districts",
        "source_urls": [row["source_url"] for row in rows],
        "training_snapshot_sha256": digest((ROOT / "runs/detector-us-role-v2/input_snapshot.json").read_bytes()),
        "cases": len(rows), "dropped_training_overlap": dropped,
        "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(rows)} California cases; overlap drops {dropped}; sha256 {digest(output)}")


if __name__ == "__main__":
    main()
