"""Build source-checked NPS contacts disjoint from every US evaluation label."""

import collections
import hashlib
import json
from pathlib import Path

from address_keys import address_keys
from check_ready import normalized, person_key
from freeze_park_address_challenge import DATA_URL, HASHES, SOURCE_URL, case_for, flat, validate_sources


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUT = ROOT / "data/interim/silver/us-park-contacts-v1.jsonl"
MANIFEST = ROOT / "data/interim/silver/us-park-contacts-v1.manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    csv_rows, source_rows = validate_sources()
    evaluation = [json.loads(line) for line in EVAL.read_text().splitlines()]
    surfaces = {(span["kind"], normalized(span["text"]))
                for row in evaluation for span in row["expected"]}
    people = {person_key(span["text"]) for row in evaluation
              for span in row["expected"] if span["kind"] == "person"}
    addresses = set().union(*(address_keys(span["text"]) for row in evaluation
                               for span in row["expected"] if span["kind"] == "address"))
    eval_texts = {row["input"] for row in evaluation}
    rows = []
    seen_texts = set()
    seen_people = set()
    counts = collections.Counter()
    for index, source_row in enumerate(source_rows):
        case = case_for(index, source_row)
        if case is None:
            continue
        csv_row = csv_rows.get(case["source_site"])
        if csv_row is None or any(flat(left) != flat(right)
                                  for left, right in zip(csv_row[:4], source_row[:4])):
            continue
        labels = [span for span in case["expected"] if span["kind"] in {"person", "address"}]
        person = next(span["text"] for span in labels if span["kind"] == "person")
        key = person_key(person)
        if (case["input"] in eval_texts or case["input"] in seen_texts or key in seen_people
                or key in people or any((span["kind"], normalized(span["text"])) in surfaces
                                     for span in labels)
                or any(address_keys(span["text"]) & addresses for span in labels
                       if span["kind"] == "address")):
            continue
        text = case["input"]
        entities = [{"kind": span["kind"], "start": span["start"], "end": span["end"]}
                    for span in labels]
        for span in labels:
            if text.encode()[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"bad source label at NPS row {index}")
            counts[span["kind"]] += 1
        rows.append({"id": f"nps-park-{index:03}", "source": "nps-parks-2023",
                     "source_url": SOURCE_URL, "source_data_url": DATA_URL,
                     "source_json_index": index, "country": "US", "text": text,
                     "entities": entities})
        seen_texts.add(text)
        seen_people.add(key)
    if len(rows) < 120:
        raise ValueError(f"too few clean NPS contacts: {len(rows)}")
    output = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    OUT.write_bytes(output)
    MANIFEST.write_text(json.dumps({
        "kind": "source_checked_us_park_contacts",
        "source_url": SOURCE_URL, "source_data_url": DATA_URL,
        "source_sha256": HASHES, "evaluation_gold_sha256": digest(EVAL.read_bytes()),
        "documents": len(rows), "labels": dict(counts), "sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")
    print(f"built {len(rows)} NPS contacts with {dict(counts)} labels")


if __name__ == "__main__":
    main()
