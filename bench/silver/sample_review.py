"""Sample labelled silver documents for manual review and write the review sheet.

Run from the repository root after `label.py`:

    python bench/silver/sample_review.py [round]

Writes `data/raw/silver/review-sample[-rN].jsonl` and `data/raw/silver/review-sheet[-rN].csv`
with one row per predicted span. The reviewer fills `verdict` and appends a row with an empty
`score` and `verdict = missing` for every entity the teacher missed.
"""

import csv
import json
import pathlib
import random
import sys

SOURCES = ["data/interim/silver/federal-register.jsonl", "data/interim/silver/govuk.jsonl"]
PER_SOURCE = 100
SEED = 42
CONTEXT = 60

INSTRUCTIONS = """\
Fill `verdict` for every row with one of: correct, wrong_kind, wrong_boundary, not_entity.
Mark `correct` only when the kind and both boundaries are right. A person name with a
trailing job title is `wrong_boundary`. A street name alone (no number, no city) predicted
as address is `not_entity` unless it is the whole address given. A government agency is
`org`. A city alone is not an address.
Then read each sampled document in review-sample.jsonl and append a row for every person,
org, or address the teacher missed: same columns, empty `score`, verdict `missing`.
"""


def span_text(text, start, end):
    b = text.encode("utf-8")
    return b[start:end].decode("utf-8")


def context(text, start, end):
    b = text.encode("utf-8")
    before = b[max(0, start - CONTEXT):start].decode("utf-8", "ignore")
    after = b[end:end + CONTEXT].decode("utf-8", "ignore")
    mark = f"{before}[[{b[start:end].decode('utf-8')}]]{after}"
    return " ".join(mark.split())


def main(round_no=1):
    suffix = "" if round_no == 1 else f"-r{round_no}"
    rng = random.Random(SEED + round_no - 1)
    out = pathlib.Path("data/raw/silver")
    out.mkdir(parents=True, exist_ok=True)
    sample = []
    for src in SOURCES:
        docs = [json.loads(l) for l in pathlib.Path(src).read_text().splitlines() if l]
        sample.extend(rng.sample(docs, min(PER_SOURCE, len(docs))))
    (out / f"review-sample{suffix}.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in sample))
    rows = 0
    with (out / f"review-sheet{suffix}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "kind", "start", "end", "text", "score", "verdict", "note", "context"])
        for d in sample:
            for e in d["entities"]:
                w.writerow([d["id"], e["kind"], e["start"], e["end"],
                            span_text(d["text"], e["start"], e["end"]), e["score"], "", "",
                            context(d["text"], e["start"], e["end"])])
                rows += 1
    print(f"{len(sample)} documents, {rows} predicted spans in {out}/review-sheet{suffix}.csv\n")
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
