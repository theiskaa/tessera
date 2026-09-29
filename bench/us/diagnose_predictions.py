"""Classify exact-span errors from a shipped bundle's saved predictions."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


KINDS = ("person", "org", "address", "email", "phone")
ACRONYM = re.compile(r"[A-Z][A-Z0-9]{1,8}\Z")


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def overlap(left, right):
    return left["start"] < right["end"] and right["start"] < left["end"]


def category(span, others):
    if any(other["kind"] == span["kind"] and
           other["start"] == span["start"] and other["end"] == span["end"]
           for other in others):
        return "exact"
    if any(other["kind"] == span["kind"] and overlap(span, other) for other in others):
        return "boundary"
    if any(overlap(span, other) for other in others):
        return "wrong_kind"
    return "no_overlap"


def org_shape(value):
    if ACRONYM.fullmatch(value):
        return "acronym"
    if len(value.split()) >= 4:
        return "long_name"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--examples", choices=KINDS)
    parser.add_argument("--category", choices=("boundary", "wrong_kind", "no_overlap"))
    parser.add_argument("--direction", choices=("gold", "pred"), default="gold")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    gold_bytes = args.gold.read_bytes()
    gold = load_rows(args.gold)
    prediction_file = json.loads(args.predictions.read_text())
    if prediction_file["provenance"]["gold_sha256"] != hashlib.sha256(gold_bytes).hexdigest():
        raise ValueError("prediction file uses a different gold set")
    prediction_rows = prediction_file["predictions"]
    predictions = {row["name"]: row["entities"] for row in prediction_rows}
    if (len(predictions) != len(prediction_rows) or len(predictions) != len(gold)
            or set(predictions) != {row["name"] for row in gold}):
        raise ValueError("predictions omit or duplicate gold cases")
    counts = collections.defaultdict(collections.Counter)
    slices = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    org_shapes = collections.defaultdict(collections.Counter)
    examples = []
    for row in gold:
        expected = row["expected"]
        predicted = predictions[row["name"]]
        for source, others, direction in ((expected, predicted, "gold"),
                                          (predicted, expected, "pred")):
            for span in source:
                status = category(span, others)
                counts[span["kind"]][f"{direction}_{status}"] += 1
                slices[row["doc_type"]][span["kind"]][f"{direction}_{status}"] += 1
                if span["kind"] == "org":
                    value = row["input"].encode()[span["start"]:span["end"]].decode()
                    org_shapes[org_shape(value)][f"{direction}_{status}"] += 1
                if (args.examples == span["kind"] and args.category == status
                        and status != "exact" and direction == args.direction):
                    source = row["input"].encode()
                    value = source[span["start"]:span["end"]].decode()
                    context = source[max(0, span["start"] - 70):
                                     min(len(source), span["end"] + 70)].decode(
                                         "utf-8", "replace").replace("\n", " ")
                    examples.append((row["name"], value, row["doc_type"], context))
    print("kind  gold  exact  boundary  wrong_kind  no_overlap  pred_nonexact")
    for kind in KINDS:
        c = counts[kind]
        print(f"{kind:7} {sum(c[f'gold_{status}'] for status in ('exact','boundary','wrong_kind','no_overlap')):4} "
              f"{c['gold_exact']:5} {c['gold_boundary']:8} {c['gold_wrong_kind']:11} "
              f"{c['gold_no_overlap']:10} {sum(c[f'pred_{status}'] for status in ('boundary','wrong_kind','no_overlap')):11}")
    print("\norganization by document type: gold / exact / boundary / no overlap")
    for doc_type, kinds in sorted(slices.items()):
        c = kinds["org"]
        total = sum(c[f"gold_{status}"] for status in
                    ("exact", "boundary", "wrong_kind", "no_overlap"))
        if total:
            print(f"{doc_type}: {total} / {c['gold_exact']} / "
                  f"{c['gold_boundary']} / {c['gold_no_overlap']}")
    print("\norganization by surface shape: gold / exact / boundary / no overlap / predicted nonexact")
    for shape in ("acronym", "long_name", "other"):
        c = org_shapes[shape]
        total = sum(c[f"gold_{status}"] for status in
                    ("exact", "boundary", "wrong_kind", "no_overlap"))
        print(f"{shape}: {total} / {c['gold_exact']} / {c['gold_boundary']} / "
              f"{c['gold_no_overlap']} / "
              f"{sum(c[f'pred_{status}'] for status in ('boundary','wrong_kind','no_overlap'))}")
    for name, value, doc_type, text in examples[:args.limit]:
        print(f"{name} [{doc_type}] {value!r}: {text}")


if __name__ == "__main__":
    main()
