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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def classify(source, others):
    """Consume exact matches once; repeated predictions remain errors."""
    key = lambda span: (span["kind"], span["start"], span["end"])
    remaining = collections.Counter(key(span) for span in others)
    statuses = []
    for span in source:
        identity = key(span)
        if remaining[identity]:
            remaining[identity] -= 1
            statuses.append("exact")
        else:
            status = category(span, others)
            statuses.append("duplicate" if status == "exact" else status)
    return statuses


def load_predictions(path, gold_bytes, gold):
    content = json.loads(path.read_text())
    if content["provenance"]["gold_sha256"] != hashlib.sha256(gold_bytes).hexdigest():
        raise ValueError("prediction file uses a different gold set")
    rows = content["predictions"]
    predictions = {row["name"]: row["entities"] for row in rows}
    if (len(predictions) != len(rows) or len(predictions) != len(gold)
            or set(predictions) != {row["name"] for row in gold}):
        raise ValueError("predictions omit or duplicate gold cases")
    for row in gold:
        text = row["input"].encode()
        for span in predictions[row["name"]]:
            if (span["kind"] not in KINDS or type(span["start"]) is not int
                    or type(span["end"]) is not int
                    or not 0 <= span["start"] < span["end"] <= len(text)):
                raise ValueError("invalid prediction span")
            text[span["start"]:span["end"]].decode("utf-8")
    return content, predictions


def validate_gold(gold):
    if not gold or len({row["name"] for row in gold}) != len(gold):
        raise ValueError("gold cases are empty or have duplicate names")
    for row in gold:
        text = row["input"].encode()
        keys = set()
        for span in row["expected"]:
            key = (span["kind"], span["start"], span["end"])
            if (span["kind"] not in KINDS or type(span["start"]) is not int
                    or type(span["end"]) is not int
                    or not 0 <= span["start"] < span["end"] <= len(text)
                    or key in keys):
                raise ValueError("invalid or duplicate gold span")
            value = text[span["start"]:span["end"]].decode("utf-8")
            if "text" in span and span["text"] != value:
                raise ValueError("gold span does not slice to its text")
            keys.add(key)


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
    parser.add_argument("--raw-predictions", type=Path,
                        help="decoded candidates before confidence filtering")
    parser.add_argument("--examples", choices=KINDS)
    parser.add_argument("--category", choices=("boundary", "wrong_kind", "no_overlap", "duplicate"))
    parser.add_argument("--direction", choices=("gold", "pred"), default="gold")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--source-groups", action="store_true")
    args = parser.parse_args()
    gold_bytes = args.gold.read_bytes()
    gold = load_rows(args.gold)
    validate_gold(gold)
    prediction_file, predictions = load_predictions(args.predictions, gold_bytes, gold)
    raw = None
    if args.raw_predictions:
        raw_file, raw = load_predictions(args.raw_predictions, gold_bytes, gold)
        for key in ("best_sha256", "config_sha256", "bundle_sha256", "decoder_contract", "detector_decoder", "diagnostic_decoder"):
            if raw_file["provenance"].get(key) != prediction_file["provenance"].get(key):
                raise ValueError("raw and filtered predictions use different models")
    covered_kinds = KINDS
    if prediction_file.get("system") == "detector_f32_model_only":
        covered_kinds = KINDS[:3]
        print("Scope: neural model spans; email and phone rules are not evaluated.")
    counts = collections.defaultdict(collections.Counter)
    slices = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    source_groups = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    org_shapes = collections.defaultdict(collections.Counter)
    examples = []
    for row in gold:
        expected = row["expected"]
        predicted = predictions[row["name"]]
        for source, others, direction in ((expected, predicted, "gold"),
                                          (predicted, expected, "pred")):
            for span, status in zip(source, classify(source, others)):
                counts[span["kind"]][f"{direction}_{status}"] += 1
                doc_type = row.get("doc_type", "training_seen")
                slices[doc_type][span["kind"]][f"{direction}_{status}"] += 1
                source_groups[row.get("source_group", "unknown")][span["kind"]][
                    f"{direction}_{status}"] += 1
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
                    examples.append((row["name"], value, doc_type, context))
    print("kind  gold  exact  boundary  wrong_kind  no_overlap  pred_nonexact")
    for kind in covered_kinds:
        c = counts[kind]
        print(f"{kind:7} {sum(c[f'gold_{status}'] for status in ('exact','boundary','wrong_kind','no_overlap')):4} "
              f"{c['gold_exact']:5} {c['gold_boundary']:8} {c['gold_wrong_kind']:11} "
              f"{c['gold_no_overlap']:10} {sum(c[f'pred_{status}'] for status in ('boundary','wrong_kind','no_overlap','duplicate')):11}")
    if raw is not None:
        print("\nmissed gold: correct raw candidate filtered / wrong boundary / wrong kind / no candidate")
        for kind in covered_kinds:
            c = collections.Counter()
            for row in gold:
                filtered_status = classify(row["expected"], predictions[row["name"]])
                raw_status = classify(row["expected"], raw[row["name"]])
                for span, after, before in zip(row["expected"], filtered_status, raw_status):
                    if span["kind"] == kind and after != "exact":
                        c[before] += 1
            print(f"{kind}: {c['exact']} / {c['boundary']} / {c['wrong_kind']} / {c['no_overlap']}")
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
              f"{sum(c[f'pred_{status}'] for status in ('boundary','wrong_kind','no_overlap','duplicate'))}")
    if args.source_groups:
        print("\nsource group  kind  gold  predicted  exact  exact_f1")
        for group, kinds in sorted(source_groups.items()):
            for kind in covered_kinds:
                c = kinds[kind]
                gold_count = sum(c[f"gold_{status}"] for status in
                                 ("exact", "boundary", "wrong_kind", "no_overlap"))
                predicted_count = sum(c[f"pred_{status}"] for status in
                                      ("exact", "boundary", "wrong_kind", "no_overlap", "duplicate"))
                if gold_count or predicted_count:
                    exact = c["gold_exact"]
                    f1 = 200 * exact / (gold_count + predicted_count)
                    print(f"{group}  {kind}  {gold_count}  {predicted_count}  "
                          f"{exact}  {f1:.1f}")
    for name, value, doc_type, text in examples[:args.limit]:
        print(f"{name} [{doc_type}] {value!r}: {text}")


if __name__ == "__main__":
    main()
