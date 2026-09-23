"""Score a reviewed silver sheet and record which kinds the teacher is trusted for.

Run from the repository root after filling the sheet:

    python bench/silver/score_review.py [round] [--reviewers N] [--reviewed-by WHO]

Writes `data/manifests/silver-review.json`, keeping earlier rounds.
"""

import argparse
import collections
import csv
import json
import pathlib
import sys

KINDS = ["person", "org", "address"]
VERDICTS = {"correct", "wrong_kind", "wrong_boundary", "not_entity", "missing"}
MIN_PRECISION = 0.85
MIN_RECALL = 0.80
MANIFEST = pathlib.Path("data/manifests/silver-review.json")


def score(rows):
    """Per kind: verdict counts, precision, and estimated recall."""
    counts = {k: collections.Counter() for k in KINDS}
    for r in rows:
        counts[r["kind"]][r["verdict"]] += 1
    out = {}
    for k, c in counts.items():
        predicted = c["correct"] + c["wrong_kind"] + c["wrong_boundary"] + c["not_entity"]
        relevant = c["correct"] + c["missing"] + c["wrong_boundary"]
        precision = c["correct"] / predicted if predicted else 0.0
        recall = c["correct"] / relevant if relevant else 0.0
        out[k] = {
            "counts": dict(c),
            "precision": round(precision, 4),
            "recall_estimate": round(recall, 4),
            "accepted": precision >= MIN_PRECISION and recall >= MIN_RECALL,
        }
    return out


def sweep(rows, thresholds=(0.6, 0.7, 0.8, 0.9)):
    """Precision and recall per kind if only predictions scoring at least `t` were kept. A
    higher threshold keeps a subset of the reviewed predictions, so the reviewed sheet
    measures it exactly: dropped correct or mis-bounded spans become misses."""
    out = {}
    for t in thresholds:
        per = {}
        for k in KINDS:
            rel = correct = kept = 0
            for r in rows:
                if r["kind"] != k:
                    continue
                v = r["verdict"]
                if v in ("correct", "wrong_boundary", "missing"):
                    rel += 1
                if v == "missing" or float(r["score"]) < t:
                    continue
                kept += 1
                correct += v == "correct"
            per[k] = {"precision": round(correct / kept, 4) if kept else 0.0,
                      "recall_estimate": round(correct / rel, 4) if rel else 0.0}
        out[str(t)] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", nargs="?", type=int, default=1)
    ap.add_argument("--reviewers", type=int, default=1)
    ap.add_argument("--reviewed-by", default="human",
                    help="who filled the verdicts: `human` or `model`")
    args = ap.parse_args()
    suffix = "" if args.round == 1 else f"-r{args.round}"
    sheet = pathlib.Path(f"data/raw/silver/review-sheet{suffix}.csv")
    with sheet.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r["verdict"] not in VERDICTS or r["kind"] not in KINDS]
    if bad:
        sys.exit(f"{len(bad)} rows without a valid kind and verdict, first: {bad[0]}")
    result = score(rows)
    print(f"{'kind':<9}{'precision':>10}{'recall':>9}  accepted")
    for k, v in result.items():
        print(f"{k:<9}{v['precision']:>10.3f}{v['recall_estimate']:>9.3f}  {v['accepted']}")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"rounds": []}
    manifest["rounds"] = [r for r in manifest["rounds"] if r["round"] != args.round]
    manifest["rounds"].append({
        "round": args.round,
        "sheet_rows": len(rows),
        "documents": len({r["doc_id"] for r in rows}),
        "reviewers": args.reviewers,
        "reviewed_by": args.reviewed_by,
        "sample_seed": 42 + args.round - 1,
        "per_kind": result,
        "threshold_sweep": sweep(rows),
    })
    manifest["rounds"].sort(key=lambda r: r["round"])
    latest = manifest["rounds"][-1]["per_kind"]
    manifest.update({
        "source": "silver-review",
        "thresholds": {"precision": MIN_PRECISION, "recall_estimate": MIN_RECALL},
        "accepted_kinds": [k for k in KINDS if latest[k]["accepted"]],
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"accepted kinds: {manifest['accepted_kinds']}")


if __name__ == "__main__":
    main()
