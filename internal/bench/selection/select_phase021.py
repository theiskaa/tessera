"""Rank completed detector epochs from real-document reports only.

This is a post-training gate. It neither trains nor evaluates a model. Each epoch
must have a gated quantized file, exported bundle, and `eval --gold` report.
It checks the checkpoint, quantization gate, export metadata, and report chain.
Passing this selector does not approve release.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import dev_v2_lock

COUNTRIES = ("US", "GB", "DE", "GE", "JP")
KINDS = ("person", "org", "address", "email", "phone")
TARGETS = {"person_exact_f1": 0.90, "address_exact_f1": 0.80,
           "address_found_recall": 0.95}
REGRESSION_TOLERANCE = 0.02
MIN_REGRESSION_SUPPORT = 50
EXPECTED_GOLD = "3940db21008a51dc71e78fdb390b1be3b1f69ca4596b4c49b50da0af77b2abbd"
EXPECTED_BASELINE = "cb056415282f79504c5b14f9766f126fc5379cf3941503bb030a00fec530b4c6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def gold_support(path):
    names = set()
    counts = Counter()
    countries = {}
    cases = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name, country = row["name"], row["country"]
        if name in names:
            raise ValueError(f"duplicate gold case: {name}")
        names.add(name)
        cases += 1
        country_counts = countries.setdefault(country, Counter())
        for span in row["expected"]:
            kind = span["kind"]
            if kind not in KINDS:
                raise ValueError(f"unknown gold kind: {kind}")
            counts[kind] += 1
            country_counts[kind] += 1
    return {"cases": cases, "by_kind": counts, "by_country": countries}


def valid_rate(value, label):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{label}: invalid metric")


def verify_metrics(metrics, expected_gold, label):
    for field in ("gold", "predicted", "exact_tp"):
        value = metrics.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}: invalid {field} count")
    gold, predicted, correct = (metrics[field] for field in ("gold", "predicted", "exact_tp"))
    if gold != expected_gold or correct > min(gold, predicted):
        raise ValueError(f"{label}: gold support or exact match count differs")
    exact = metrics.get("exact")
    if not isinstance(exact, dict):
        raise ValueError(f"{label}: missing exact scores")
    calculated = {"precision": correct / predicted if predicted else 0.0,
                  "recall": correct / gold if gold else 0.0,
                  "f1": 2 * correct / (gold + predicted) if gold + predicted else 0.0}
    for metric, expected in calculated.items():
        value = exact.get(metric)
        valid_rate(value, f"{label} exact {metric}")
        if abs(value - expected) > 0.000051:
            raise ValueError(f"{label}: exact {metric} arithmetic differs")
    for group, predicted_field, gold_field in (
        ("lenient", "lenient_predicted_hits", "lenient_gold_hits"),
        ("lenient_one_to_one", "one_to_one_predicted_hits", "one_to_one_gold_hits"),
    ):
        values = metrics.get(group)
        if not isinstance(values, dict):
            raise ValueError(f"{label}: missing {group} scores")
        predicted_hits = metrics.get(predicted_field)
        gold_hits = metrics.get(gold_field)
        if (not isinstance(predicted_hits, int) or isinstance(predicted_hits, bool)
                or not correct <= predicted_hits <= predicted
                or not isinstance(gold_hits, int) or isinstance(gold_hits, bool)
                or not correct <= gold_hits <= gold):
            raise ValueError(f"{label}: invalid {group} hit counts")
        precision = predicted_hits / predicted if predicted else 0.0
        recall = gold_hits / gold if gold else 0.0
        expected_rates = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall)
                          if precision + recall else 0.0}
        for metric, expected in expected_rates.items():
            value = values.get(metric)
            valid_rate(value, f"{label} {group} {metric}")
            if abs(value - expected) > 0.000051:
                raise ValueError(f"{label}: {group} {metric} arithmetic differs")
    valid_rate(metrics.get("boundary_accuracy"), f"{label} boundary accuracy")
    return predicted, correct


def verify_support(report, support, label):
    systems = [system for system in report.get("systems", [])
               if system.get("system") == "tessera"]
    if len(systems) != 1 or systems[0].get("cases") != support["cases"]:
        raise ValueError(f"{label}: report case count differs from selected gold")
    system = systems[0]
    by_kind = system.get("by_kind")
    country = system.get("by_slice", {}).get("country")
    if (not isinstance(by_kind, dict) or set(by_kind) != set(KINDS)
            or not isinstance(country, dict) or set(country) != set(support["by_country"])):
        raise ValueError(f"{label}: report slices differ from selected gold")
    kind_totals = {}
    for kind in KINDS:
        kind_totals[kind] = verify_metrics(by_kind[kind], support["by_kind"][kind],
                                           f"{label} all/{kind}")
    predicted, correct = verify_metrics(system["overall"], sum(support["by_kind"].values()),
                                        f"{label} overall")
    if predicted != sum(values[0] for values in kind_totals.values()) or \
            correct != sum(values[1] for values in kind_totals.values()):
        raise ValueError(f"{label}: overall counts differ from kind slices")
    for kind in KINDS:
        country_predicted = country_correct = 0
        for country_name, expected in support["by_country"].items():
            kinds = country[country_name]
            if kind not in kinds:
                if expected[kind]:
                    raise ValueError(f"{label}: missing {country_name}/{kind} gold slice")
                continue
            values = verify_metrics(kinds[kind], expected[kind],
                                    f"{label} {country_name}/{kind}")
            country_predicted += values[0]
            country_correct += values[1]
        if (country_predicted, country_correct) != kind_totals[kind]:
            raise ValueError(f"{label}: {kind} country totals differ from kind slice")


def bundle_metadata(path):
    with Path(path).open("rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        if not 0 < header_size <= 16_000_000:
            raise ValueError(f"{path}: invalid safetensors header size")
        header = json.loads(file.read(header_size))
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing export metadata")
    return metadata


def report_scores(report, label):
    if report.get("country_hint_mode") != "auto":
        raise ValueError(f"{label}: expected automatic country hints")
    systems = [system for system in report.get("systems", [])
               if system.get("system") == "tessera"]
    if len(systems) != 1:
        raise ValueError(f"{label}: expected exactly one Tessera score")
    by_country = systems[0]["by_slice"]["country"]
    scores = {}
    for country in COUNTRIES:
        try:
            kinds = by_country[country]
            if kinds["person"]["gold"] <= 0 or kinds["address"]["gold"] <= 0:
                raise ValueError(f"{label}: {country} has no person or address gold")
            person = kinds["person"]
            address = kinds["address"]
            values = {
                "person_exact_f1": 2 * person["exact_tp"] /
                                   (person["gold"] + person["predicted"]),
                "address_exact_f1": 2 * address["exact_tp"] /
                                    (address["gold"] + address["predicted"]),
                "address_found_recall": address["one_to_one_gold_hits"] / address["gold"],
            }
        except (KeyError, TypeError) as error:
            raise ValueError(f"{label}: missing {country} score: {error}") from error
        for metric, value in values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label}: invalid {country} {metric} score")
            scores[f"{country}.{metric}"] = float(value)
    return scores


def regression_scores(report):
    system = next(system for system in report["systems"] if system["system"] == "tessera")
    scores = {}

    def add_exact(key, gold, predicted, correct):
        scores[key] = (2 * correct / (gold + predicted) if gold + predicted else 0.0, gold)

    overall = system["overall"]
    add_exact("all.overall_exact_f1", overall["gold"], overall["predicted"],
              overall["exact_tp"])
    for kind in KINDS:
        metrics = system["by_kind"][kind]
        add_exact(f"all.{kind}_exact_f1", metrics["gold"], metrics["predicted"],
                  metrics["exact_tp"])
    for country in COUNTRIES:
        totals = Counter()
        kinds = system["by_slice"]["country"][country]
        for kind in KINDS:
            metrics = kinds.get(kind, {})
            gold, predicted, correct = (metrics.get(field, 0)
                                        for field in ("gold", "predicted", "exact_tp"))
            add_exact(f"{country}.{kind}_exact_f1", gold, predicted, correct)
            totals.update({"gold": gold, "predicted": predicted, "exact_tp": correct})
            if kind == "address":
                found = metrics.get("one_to_one_gold_hits", 0)
                scores[f"{country}.address_found_recall"] = (found / gold if gold else 0.0,
                                                             gold)
        add_exact(f"{country}.overall_exact_f1", totals["gold"], totals["predicted"],
                  totals["exact_tp"])
    return scores


def verify_report(path, bundle, gold_hash, evaluator_hash, label, support):
    report = read_json(path)
    provenance = report.get("provenance", {})
    if provenance.get("gold_sha256") != gold_hash:
        raise ValueError(f"{label}: gold hash differs from the frozen input")
    if provenance.get("evaluator_sha256") != evaluator_hash:
        raise ValueError(f"{label}: evaluator hash differs from the baseline")
    if provenance.get("bundle_sha256") != digest(bundle):
        raise ValueError(f"{label}: bundle hash differs from the evaluated bundle")
    verify_support(report, support, label)
    return report_scores(report, label), regression_scores(report)


def completed_epochs(run_dir):
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        raise ValueError("run has no completed training summary")
    summary = read_json(summary_path)
    metrics = Path(run_dir) / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    epochs = [row.get("epoch") for row in rows]
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError("training metrics have missing or repeated epoch numbers")
    if not isinstance(summary.get("steps"), int) or summary["steps"] <= 0 or \
            summary.get("best_epoch") not in epochs:
        raise ValueError("run has no valid completed training summary")
    return epochs


def select(baseline_report, baseline_bundle, manifest_path, run_dir, gold, evaluator,
           development_set="dev-v1"):
    gold_hash = digest(gold)
    if development_set == "dev-v1":
        expected_gold = EXPECTED_GOLD
        v17_scores = None
    elif development_set == "dev-v2":
        v17_scores = dev_v2_lock.verify(gold)
        expected_gold = dev_v2_lock.GOLD_SHA256
    else:
        raise ValueError(f"unknown development set: {development_set}")
    if gold_hash != expected_gold:
        raise ValueError("gold file differs from the frozen Phase 021 development set")
    support = gold_support(gold)
    if digest(baseline_bundle) != EXPECTED_BASELINE:
        raise ValueError("baseline bundle differs from the frozen V17 reference")
    if v17_scores is not None and digest(baseline_report) != dev_v2_lock.NATIVE_V17_REPORT_SHA256:
        raise ValueError("dev-v2 V17 report differs from the pinned native baseline")
    evaluator_hash = digest(evaluator)
    baseline, baseline_regression = verify_report(baseline_report, baseline_bundle, gold_hash,
                                                  evaluator_hash, "baseline", support)
    if v17_scores is not None:
        dev_v2_lock.verify_native_v17(read_json(baseline_report), v17_scores)
    manifest = read_json(manifest_path)
    entries = manifest.get("epochs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("epoch manifest is empty")
    actual_epochs = completed_epochs(run_dir)
    epoch_numbers = [entry.get("epoch") for entry in entries]
    if epoch_numbers != actual_epochs:
        raise ValueError("epoch manifest must contain every completed epoch in order")
    ranked = []
    decisions = []
    for entry in entries:
        epoch = entry["epoch"]
        checkpoint = Path(run_dir) / "checkpoints" / f"epoch-{epoch}.mpk"
        checkpoint_hash = digest(checkpoint)
        if checkpoint_hash != entry.get("checkpoint_sha256"):
            raise ValueError(f"epoch {epoch}: checkpoint hash mismatch")
        gate = read_json(entry["gate"])
        artifact_hashes = {
            "best_sha256": checkpoint_hash,
            "quantized_sha256": digest(entry["quantized"]),
            "input_snapshot_sha256": digest(entry["input_snapshot"]),
            "config_sha256": digest(entry["config"]),
        }
        if artifact_hashes["config_sha256"] != digest(Path(run_dir) / "config.toml") or \
                artifact_hashes["input_snapshot_sha256"] != digest(
                    Path(run_dir) / "input_snapshot.json"):
            raise ValueError(f"epoch {epoch}: staged config or inputs differ from the trained run")
        if gate.get("gate_version") != 2 or gate.get("passed") is not True or any(
                gate.get(key) != value for key, value in artifact_hashes.items()):
            raise ValueError(f"epoch {epoch}: quantization gate does not bind the checkpoint and weights")
        metadata = bundle_metadata(entry["bundle"])
        for key in ("best_sha256", "quantized_sha256", "input_snapshot_sha256"):
            if metadata.get(f"detector_{key}") != artifact_hashes[key]:
                raise ValueError(f"epoch {epoch}: exported bundle does not bind detector {key}")
        scores, regression = verify_report(entry["report"], entry["bundle"], gold_hash,
                                           evaluator_hash, f"epoch {epoch}", support)
        if regression.keys() != baseline_regression.keys() or any(
                regression[key][1] != baseline_regression[key][1]
                for key in baseline_regression):
            raise ValueError(f"epoch {epoch}: regression score support differs from baseline")
        manual_review = [key for key, (_, gold_count) in baseline_regression.items()
                         if gold_count < MIN_REGRESSION_SUPPORT]
        losses = [key for key, (value, gold_count) in regression.items()
                  if gold_count >= MIN_REGRESSION_SUPPORT and
                  value < baseline_regression[key][0] - REGRESSION_TOLERANCE]
        ge_found_gain = scores["GE.address_found_recall"] - baseline["GE.address_found_recall"]
        ge_exact_gain = scores["GE.address_exact_f1"] - baseline["GE.address_exact_f1"]
        reasons = [f"regression over two points: {key} "
                   f"({baseline_regression[key][0]:.4f} -> {regression[key][0]:.4f})"
                   for key in losses]
        reasons.extend(f"manual review required: {key} has "
                       f"{baseline_regression[key][1]} gold spans" for key in manual_review)
        if ge_found_gain < 0.05:
            reasons.append("GE address found recall gain below five points")
        if ge_exact_gain < 0.01:
            reasons.append("GE address exact F1 gain below one point")
        gaps = [max(0.0, target - scores[f"{country}.{metric}"])
                for country in COUNTRIES for metric, target in TARGETS.items()]
        maximum, total = max(gaps), sum(gaps)
        decisions.append({"epoch": epoch, "eligible": not reasons, "reasons": reasons,
                          "manual_review_required": bool(manual_review),
                          "maximum_shortfall": maximum, "total_shortfall": total,
                          "report_sha256": digest(entry["report"])})
        if not reasons:
            ranked.append((maximum, total, epoch))
    winner = min(ranked)[2] if ranked else None
    result = {"selected_epoch": winner, "detector_lineage_verified": True,
            "release_eligible": False,
            "baseline_report_sha256": digest(baseline_report),
            "gold_sha256": gold_hash, "evaluator_sha256": evaluator_hash,
            "epochs": decisions}
    if v17_scores is not None:
        result.update({"development_set": development_set,
                       "development_manifest_sha256": dev_v2_lock.MANIFEST_SHA256,
                       "v17_reference_predictions_sha256": dev_v2_lock.PREDICTIONS_SHA256})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--baseline-bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--development-set", choices=("dev-v1", "dev-v2"), default="dev-v1")
    args = parser.parse_args()
    result = select(args.baseline_report, args.baseline_bundle, args.manifest,
                    args.run, args.gold, args.evaluator, args.development_set)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["selected_epoch"] is None:
        parser.exit(1, "No epoch passed the frozen real-data guards.\n")


if __name__ == "__main__":
    main()
