"""Verify the corrected development gold and its paired V17 reference."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path("data/interim/review/dev-v2-ge-shell-exclusion-20260927-v1")
EXPECTED = {
    "data/interim/review/gold-all.jsonl": "3940db21008a51dc71e78fdb390b1be3b1f69ca4596b4c49b50da0af77b2abbd",
    "data/interim/review/gold-all-with-hosts.jsonl": "e283b6db04469fc67945f0204a597fe25da092fa2f91062073325f1213028887",
    "internal/reports/m7/older-dev-publisher-index-v1.jsonl": "bb42b111576b22f156e1c0a076e14f8a41c6b832062784da924c82ba414cdb2e",
    "internal/reports/m7/paragraph-ablation-auto-baseline.jsonl": "53374bf06a46d555ed57ef5b3a2d1d0248a315dd12515fd87a4169437ee37f9a",
    "runs/eval-v17-auto/eval/review.json": "8935384e4a9188ddfc112972fe3890168ff270544b019c9f2d091f2e5ee23c02",
    "internal/reports/m7/build-dev-v2-ge-shell-exclusion-20260927.py": "b2b52751df6d8ab24804dd2ab1ef992f4130711b274ba2f99ead7bde94067bf8",
    str(DIRECTORY / "gold-all.jsonl"): "4d2c5ce2834c1183dc2868184da1c45ce7c5be8385a993b7df6884898ee7d395",
    str(DIRECTORY / "gold-all-with-hosts.jsonl"): "f9ecfc77eaaf0e334757396c5a7be881edd552a281039cffcffc6744dc08f02e",
    str(DIRECTORY / "source-index.jsonl"): "0e101c4a1498741d437cfd9823489d2757d3d89179018dafc4744e93c6b9989f",
    str(DIRECTORY / "v17-auto-predictions.jsonl"): "0ae22c2daa8220fbb165cab0b4259d7050bdfbea0f7573770141b1162f13aac0",
    str(DIRECTORY / "v17-exact-scores.json"): "b7b00a4db9a69be52018181abff159573ad24e1c1c063cb1a59b1577cd423c07",
    str(DIRECTORY / "removal-manifest.json"): "ae2bc138bf6a6d93e5f4c12fa142782a05bcaef0f7ece5842aaa4c22ed5e7cf2",
}
EXCLUDED = frozenset(
    f"ge-economy-www.economy.ge_page_news_nw_{number}" for number in (3238, 3239, 3240)
)
GOLD_SHA256 = EXPECTED[str(DIRECTORY / "gold-all.jsonl")]
MANIFEST_SHA256 = EXPECTED[str(DIRECTORY / "removal-manifest.json")]
PREDICTIONS_SHA256 = EXPECTED[str(DIRECTORY / "v17-auto-predictions.jsonl")]
NATIVE_V17_REPORT_SHA256 = "6930f8409940a952848c9a9ed5202a7cbe8346a87f7aa8da0d8e5eb0add3d185"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def records(data):
    return [(line, json.loads(line)) for line in data.splitlines(keepends=True) if line.strip()]


def verify(gold_path, root=ROOT):
    """Return pinned V17 scores only when the complete dev-v2 lineage matches."""
    root = Path(root)
    expected_gold = root / DIRECTORY / "gold-all.jsonl"
    if Path(gold_path).resolve() != expected_gold.resolve():
        raise ValueError("dev-v2 requires the pinned corrected gold path")
    data = {}
    for name, expected_hash in EXPECTED.items():
        data[name] = (root / name).read_bytes()
        if digest(data[name]) != expected_hash:
            raise ValueError(f"dev-v2 pinned file changed: {name}")

    manifest = json.loads(data[str(DIRECTORY / "removal-manifest.json")])
    if (manifest.get("retained_rows") != 874
            or {row["id"] for row in manifest.get("exclusions", [])} != EXCLUDED
            or len(manifest["exclusions"]) != 3):
        raise ValueError("dev-v2 exclusion manifest differs")
    original_gold = records(data["data/interim/review/gold-all.jsonl"])
    original_hosts = records(data["data/interim/review/gold-all-with-hosts.jsonl"])
    original_predictions = records(data["internal/reports/m7/paragraph-ablation-auto-baseline.jsonl"])
    original_index = records(data["internal/reports/m7/older-dev-publisher-index-v1.jsonl"])
    if not all(len(rows) == 877 for rows in
               (original_gold, original_hosts, original_predictions, original_index)):
        raise ValueError("dev-v1 input count differs")
    gold_names = [row["name"] for _, row in original_gold]
    if len(set(gold_names)) != 877 or not EXCLUDED <= set(gold_names):
        raise ValueError("dev-v1 gold identity differs")
    for rows in (original_hosts, original_predictions):
        if [row["name"] for _, row in rows] != gold_names:
            raise ValueError("dev-v1 paired row order differs")
    index_by_id = {row["id"]: line for line, row in original_index}
    if len(index_by_id) != 877 or set(index_by_id) != set(gold_names):
        raise ValueError("dev-v1 source index identity differs")
    for source_name, destination_name in (
        ("data/interim/review/gold-all.jsonl", "gold-all.jsonl"),
        ("data/interim/review/gold-all-with-hosts.jsonl", "gold-all-with-hosts.jsonl"),
        ("internal/reports/m7/paragraph-ablation-auto-baseline.jsonl", "v17-auto-predictions.jsonl"),
    ):
        filtered = b"".join(line for line, row in records(data[source_name])
                            if row["name"] not in EXCLUDED)
        if filtered != data[str(DIRECTORY / destination_name)]:
            raise ValueError(f"dev-v2 retained rows differ: {destination_name}")
    filtered_index = b"".join(index_by_id[name] for name in gold_names if name not in EXCLUDED)
    if filtered_index != data[str(DIRECTORY / "source-index.jsonl")]:
        raise ValueError("dev-v2 source index differs")

    gold_rows = records(data[str(DIRECTORY / "gold-all.jsonl")])
    prediction_rows = records(data[str(DIRECTORY / "v17-auto-predictions.jsonl")])
    source_rows = records(data[str(DIRECTORY / "source-index.jsonl")])
    if len(gold_rows) != 874 or len(prediction_rows) != 874 or len(source_rows) != 874:
        raise ValueError("dev-v2 retained row count differs")
    for (_, gold), (_, prediction), (_, source) in zip(gold_rows, prediction_rows, source_rows):
        if (gold["name"] != prediction["name"] or gold["country"] != prediction["country"]
                or gold["expected"] != prediction["gold"] or source["id"] != gold["name"]
                or source["country"] != gold["country"]):
            raise ValueError(f"dev-v2 V17 prediction/gold mismatch: {gold['name']}")
        if digest(gold["input"].encode()) != source["gold_text_sha256"]:
            raise ValueError(f"dev-v2 source text hash differs: {gold['name']}")
        raw_path = (root / source["raw_path"]).resolve()
        if not raw_path.is_relative_to((root / "data/raw/review").resolve()):
            raise ValueError(f"dev-v2 raw path outside review corpus: {gold['name']}")
        try:
            raw_bytes = raw_path.read_bytes()
        except OSError as error:
            raise ValueError(f"dev-v2 raw source unavailable: {gold['name']}") from error
        if digest(raw_bytes) != source["raw_file_sha256"]:
            raise ValueError(f"dev-v2 raw source hash differs: {gold['name']}")
        raw = json.loads(raw_bytes)
        if (raw["id"] != gold["name"] or raw["url"] != source["raw_url"]
                or raw["text"] != gold["input"] or raw["country"] != gold["country"]
                or raw["source"] != source["source"]):
            raise ValueError(f"dev-v2 raw source linkage differs: {gold['name']}")
    scores = json.loads(data[str(DIRECTORY / "v17-exact-scores.json")])
    if scores["v1"]["rows"] != 877 or scores["v2"]["rows"] != 874:
        raise ValueError("dev-v2 V17 baseline count differs")
    return scores


def verify_native_v17(report, scores):
    """Check that a fresh native V17 report matches the paired saved predictions."""
    systems = [system for system in report.get("systems", [])
               if system.get("system") == "tessera"]
    if len(systems) != 1 or systems[0].get("cases") != 874:
        raise ValueError("dev-v2 native V17 report must contain 874 Tessera cases")

    def matches(actual, expected, label):
        for field, expected_field in (("gold", "gold"), ("predicted", "predicted"),
                                      ("exact_tp", "correct")):
            if actual[field] != expected[expected_field]:
                raise ValueError(f"dev-v2 native V17 {label} {field} differs")
        if abs(actual["exact"]["f1"] - expected["exact_f1"]) > 0.000051:
            raise ValueError(f"dev-v2 native V17 {label} exact F1 differs")

    matches(systems[0]["overall"], scores["v2"]["all"]["micro"], "overall")
    for country in ("US", "GB", "DE", "GE", "JP"):
        actual = systems[0]["by_slice"]["country"][country]
        expected = scores["v2_by_country"][country]
        for kind in ("person", "org", "address", "email", "phone"):
            matches(actual[kind], expected[kind], f"{country}/{kind}")
