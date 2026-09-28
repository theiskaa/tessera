"""Check that frozen US evaluation and detector inputs still agree."""

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
SILVER = ROOT / "data/interim/silver"
PROCESSED = ROOT / "data/processed/detector-us-v1"
PARSER = ROOT / "data/processed/parser-us-v1"
KIND = {"person", "org", "address"}
SOURCE_ARTIFACT = re.compile(r'">|</?[A-Za-z][^>]*>|&(?:amp|quot|nbsp|lt|gt);')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def person_key(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    if "," in value:
        surname, given = value.split(",", 1)
        given_parts = re.findall(r"[a-z]+", given)
        surname_parts = re.findall(r"[a-z]+", surname)
        return (given_parts[0], surname_parts[-1]) if given_parts and surname_parts else None
    parts = re.findall(r"[a-z]+", value)
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        parts.pop()
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


def spans(row):
    text = row.get("input", row.get("text")).encode()
    result = []
    for span in row.get("expected", row.get("entities")):
        value = text[span["start"]:span["end"]].decode()
        require(span["start"] < span["end"] and value, "empty or invalid label")
        if "text" in span:
            require(value == span["text"], "label differs from source text")
        result.append((span["kind"], span["start"], span["end"], value))
    for left, right in zip(sorted(result, key=lambda item: item[1]),
                           sorted(result, key=lambda item: item[1])[1:]):
        require(left[2] <= right[1], "overlapping labels")
    return result


def main():
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("check_ready needs pyarrow to read the generated Parquet data") from error

    exclusion_path = REVIEW / "us-eval-exclusions-v1.jsonl"
    exclusion_hash = digest(exclusion_path.read_bytes())
    exclusions = load(exclusion_path)
    names = set()
    texts = set()
    surfaces = collections.defaultdict(set)
    people = set()
    for row in exclusions:
        require(row["country"] == "US", "non-US evaluation case")
        require(row["name"] not in names, "duplicate evaluation name")
        require(digest(row["input"].encode()) not in texts, "duplicate evaluation text")
        names.add(row["name"])
        texts.add(digest(row["input"].encode()))
        for kind, _, _, value in spans(row):
            if kind in KIND:
                surfaces[kind].add(normalized(value))
                if kind == "person" and person_key(value):
                    people.add(person_key(value))

    staff_manifest = json.loads((REVIEW / "us-staff-challenge-v1.manifest.json").read_text())
    staff_data = (REVIEW / "us-staff-challenge-v1.jsonl").read_bytes()
    require(digest(staff_data) == staff_manifest["sha256"], "staff challenge changed")
    base_data = (REVIEW / "us-dev-office-exclusions-v1.jsonl").read_bytes()
    require(exclusion_path.read_bytes() == base_data + staff_data,
            "evaluation exclusions do not match the frozen evaluation sets")
    require(staff_manifest["existing_evaluation_sha256"] == digest(base_data),
            "staff challenge base changed")
    for slug, expected in staff_manifest["sources_sha256"].items():
        require(digest((ROOT / f"data/raw/us-release-eval/{slug}.html").read_bytes()) == expected,
                f"staff source changed: {slug}")

    parser_sample = json.loads((PARSER / "sample.json").read_text())
    parser_checks = parser_sample["checks"]
    require(parser_sample["filters"]["countries"] == ["US"] and parser_checks["passed"],
            "US parser sample audit did not pass")
    require(all(value == 0 for key, value in parser_checks["counts"].items()
                if key != "postcode_and_number_shared"),
            "US parser sample has an unresolved split or label error")
    for split, expected in (("train", 62549), ("valid", 3000), ("test", 3000)):
        require(parquet.ParquetFile(PARSER / f"{split}.parquet").metadata.num_rows == expected,
                f"wrong US parser {split} size")

    seen = set()
    real_counts = collections.Counter()
    for filename, key in (("us-reviewed-strict-v1", "evaluation_gold_sha256"),
                          ("us-house-staff-v1", "evaluation_sha256")):
        path = SILVER / f"{filename}.jsonl"
        manifest = json.loads((SILVER / f"{filename}.manifest.json").read_text())
        require(manifest["sha256"] == digest(path.read_bytes()), f"{filename} changed")
        require(manifest[key] == exclusion_hash, f"{filename} uses stale evaluation exclusions")
        if filename == "us-house-staff-v1":
            require(manifest["source_sha256"] == digest(
                (ROOT / "data/raw/us-staff/house-phonebook-2023.pdf").read_bytes()),
                "House directory source changed")
        rows = load(path)
        require(len(rows) == manifest["documents"], f"{filename} document count changed")
        file_counts = collections.Counter()
        for row in rows:
            require(row["country"] == "US", "non-US training document")
            text_hash = digest(row["text"].encode())
            require(text_hash not in seen and text_hash not in texts,
                    "duplicate or evaluation training text")
            require(not SOURCE_ARTIFACT.search(row["text"]), "HTML artifact in real training text")
            seen.add(text_hash)
            for kind, _, _, value in spans(row):
                if kind not in KIND:
                    continue
                require(normalized(value) not in surfaces[kind],
                        f"evaluation entity in real training: {value}")
                if filename == "us-house-staff-v1" and kind == "person":
                    require(person_key(value) not in people,
                            f"evaluation person alias in House training: {value}")
                real_counts[kind] += 1
                file_counts[kind] += 1
        if filename == "us-reviewed-strict-v1":
            require(dict(file_counts) == manifest["labels"], "strict silver label count changed")
        else:
            require(file_counts["person"] == manifest["counts"]["person_labels"],
                    "House label count changed")

    generator = json.loads((ROOT / "data/manifests/detector-synthetic.json").read_text())
    require(generator["exclude_gold_sha256"] == exclusion_hash,
            "synthetic data uses stale evaluation exclusions")
    roster = ROOT / "trainer/data/us-public-bodies.json"
    require(generator["public_bodies_sha256"] == digest(roster.read_bytes()),
            "synthetic data uses stale public bodies")
    synthetic_counts = collections.Counter()
    for split, expected in (("train", 170000), ("valid", 10000), ("test", 20000)):
        file = parquet.ParquetFile(PROCESSED / f"{split}.parquet")
        require(file.metadata.num_rows == expected, f"wrong {split} size")
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["country", "text", "entities_json"]):
            for row in batch.to_pylist():
                require(row["country"] == "US", "non-US synthetic document")
                labels = json.loads(row["entities_json"])
                for kind, _, _, value in spans({"text": row["text"], "entities": labels}):
                    if kind in KIND:
                        require(normalized(value) not in surfaces[kind],
                                f"evaluation entity in synthetic {split}: {value}")
                        synthetic_counts[kind] += 1
    print(f"US data gate passed: {len(exclusions)} held cases, {len(seen)} real training documents, "
          f"{sum(synthetic_counts.values())} synthetic model labels; "
          f"real labels {dict(real_counts)}")


if __name__ == "__main__":
    main()
