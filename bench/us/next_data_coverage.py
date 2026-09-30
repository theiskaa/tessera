"""Measure distinct real examples and source documents in the proposed US data."""

import argparse
import collections
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from check_us_next_data_split import (
    ADDITIONS, REPLACEMENTS, proposed_sources, rows, verify_source_manifest,
)
from source_identity import document_key


LONG_WORDS = 200
KINDS = ("person", "org", "address")
ADDRESS_PREFIX = re.compile(
    r"\b(?:room|suite|building|floor|mail\s+(?:code|stop))\b|\bm\s*/\s*s\b|\bms\s*:"
    r"|\bstop\s+(?=[a-z0-9-]*\d)[a-z0-9-]+\b", re.I)
TARGETS = Path(__file__).with_name("fixtures") / "us-next-data-targets-v1.json"


def source_host(row):
    """Describe the source website without treating it as a publisher identity."""
    key = document_key(row)
    if key and key.startswith("fr:"):
        return "federalregister.gov"
    url = row.get("source_url")
    if not url and key and key.startswith("url:"):
        url = key[4:]
    host = (urlsplit(url or "").hostname or "").casefold()
    return (host[4:] if host.startswith("www.") else host) or "unknown"


def summarize(group):
    """Count repeated mentions separately from distinct supervised documents."""
    keys = [document_key(row) for row in group]
    if any(key is None for key in keys):
        raise ValueError("a proposed example has no source document identity")
    long_rows = [row for row in group if len(row["text"].split()) >= LONG_WORDS]
    result = {
        "examples": len(group),
        "source_documents": len(set(keys)),
        "long_examples": len(long_rows),
        "long_source_documents": len({document_key(row) for row in long_rows}),
        "spans": dict(sorted(collections.Counter(
            span["kind"] for row in group for span in row["entities"]).items())),
        "by_kind": {},
    }
    for kind in KINDS:
        positives = [row for row in group
                     if any(span["kind"] == kind for span in row["entities"])]
        long_positives = [row for row in long_rows
                          if any(span["kind"] == kind for span in row["entities"])]
        result["by_kind"][kind] = {
            "examples": len(positives),
            "source_documents": len({document_key(row) for row in positives}),
            "long_examples": len(long_positives),
            "long_source_documents": len({document_key(row) for row in long_positives}),
            "long_source_websites": sorted({source_host(row) for row in long_positives}),
        }
    prefixed = []
    for row in long_rows:
        encoded = row["text"].encode()
        if any(span["kind"] == "address" and ADDRESS_PREFIX.search(
                encoded[span["start"]:span["end"]].decode())
               for span in row["entities"]):
            prefixed.append(row)
    result["long_prefixed_address_source_documents"] = len({
        document_key(row) for row in prefixed})
    result["long_mixed_source_documents"] = len({
        document_key(row) for row in long_rows
        if set(KINDS) <= {span["kind"] for span in row["entities"]}})
    return result


def inventory():
    """Count only pinned proposed inputs, preserving their configured order."""
    sources = []
    new_sources = set(ADDITIONS) | set(REPLACEMENTS.values())
    for path in proposed_sources():
        verify_source_manifest(path, require_eligible=path.stem in new_sources)
        sources.append((path, rows(path)))
    return sources


def coverage_checks(summary, targets):
    """Measure each fixed collection target without averaging overlapping quotas."""
    if targets["long_minimum_words"] != LONG_WORDS:
        raise ValueError("coverage word threshold differs from the fixed target policy")
    measurements = [("distinct long documents", summary["long_source_documents"],
                     targets["long_source_documents"])]
    measurements.extend(
        (f"long documents with {kind}", summary["by_kind"][kind]["long_source_documents"],
         targets["long_positive_source_documents"][kind]) for kind in KINDS)
    websites = set(summary["by_kind"]["address"]["long_source_websites"])
    websites -= {"www.federalregister.gov", "federalregister.gov", "unknown"}
    measurements.extend([
        ("long addresses with extra parts", summary["long_prefixed_address_source_documents"],
         targets["long_prefixed_address_source_documents"]),
        ("other websites with long addresses", len(websites),
         targets["long_address_non_federal_register_websites"]),
    ])
    checks = []
    for name, current, required in measurements:
        if not isinstance(required, int) or isinstance(required, bool) or required <= 0:
            raise ValueError(f"invalid coverage target: {name}")
        checks.append({"name": name, "current": current, "required": required,
                       "missing": max(0, required - current), "passed": current >= required,
                       "progress_percent": round(100 * min(current, required) / required, 1)})
    return checks


def main():
    """Print current coverage without writing reports or starting training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print structured coverage")
    parser.add_argument("--by-file", action="store_true", help="include individual file counts")
    args = parser.parse_args()
    sources = inventory()
    all_rows = [row for _, group in sources for row in group]
    result = summarize(all_rows)
    result["long_minimum_words"] = LONG_WORDS
    if args.by_file:
        result["by_file"] = {path.stem: summarize(group) for path, group in sources}
    source_counts = collections.Counter(document_key(row) for row in all_rows)
    result["largest_source_documents"] = dict(source_counts.most_common(10))
    targets = json.loads(TARGETS.read_text())
    result["target_version"] = targets["version"]
    result["coverage_checks"] = coverage_checks(result, targets)
    result["coverage_passed"] = all(check["passed"] for check in result["coverage_checks"])
    result["scope"] = "collection coverage only; integrity, labels, split, encoding and sampling are separate gates"
    if args.json or args.by_file:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{result['examples']} examples from {result['source_documents']} source documents")
        print(f"Fixed collection targets v{targets['version']}; long means >= {LONG_WORDS} words")
        for check in result["coverage_checks"]:
            print(f"{check['name']}: {check['current']}/{check['required']} "
                  f"({check['progress_percent']}%), missing {check['missing']}")
        print(f"Long documents containing all three learned kinds: {result['long_mixed_source_documents']}")
        print(result["scope"])
    if not result["coverage_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
