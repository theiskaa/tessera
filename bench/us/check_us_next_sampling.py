"""Audit proposed US sampling from supplied trainer counts without starting training."""

import argparse
import ast
import collections
import json
import re
from pathlib import Path

from check_us_next_data_split import (
    ADDITIONS, REPLACEMENTS, ROOT, digest, proposed_sources, rows, verify_source_manifest,
)
from next_data_coverage import TARGETS
from source_identity import document_key


def positive(value, name):
    """Reject booleans, fractions and nonpositive sampling counts."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def config_values(text):
    """Read only the known sampling fields from their TOML sections."""
    result = {}
    for section, fields in {
        "train": {"epochs": None, "batch_size": None},
        "detector": {"silver": None, "silver_repeats": [], "silver_repeat": 1,
                     "synthetic_per_epoch": None},
    }.items():
        blocks = re.findall(r"(?ms)^\[" + section + r"\]\s*(.*?)"
                            r"(?=^\[|\Z)", text)
        if len(blocks) != 1:
            raise ValueError(f"expected one [{section}] section")
        for name, default in fields.items():
            matches = re.findall(r"(?m)^\s*" + name + r"\s*=\s*"
                                 r"(\[[^\]]*\]|[^\n]+)", blocks[0])
            if len(matches) > 1 or not matches and default is None:
                raise ValueError(f"missing or repeated {section}.{name}")
            try:
                result[name] = ast.literal_eval(matches[0]) if matches else default
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"unsupported {section}.{name} value") from error
    for name in ("epochs", "batch_size", "silver_repeat", "synthetic_per_epoch"):
        positive(result[name], name)
    paths, repeats = result["silver"], result["silver_repeats"]
    if (not isinstance(paths, list) or not paths or
            any(not isinstance(path, str) for path in paths) or
            len(set(paths)) != len(paths)):
        raise ValueError("silver must contain distinct source paths")
    if not isinstance(repeats, list):
        raise ValueError("silver_repeats must be a list")
    repeats = repeats or [result["silver_repeat"]] * len(paths)
    if len(repeats) != len(paths):
        raise ValueError("silver_repeats must match the source paths")
    for repeat in repeats:
        positive(repeat, "silver repeat")
    result["silver_repeats"] = repeats
    return result


def summarize(config, counts, sources, targets):
    """Require one encoded piece per row before attributing document exposure."""
    if config["silver"] != [name for name, _ in sources]:
        raise ValueError("config silver paths differ from proposed_sources()")
    for name in ("duplicate_documents", "unreachable_spans"):
        if type(counts.get(name)) is not int or counts[name] != 0:
            raise ValueError(f"{name} must be zero")
    lengths = [len(group) for _, group in sources]
    pieces, tokens = counts.get("source_pieces"), counts.get("source_content_tokens")
    for name, values in (("source_pieces", pieces), ("source_content_tokens", tokens)):
        if not isinstance(values, list) or len(values) != len(sources):
            raise ValueError(f"{name} must match the proposed source list")
        for value in values:
            positive(value, name)
    if pieces != lengths:
        raise ValueError("source_pieces differs from input rows; split/deduplicated attribution unsupported")
    for name in ("documents", "pieces", "content_tokens"):
        positive(counts.get(name), name)
    if (counts["documents"] != sum(lengths) or counts["pieces"] != sum(pieces) or
            counts["content_tokens"] != sum(tokens) or
            any(token_count < piece_count for token_count, piece_count in zip(tokens, pieces))):
        raise ValueError("inconsistent document, piece or content-token totals")
    countries = counts.get("by_country", {})
    if (set(countries) != {"US"} or
            countries["US"].get("documents") != counts["documents"] or
            countries["US"].get("pieces") != counts["pieces"]):
        raise ValueError("country document/piece counts disagree with US totals")
    epochs = positive(config["epochs"], "epochs")
    batch = positive(config["batch_size"], "batch_size")
    synthetic = positive(config["synthetic_per_epoch"], "synthetic_per_epoch")
    repeats = config["silver_repeats"]
    if len(repeats) != len(sources):
        raise ValueError("silver repeats differ from source count")
    documents = collections.Counter()
    seen_text = set()
    by_file = {}
    long_draws = real_draws = token_draws = 0
    long_words = positive(targets["long_minimum_words"], "long_minimum_words")
    for (name, group), repeat, content_tokens in zip(sources, repeats, tokens):
        positive(repeat, "silver repeat")
        for row in group:
            key = document_key(row)
            if key is None or row.get("country") != "US" or not row["text"].strip():
                raise ValueError(f"invalid US source row in {name}")
            if row["text"] in seen_text:
                raise ValueError("duplicate source text despite zero diagnostic duplicates")
            seen_text.add(row["text"])
            documents[key] += repeat
        long_count = sum(len(row["text"].split()) >= long_words for row in group)
        real_draws += len(group) * repeat
        long_draws += long_count * repeat
        token_draws += content_tokens * repeat
        by_file[name] = {"pieces": len(group), "long_pieces": long_count,
                         "repeats_per_epoch": repeat, "repeats_all_epochs": repeat * epochs,
                         "draws_per_epoch": len(group) * repeat,
                         "content_token_draws_per_epoch": content_tokens * repeat}
    total = synthetic + real_draws
    largest_key, largest_draws = documents.most_common(1)[0]
    fractions = {"synthetic": synthetic / total, "real": real_draws / total,
                 "long_real": long_draws / total,
                 "maximum_source_document": largest_draws / total}
    policy = targets["sampling"]
    checks = [
        ("maximum_synthetic_fraction", fractions["synthetic"], "maximum"),
        ("minimum_long_real_fraction_of_all_draws", fractions["long_real"], "minimum"),
        ("maximum_long_real_fraction_of_all_draws", fractions["long_real"], "maximum"),
        ("maximum_single_source_document_fraction", fractions["maximum_source_document"], "maximum"),
    ]
    checks = [{"name": name, "current": current, "required": policy[name],
               "passed": current <= policy[name] if bound == "maximum" else current >= policy[name]}
              for name, current, bound in checks]
    return {
        "scope": "calculations from supplied check-silver diagnostics; not provenance verification "
                 "that the trainer counts came from this config; synthetic shard size is not verified",
        "epochs": epochs, "batch_size": batch, "long_minimum_words": long_words,
        "draws_per_epoch": {"total": total, "synthetic": synthetic, "real": real_draws,
                            "long_real": long_draws},
        "draws_all_epochs": {"total": total * epochs, "synthetic": synthetic * epochs,
                             "real": real_draws * epochs, "long_real": long_draws * epochs},
        "real_content_token_draws_per_epoch": token_draws,
        "real_content_token_draws_all_epochs": token_draws * epochs,
        "optimizer_updates": ((total + batch - 1) // batch) * epochs,
        "schedule_scope": "configured epochs; early stopping may reduce actual exposure",
        "fractions_of_all_draws": fractions, "source_documents": len(documents),
        "largest_source_document": {"key": largest_key, "draws_per_epoch": largest_draws,
                                    "draws_all_epochs": largest_draws * epochs,
                                    "fraction_of_real_draws": largest_draws / real_draws},
        "by_file": by_file, "sampling_checks": checks,
        "sampling_passed": all(check["passed"] for check in checks),
    }


def main():
    """Verify source manifests and print the sampling audit as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--silver-counts", required=True, type=Path)
    args = parser.parse_args()
    config = config_values(args.config.read_text())
    paths = proposed_sources()
    if config["silver"] != [str(path.relative_to(ROOT)) for path in paths]:
        raise ValueError("config silver paths differ from proposed_sources()")
    new_names = set(ADDITIONS) | set(REPLACEMENTS.values())
    for path in paths:
        verify_source_manifest(path, require_eligible=path.stem in new_names)
    sources = [(str(path.relative_to(ROOT)), rows(path)) for path in paths]
    result = summarize(config, json.loads(args.silver_counts.read_text()), sources,
                       json.loads(TARGETS.read_text()))
    inputs = [args.config, args.silver_counts, TARGETS, Path(__file__), *paths,
              *(path.with_suffix(".manifest.json") for path in paths)]
    result["input_sha256"] = {str(path.resolve()): digest(path) for path in inputs}
    result["source_manifests_verified"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["sampling_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
