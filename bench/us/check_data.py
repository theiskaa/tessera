"""Verify configured US detector data, provenance hashes and evaluation separation."""

import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

from holdout import collisions, contact_collisions, synthetic_sources
from source_identity import document_key, source_url_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/detector-shared-v5.toml"


def digest(path):
    """Hash exact file bytes, including locally retained recipe dependencies."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_field(text, section, name):
    """Read a required string or list from a known TOML section on Python 3.9."""
    blocks = re.findall(r"(?ms)^\[" + re.escape(section) + r"\][ \t]*(?:#[^\n]*)?\n"
                        r"(.*?)(?=^\[|\Z)", text)
    if len(blocks) != 1:
        raise ValueError(f"expected one [{section}] section")
    values = re.findall(r"(?m)^[ \t]*" + re.escape(name) + r"[ \t]*=[ \t]*"
                        r"(\[[^\]]*\]|[^\n]+)", blocks[0])
    if len(values) != 1:
        raise ValueError(f"expected one {section}.{name}")
    try:
        return ast.literal_eval(values[0])
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"unsupported {section}.{name}") from error


def read_config(path):
    """Keep the silver list in configured order without importing recipe membership."""
    text = path.read_text()
    result = {name: config_field(text, section, name) for section, name in (
        ("detector", "silver"), ("generate", "exclude_gold"),
        ("data", "processed"), ("data", "manifests"))}
    silver = result["silver"]
    if (not isinstance(silver, list) or not silver or
            any(not isinstance(name, str) or not name for name in silver)):
        raise ValueError("silver must be a nonempty list of file paths")
    if len({(ROOT / name).resolve() for name in silver}) != len(silver):
        raise ValueError("silver contains duplicate file paths")
    for name in ("exclude_gold", "processed", "manifests"):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"{name} must be a nonempty path")
    return result


def verified_rows(path, *, evaluation=False):
    """Verify a dataset manifest and every dependency with an explicit path."""
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest.get("sha256") != digest(path):
        raise ValueError(f"dataset hash changed: {path}")
    eligible = manifest.get("training_eligible")
    if evaluation and eligible is not False or not evaluation and eligible is False:
        raise ValueError(f"wrong training eligibility: {path}")
    dependencies = manifest.get("input_sha256", {})
    if not isinstance(dependencies, dict):
        raise ValueError(f"invalid input_sha256 mapping: {path}")
    dependencies = dict(dependencies)
    source_hashes = manifest.get("source_sha256", {})
    if isinstance(source_hashes, dict):
        for relative, expected in source_hashes.items():
            if relative in dependencies and dependencies[relative] != expected:
                raise ValueError(f"conflicting dependency hash: {relative}")
            dependencies[relative] = expected
    for relative, expected in dependencies.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"dataset dependency changed: {relative}")
    group = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for field in ("cases", "documents"):
        if field in manifest and manifest[field] != len(group):
            raise ValueError(f"dataset {field} differs from rows: {path}")
    if not group or any(row.get("country") != "US" for row in group):
        raise ValueError(f"empty or non-US dataset: {path}")
    return group


def shingles(text):
    """Return normalized twelve-word windows, including across line breaks."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[index:index + 12]) for index in range(len(words) - 11)}


def prose_overlaps(real, gold):
    """Report shared prose from every real row, including historical inputs."""
    index = collections.defaultdict(set)
    for row in gold:
        for fragment in shingles(row["input"]):
            index[fragment].add(row["name"])
    hits = {}
    for origin, text, _ in real:
        matches = {name for fragment in shingles(text) for name in index.get(fragment, ())}
        if matches:
            hits[origin] = sorted(matches)
    return hits


def overlap_checks(gold, sources, *, allow_relay=False):
    """Keep the existing learned aliases and substantive contact policy."""
    learned = collisions(gold, sources(), strict_aliases=True)
    contacts = contact_collisions(gold, sources())
    contact_hits = {}
    for name, kinds in contacts.items():
        kept = {kind: sorted(matches) for kind, matches in kinds.items()
                if not (allow_relay and kind == "phone" and
                        all(re.sub(r"\D", "", value) == "711" for _, value in matches))}
        if kept:
            contact_hits[name] = kept
    return {"learned": {name: {kind: sorted(matches) for kind, matches in kinds.items()}
                        for name, kinds in learned.items()}, "contacts": contact_hits}


def verify_synthetic(config):
    """Require exact configured silver/exclusions and all three generated shard hashes."""
    manifest = json.loads((ROOT / config["manifests"] / "detector-synthetic.json").read_text())
    silver = {name: digest(ROOT / name) for name in config["silver"]}
    if (manifest.get("source") != "tessera-generator" or
            manifest.get("real_silver_sha256") != silver or
            manifest.get("exclude_gold") != config["exclude_gold"] or
            manifest.get("exclude_gold_sha256") != digest(ROOT / config["exclude_gold"])):
        raise ValueError("generator manifest differs from configured silver or exclusions")
    shards = manifest.get("synthetic_parquet_sha256", {})
    if set(shards) != {"train", "valid", "test"}:
        raise ValueError("generator manifest must contain exactly three shard hashes")
    for split, expected in shards.items():
        if digest(ROOT / config["processed"] / f"{split}.parquet") != expected:
            raise ValueError(f"synthetic {split} shard changed")


def audit(config, *, strict_prose=False):
    """Check frozen inputs and return every separation failure without changing data."""
    gold = verified_rows(ROOT / config["exclude_gold"], evaluation=True)
    if len({row["name"] for row in gold}) != len(gold):
        raise ValueError("duplicate evaluation case names")
    real = []
    source_hits = []
    eval_keys = {document_key(row) for row in gold} - {None}
    eval_urls = {source_url_key(row) for row in gold} - {None}
    counts = {}
    for name in config["silver"]:
        group = verified_rows(ROOT / name)
        counts[name] = len(group)
        for row in group:
            origin = f"{name}:{row['id']}"
            key = document_key(row)
            if key is None or key in eval_keys or source_url_key(row) in eval_urls:
                source_hits.append(origin)
            real.append((origin, row["text"], row["entities"]))
    verify_synthetic(config)
    real_hits = overlap_checks(gold, lambda: iter(real), allow_relay=True)
    synthetic_hits = overlap_checks(
        gold, lambda: synthetic_sources(ROOT / config["processed"]))
    prose = prose_overlaps(real, gold)
    failures = {"source_or_url": source_hits,
                "real_prose": prose if strict_prose else {},
                "real_learned": real_hits["learned"], "real_contacts": real_hits["contacts"],
                "synthetic_learned": synthetic_hits["learned"],
                "synthetic_contacts": synthetic_hits["contacts"]}
    return {"passed": not any(failures.values()), "real_rows": len(real),
            "evaluation_cases": len(gold), "by_source": counts, "failures": failures,
            "prose_check": {"strict": strict_prose, "overlaps": prose,
                            "scope": "all real rows; informational unless --strict-prose; "
                                     "does not replace a reviewed subset's prose gate"},
            "scope": "all real rows and synthetic train/valid overlap checks; "
                     "all three synthetic shard hashes verified; shared real 711 accepted"}


def main():
    """Print a read-only audit for the supplied config, defaulting to current V5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--strict-prose", action="store_true",
                        help="also fail on any twelve-word overlap in all real rows")
    args = parser.parse_args()
    result = audit(read_config(args.config), strict_prose=args.strict_prose)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
