"""Check proposed US real data and synthetic labels against reserved evaluation."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from active_sources import V4_SILVER
from check_holdout_overlap import collisions, contact_collisions, synthetic_sources
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key, source_url_key


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/detector-shared-v4.toml"
REPLACEMENTS = {
    "us-federal-org-reviewed-safe-v1": "us-federal-org-reviewed-safe-v2",
    "us-reviewed-org-paragraphs-v4": "us-reviewed-org-paragraphs-strict-v1",
    "us-v4-reviewed-additions-v2": "us-v4-reviewed-additions-strict-v2",
    "us-v4-reviewed-historical-acronyms-v1":
        "us-v4-reviewed-historical-acronyms-strict-v1",
}
ADDITIONS = (
    "us-v5-reviewed-long-org-prose-strict-v1",
    "us-reviewed-extra-long-org-prose-strict-v1",
    "us-reviewed-two-paragraph-org-prose-strict-v1",
    "us-govman-reviewed-org-prose-v2",
    "us-address-prefix-reviewed-v1",
    "us-official-mail-code-reviewed-v1",
    "us-person-long-reviewed-v2",
    "us-person-long-reviewed-v3",
    "us-address-long-reviewed-v1",
    "us-person-keltner-policy-reviewed-v1",
    "us-address-long-reviewed-v2",
    "us-long-prose-reviewed-v1",
)
EVALUATION = tuple(ROOT / "data/interim/review" / name for name in (
    "us-eval-exclusions-v4.jsonl",
    "us-long-org-eval-gold-v4.jsonl",
    "us-doe-org-overviews-strict-gold-v1.jsonl",
))


def rows(path):
    """Read a frozen JSONL source."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    """Hash a frozen data file before checking its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposed_sources():
    """Resolve the exact next-run inputs while preserving the historical config."""
    match = re.search(r"(?m)^silver\s*=\s*(\[[^\]]*\])", CONFIG.read_text())
    if match is None:
        raise ValueError("V4 silver config is missing")
    configured = ast.literal_eval(match.group(1))
    expected = [f"data/interim/silver/{name}.jsonl" for name, _ in V4_SILVER]
    if configured != expected or any(configured.count(
            f"data/interim/silver/{name}.jsonl") != 1 for name in REPLACEMENTS):
        raise ValueError("V4 historical silver membership changed")
    source_paths = [ROOT / f"data/interim/silver/{REPLACEMENTS.get(Path(path).stem, Path(path).stem)}.jsonl"
                    for path in configured]
    source_paths += [ROOT / f"data/interim/silver/{name}.jsonl"
                     for name in ADDITIONS]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("proposed US real data repeats a file")
    return source_paths


def verify_source_manifest(path, *, require_eligible):
    """Check every proposed file, including unchanged historical inputs."""
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest["sha256"] != digest(path):
        raise ValueError(f"reviewed silver changed: {path}")
    if require_eligible and not manifest.get("training_eligible", False):
        raise ValueError(f"new reviewed silver is not eligible: {path}")
    for input_name, expected_hash in manifest.get("input_sha256", {}).items():
        if digest(ROOT / input_name) != expected_hash:
            raise ValueError(f"reviewed silver input changed: {input_name}")


def main():
    """Fail on source, labeled referent, or new-prose leakage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-version", choices=("v4", "v5"), default="v4")
    version = parser.parse_args().synthetic_version
    processed = ROOT / f"data/processed/detector-us-{version}"
    source_paths = proposed_sources()
    replaced_paths = [ROOT / f"data/interim/silver/{name}.jsonl"
                      for name in REPLACEMENTS.values()]
    new_paths = set(replaced_paths) | set(source_paths[-len(ADDITIONS):])
    for path in source_paths:
        verify_source_manifest(path, require_eligible=path in new_paths)
    synthetic_manifest = json.loads((ROOT / f"data/manifests/{version}/detector-synthetic.json").read_text())
    for split in ("train", "valid", "test"):
        path = processed / f"{split}.parquet"
        if synthetic_manifest["synthetic_parquet_sha256"][split] != digest(path):
            raise ValueError(f"{version} synthetic {split} shard changed")
    source_rows = [(path, rows(path)) for path in source_paths]
    real = [row for _, group in source_rows for row in group]
    if (len(source_rows) != 34 or len(real) != 1804 or
            [len(group) for _, group in source_rows[-len(ADDITIONS):]] !=
            [34, 27, 16, 12, 15, 4, 5, 4, 16, 1, 8, 2]):
        raise ValueError("proposed US real data membership changed")
    evaluation = [row for path in EVALUATION for row in rows(path)]
    if len(evaluation) != 371:
        raise ValueError("US reserved evaluation membership changed")
    train_keys = {document_key(row) for row in real}
    eval_keys = {document_key(row) for row in evaluation}
    if None in train_keys or train_keys & eval_keys:
        raise ValueError("proposed real data shares an evaluation source")
    train_urls = {source_url_key(row) for row in real} - {None}
    eval_urls = {source_url_key(row) for row in evaluation} - {None}
    if train_urls & eval_urls:
        raise ValueError("proposed real data shares an evaluation page URL")
    real_hits = collisions(
        evaluation, ((row["id"], row["text"], row["entities"]) for row in real),
        strict_aliases=True)
    if real_hits:
        raise ValueError(f"proposed real data shares held-out labels: {sorted(real_hits)}")
    contacts = contact_collisions(
        evaluation, ((row["id"], row["text"], row["entities"]) for row in real))
    substantive_contacts = {
        name: {kind: matches for kind, matches in kinds.items()
               if kind != "phone" or any(re.sub(r"\D", "", value) != "711"
                                         for _, value in matches)}
        for name, kinds in contacts.items()
    }
    substantive_contacts = {name: kinds for name, kinds in substantive_contacts.items()
                            if kinds}
    if substantive_contacts:
        raise ValueError(f"proposed real data shares held-out contacts: "
                         f"{sorted(substantive_contacts)}")
    synthetic_hits = collisions(
        evaluation, synthetic_sources(processed),
        strict_aliases=True)
    if synthetic_hits:
        raise ValueError(f"{version} synthetic data shares held-out labels: {sorted(synthetic_hits)}")
    synthetic_contacts = contact_collisions(
        evaluation, synthetic_sources(processed))
    if synthetic_contacts:
        raise ValueError(f"{version} synthetic data shares held-out contacts: "
                         f"{sorted(synthetic_contacts)}")
    heldout_fragments = set().union(*(shingles(row["input"], 12)
                                      for row in evaluation))
    new_rows = [row for _, group in source_rows[-len(ADDITIONS):] for row in group]
    repeated_new = [row["id"] for row in new_rows
                    if shingles(row["text"], 12) & heldout_fragments]
    if repeated_new:
        raise ValueError(f"new real prose repeats evaluation: {repeated_new}")
    print(f"US proposed split passed ({version} synthetic): {len(real)} real rows, "
          f"{len(new_rows)} new rows, {len(evaluation)} evaluation cases, "
          "zero held-out source, substantive label/contact, or new-prose overlaps; "
          "shared 711 relay number accepted")


if __name__ == "__main__":
    main()
