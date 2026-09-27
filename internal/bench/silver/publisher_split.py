"""Validate reviewed publisher roles for a development-disjoint silver export."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


ENTITY_ID = re.compile(r"^[a-z][a-z0-9_-]*(?::[a-z0-9][a-z0-9._-]*)+$")
ROLES = ("issuers", "controlling_parents", "origin_publishers", "source_families",
         "subjects", "governance")
BLOCKING_ROLES = ("issuers", "controlling_parents", "origin_publishers")
COMPLETE_ROLES = ("issuers", "controlling_parents", "origin_publishers", "source_families")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def registry_aliases(registry):
    """Map reviewed aliases to unique canonical IDs without guessing from names."""
    if (not isinstance(registry, dict) or registry.get("status") != "sealed"
            or not isinstance(registry.get("entities"), list)):
        raise ValueError("publisher registry is not sealed")
    aliases = {}
    kinds = {}
    for entity in registry["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("invalid publisher registry entity")
        canonical = entity.get("id")
        kind = entity.get("kind")
        names = entity.get("aliases")
        if (not isinstance(canonical, str) or not ENTITY_ID.fullmatch(canonical)
                or not isinstance(kind, str) or kind not in {"institution", "source_family"}
                or not isinstance(names, list) or not all(isinstance(v, str) for v in names)):
            raise ValueError("invalid publisher registry entity")
        for name in [canonical, *names]:
            if not ENTITY_ID.fullmatch(name) or name in aliases:
                raise ValueError(f"duplicate or invalid publisher alias {name}")
            aliases[name] = canonical
        kinds[canonical] = kind
    if not aliases:
        raise ValueError("empty publisher registry")
    return aliases, kinds


def reviewed_role_ids(record, aliases, kinds):
    """Resolve each reviewed role without merging institutions and source families."""
    if not isinstance(record, dict) or set(record) != {"roles", "completeness", "reviewers", "evidence"}:
        raise ValueError("publisher role record shape is invalid")
    roles = record["roles"]
    completion = record["completeness"]
    reviewers = record["reviewers"]
    if (not isinstance(roles, dict) or set(roles) != set(ROLES)
            or not isinstance(completion, dict) or set(completion) != set(COMPLETE_ROLES)
            or any(completion[role] != "complete" for role in COMPLETE_ROLES)
            or not isinstance(reviewers, list) or len(reviewers) < 2
            or any(not isinstance(v, str) or not v.strip() for v in reviewers)
            or len(set(reviewers)) != len(reviewers)
            or not isinstance(record["evidence"], str) or not record["evidence"].strip()):
        raise ValueError("publisher roles lack complete independent review")
    result = {role: set() for role in ROLES}
    for role in ROLES:
        values = roles[role]
        if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"invalid {role} publisher role list")
        for value in values:
            if not isinstance(value, str) or value not in aliases:
                raise ValueError(f"unknown publisher ID {value}")
            canonical = aliases[value]
            expected = "source_family" if role == "source_families" else "institution"
            if kinds[canonical] != expected:
                raise ValueError(f"wrong publisher entity kind for {role}: {value}")
            result[role].add(canonical)
    if not roles["issuers"]:
        raise ValueError("publisher roles have no page issuer")
    return result


def blocking_ids(roles):
    """Return issuer, parent, and original-publisher closure only."""
    return set().union(*(roles[role] for role in BLOCKING_ROLES))


def load_development(manifest_path, gold_sha, gold_rows, root, pinned_manifest_sha,
                     pinned_index_sha, pinned_registry_sha, pinned_source_index_sha):
    """Load a code-pinned complete index for the exact selected development gold."""
    if manifest_path is None:
        raise ValueError("missing sealed development publisher manifest")
    if not all((pinned_manifest_sha, pinned_index_sha, pinned_registry_sha,
                pinned_source_index_sha)):
        raise ValueError("development publisher manifest, index, registry, and source index lack code pins")
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != pinned_manifest_sha:
        raise ValueError("development publisher manifest SHA differs from code pin")
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get("status") != "sealed"
            or not all(isinstance(manifest.get(v), str) and manifest[v].strip()
                       for v in ("version", "reviewer", "reviewed_at"))):
        raise ValueError("development publisher manifest is not sealed")
    if manifest.get("source_family_policy") not in {"record_only", "disjoint"}:
        raise ValueError("development publisher source-family policy missing")
    if manifest.get("gold_sha256") != gold_sha:
        raise ValueError("development publisher manifest uses different gold")
    if (manifest.get("index_sha256") != pinned_index_sha
            or manifest.get("registry_sha256") != pinned_registry_sha
            or manifest.get("source_index_sha256") != pinned_source_index_sha):
        raise ValueError("development publisher identity differs from code pins")
    index_path = Path(manifest["index_path"])
    registry_path = Path(manifest["registry_path"])
    source_index_path = Path(manifest["source_index_path"])
    if (digest(index_path.read_bytes()) != pinned_index_sha
            or digest(registry_path.read_bytes()) != pinned_registry_sha
            or digest(source_index_path.read_bytes()) != pinned_source_index_sha):
        raise ValueError("development publisher index, registry, or source index SHA mismatch")
    aliases, kinds = registry_aliases(json.loads(registry_path.read_text()))
    index = read_jsonl(index_path)
    source_index = read_jsonl(source_index_path)
    if (len(index) != len(gold_rows) or len(source_index) != len(gold_rows)
            or manifest.get("rows") != len(gold_rows)):
        raise ValueError("development publisher index row count mismatch")
    expected = {row["name"] if "name" in row else row["id"]: row for row in gold_rows}
    if len(expected) != len(gold_rows):
        raise ValueError("development publisher gold has duplicate IDs")
    sources = {}
    for source in source_index:
        if not isinstance(source, dict):
            raise ValueError("invalid development source index row")
        identity = source.get("id")
        if identity in sources or identity not in expected:
            raise ValueError("development source index ID mismatch")
        sources[identity] = source
        gold = expected[identity]
        gold_text = gold["input"] if "input" in gold else gold["text"]
        if (source.get("gold_text_sha256") != digest(gold_text.encode())
                or source.get("country") != gold.get("country")):
            raise ValueError(f"development source index gold mismatch {identity}")
        path_value = source.get("raw_path")
        if not isinstance(path_value, str):
            raise ValueError(f"development source raw path missing {identity}")
        raw_path = root / path_value
        if not raw_path.resolve().is_relative_to((root / "data/raw").resolve()):
            raise ValueError(f"development source raw path outside tree {identity}")
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        raw_url = source.get("raw_url")
        host = urlsplit(raw_url).hostname if isinstance(raw_url, str) else None
        host = host.lower().removeprefix("www.") if host else None
        country_host = f"{gold.get('country')}|{host}"
        if (digest(raw_bytes) != source.get("raw_file_sha256")
                or not isinstance(raw, dict) or raw.get("id") != identity
                or raw.get("text") != gold_text or raw.get("url") != raw_url
                or raw.get("country") != gold.get("country")
                or raw.get("source") != source.get("source")
                or source.get("source_host") != host
                or gold.get("source_host") != country_host):
            raise ValueError(f"development source raw/gold mismatch {identity}")
    seen = set()
    blocks = set()
    families = set()
    for row in index:
        if not isinstance(row, dict):
            raise ValueError("invalid development publisher index row")
        identity = row.get("id")
        if identity in seen or identity not in expected:
            raise ValueError("development publisher index ID mismatch")
        seen.add(identity)
        source = sources[identity]
        if (row.get("text_sha256") != source["gold_text_sha256"]
                or any(row.get(field) != source[field] for field in
                       ("raw_path", "raw_url", "raw_file_sha256"))):
            raise ValueError(f"development publisher raw/source binding mismatch {identity}")
        roles = reviewed_role_ids(row.get("publisher"), aliases, kinds)
        blocks.update(blocking_ids(roles))
        families.update(roles["source_families"])
    if seen != set(expected):
        raise ValueError("development publisher index incomplete")
    return blocks, families, aliases, kinds, manifest["source_family_policy"]
