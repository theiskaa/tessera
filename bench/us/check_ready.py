"""Check US input integrity and pinned evaluation separation before training."""

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from active_sources import ACTIVE_SILVER, SAFE_REPLACEMENTS, V3_SILVER
from address_keys import address_keys
from build_corrected_dev import OUT as CORRECTED_GOLD
from build_corrected_dev import corrected_rows
from build_silver import legacy_cutoff
from freeze_federal_org_packet import FROZEN_SHA256 as RAW_ORG_PACKET_SHA256
from freeze_park_address_challenge import case_for as park_case_for
from freeze_r23_org_packet import FROZEN_SHA256 as R23_FIRST_PACKET_SHA256
from freeze_r23_org_packet_remaining import FROZEN_SHA256 as R23_SECOND_PACKET_SHA256
from org_aliases import active_aliases
from silver_dedupe import NearDuplicateIndex, NearTextIndex


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
SILVER = ROOT / "data/interim/silver"
PARSER = ROOT / "data/processed/parser-us-v1"
KIND = {"person", "org", "address"}
SOURCE_ARTIFACT = re.compile(r'">|</?[A-Za-z][^>]*>|&(?:amp|quot|nbsp|lt|gt);')
ACRONYM = re.compile(r"[A-Z][A-Z0-9]{1,8}\Z")
ADDRESS_PREFIX = re.compile(r"\b(?:room|suite|building|floor|mail code)\b|\bms:", re.I)


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


def matches_reviewed_source(candidate, snippet):
    source = candidate["text"].encode()
    piece = snippet["text"].encode()
    offset = source.find(piece)
    while offset >= 0:
        end = offset + len(piece)
        crossing = any(span["start"] < end and offset < span["end"]
                       and not (offset <= span["start"] and span["end"] <= end)
                       for span in candidate["entities"])
        labels = [{"kind": span["kind"], "start": span["start"] - offset,
                   "end": span["end"] - offset}
                  for span in candidate["entities"]
                  if offset <= span["start"] and span["end"] <= end]
        if not crossing and labels == snippet["entities"]:
            return True
        offset = source.find(piece, offset + 1)
    return False


def agreed_blind_rows(packet_path, packet_hash, review_paths, review_hashes):
    require(digest(packet_path.read_bytes()) == packet_hash,
            f"blind packet changed: {packet_path.name}")
    packet_rows = load(packet_path)
    packet = {row["name"]: row for row in packet_rows}
    require(len(packet) == len(packet_rows), "duplicate blind case")
    review_maps = []
    for path in review_paths:
        require(digest(path.read_bytes()) == review_hashes[path.name],
                f"blind review changed: {path.name}")
        rows = load(path)
        review = {row["name"]: row for row in rows}
        require(len(review) == len(rows) and set(review) == set(packet),
                "blind review is incomplete or has duplicate cases")
        review_maps.append(review)
    agreed = {}
    for name, case in packet.items():
        text = case["input"]
        reviews = [review[name] for review in review_maps]
        labels = [sorted((kind, start, end) for kind, start, end, _ in
                         spans({"input": text, "expected": review["entities"]}))
                  for review in reviews]
        require(all(kind in KIND for group in labels for kind, _, _ in group),
                f"unknown blind label kind: {name}")
        if not any(review.get("uncertain") for review in reviews) and labels[0] == labels[1]:
            agreed[name] = (case, labels[0])
    return agreed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", action="store_true", help="check the isolated v3 inputs")
    args = parser.parse_args()
    processed = ROOT / f"data/processed/detector-us-v{3 if args.v3 else 2}"
    config_path = ROOT / f"configs/detector-shared{'-v3' if args.v3 else ''}.toml"
    generator_path = ROOT / "data/manifests" / ("v3" if args.v3 else "") / "detector-synthetic.json"
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("check_ready needs pyarrow to read the generated Parquet data") from error

    exclusion_path = REVIEW / "us-eval-exclusions-v1.jsonl"
    legacy_exclusion_hash = digest(exclusion_path.read_bytes())
    contact_split = json.loads((REVIEW / "us-2025-contact-split-v2.manifest.json").read_text())
    contact_holdout = REVIEW / "us-2025-contact-holdout-v2.jsonl"
    require(contact_split["holdout_sha256"] == digest(contact_holdout.read_bytes()) and
            contact_split["gold_sha256"] ==
            digest((REVIEW / "us-2025-contact-gold-v5.jsonl").read_bytes()),
            "2025 contact holdout changed")
    year_holdout = REVIEW / "us-2025-year-contact-holdout-v2.jsonl"
    year_manifest = json.loads(year_holdout.with_suffix(".manifest.json").read_text())
    require(year_manifest["sha256"] == digest(year_holdout.read_bytes()) and
            year_manifest["strict_entity_holdout"] and
            not year_manifest["training_eligible"] and
            year_manifest["source_gold_sha256"] ==
            digest((REVIEW / "us-2025-year-contact-gold-v1.jsonl").read_bytes()),
            "2025 full-year contact holdout changed")
    exclusions = load(exclusion_path) + load(contact_holdout) + load(year_holdout)
    generator_exclusion_hash = legacy_exclusion_hash
    if args.v3:
        from build_us_full_eval_exclusions import MANIFEST as FULL_MANIFEST
        from build_us_full_eval_exclusions import OUT as FULL_EXCLUSIONS
        from build_us_full_eval_exclusions import combined_rows
        combined, source_hashes = combined_rows()
        full_manifest = json.loads(FULL_MANIFEST.read_text())
        require(FULL_EXCLUSIONS.read_bytes() == b"".join(
            (REVIEW / name).read_bytes() for name in source_hashes) and
            full_manifest["kind"] == "us_full_eval_exclusions_v2" and
            full_manifest["cases"] == len(combined) and
            full_manifest["source_sha256"] == source_hashes and
            full_manifest["sha256"] == digest(FULL_EXCLUSIONS.read_bytes()) and
            not full_manifest["training_eligible"],
            "full US evaluation exclusions changed")
        from build_us_v3_eval_exclusions import MANIFEST as V3_MANIFEST
        from build_us_v3_eval_exclusions import OUT as V3_EXCLUSIONS
        from build_us_v3_eval_exclusions import exclusion_rows
        exclusions = exclusion_rows()
        v3_manifest = json.loads(V3_MANIFEST.read_text())
        require(V3_EXCLUSIONS.read_bytes() ==
                FULL_EXCLUSIONS.read_bytes() +
                (REVIEW / "us-nrcs-maine-reserved-v1.jsonl").read_bytes() and
                v3_manifest["kind"] == "us_v3_eval_exclusions_v1" and
                v3_manifest["cases"] == len(exclusions) and
                v3_manifest["source_sha256"] == {
                    str(path.relative_to(ROOT)): digest(path.read_bytes())
                    for path in (FULL_EXCLUSIONS,
                                 REVIEW / "us-nrcs-maine-reserved-v1.jsonl")} and
                v3_manifest["sha256"] == digest(V3_EXCLUSIONS.read_bytes()) and
                not v3_manifest["training_eligible"],
                "US v3 evaluation exclusions changed")
        generator_exclusion_hash = v3_manifest["sha256"]
    names = set()
    texts = set()
    surfaces = collections.defaultdict(set)
    people = set()
    addresses = set()
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
                if kind == "address":
                    addresses.update(address_keys(value))

    corrected, corrected_base, _ = corrected_rows()
    corrected_manifest = json.loads(CORRECTED_GOLD.with_suffix(".manifest.json").read_text())
    require(corrected_manifest["sha256"] == digest(CORRECTED_GOLD.read_bytes()) and
            corrected_manifest["base_sha256"] == digest(corrected_base) and
            load(CORRECTED_GOLD) == corrected,
            "corrected development gold differs from its reviewed errata")
    development_mix = collections.Counter()
    for row in corrected:
        for kind, _, _, value in spans(row):
            if kind == "org":
                development_mix["org"] += 1
                development_mix["acronym"] += bool(ACRONYM.fullmatch(value))
            if kind == "address":
                development_mix["address"] += 1
                development_mix["address_prefix"] += bool(ADDRESS_PREFIX.search(value))
    excluded_by_name = {row["name"]: row for row in exclusions}
    for row in corrected:
        original = excluded_by_name.get(row["name"])
        require(original is not None and row["input"] == original["input"],
                "corrected development case is not in the training exclusions")
        for kind, _, _, value in spans(row):
            if kind in KIND:
                surfaces[kind].add(normalized(value))
                if kind == "person" and person_key(value):
                    people.add(person_key(value))
                if kind == "address":
                    addresses.update(address_keys(value))
    org_aliases = active_aliases(exclusions + corrected, include_roster=True)

    staff_manifest = json.loads((REVIEW / "us-staff-challenge-v1.manifest.json").read_text())
    staff_data = (REVIEW / "us-staff-challenge-v1.jsonl").read_bytes()
    require(digest(staff_data) == staff_manifest["sha256"], "staff challenge changed")
    park_manifest = json.loads((REVIEW / "us-park-address-challenge-v1.manifest.json").read_text())
    park_data = (REVIEW / "us-park-address-challenge-v1.jsonl").read_bytes()
    require(digest(park_data) == park_manifest["sha256"], "park challenge changed")
    ky_manifest = json.loads((REVIEW / "us-ky-superintendent-challenge-v1.manifest.json").read_text())
    ky_data = (REVIEW / "us-ky-superintendent-challenge-v1.jsonl").read_bytes()
    require(digest(ky_data) == ky_manifest["sha256"], "Kentucky challenge changed")
    ca_manifest = json.loads((REVIEW / "us-ca-superintendent-challenge-v1.manifest.json").read_text())
    ca_data = (REVIEW / "us-ca-superintendent-challenge-v1.jsonl").read_bytes()
    require(digest(ca_data) == ca_manifest["sha256"], "California challenge changed")
    or_manifest = json.loads((REVIEW / "us-or-district-challenge-v1.manifest.json").read_text())
    or_data = (REVIEW / "us-or-district-challenge-v1.jsonl").read_bytes()
    require(digest(or_data) == or_manifest["sha256"], "Oregon challenge changed")
    pa_manifest = json.loads((REVIEW / "us-pa-room-challenge-v1.manifest.json").read_text())
    pa_data = (REVIEW / "us-pa-room-challenge-v1.jsonl").read_bytes()
    require(digest(pa_data) == pa_manifest["sha256"] and
            pa_manifest["status"] in {"unseen", "exposed"},
            "Pennsylvania room challenge changed")
    pa_acronym_manifest = json.loads((REVIEW / "us-pa-acronym-challenge-v1.manifest.json").read_text())
    pa_acronym_data = (REVIEW / "us-pa-acronym-challenge-v1.jsonl").read_bytes()
    require(digest(pa_acronym_data) == pa_acronym_manifest["sha256"] and
            pa_acronym_manifest["status"] in {"unseen", "exposed"},
            "Pennsylvania acronym challenge changed")
    pa_staff_manifest = json.loads((REVIEW / "us-pa-agriculture-staff-challenge-v1.manifest.json").read_text())
    pa_staff_data = (REVIEW / "us-pa-agriculture-staff-challenge-v1.jsonl").read_bytes()
    require(digest(pa_staff_data) == pa_staff_manifest["sha256"] and
            pa_staff_manifest["status"] in {"unseen", "exposed"},
            "Pennsylvania staff challenge changed")
    base_data = (REVIEW / "us-dev-office-exclusions-v1.jsonl").read_bytes()
    require(exclusion_path.read_bytes() == base_data + staff_data + park_data + ky_data + ca_data + or_data + pa_data + pa_acronym_data + pa_staff_data,
            "evaluation exclusions do not match the frozen evaluation sets")
    require(staff_manifest["existing_evaluation_sha256"] == digest(base_data),
            "staff challenge base changed")
    for slug, expected in staff_manifest["sources_sha256"].items():
        require(digest((ROOT / f"data/raw/us-release-eval/{slug}.html").read_bytes()) == expected,
                f"staff source changed: {slug}")
    for filename, expected in park_manifest["source_sha256"].items():
        require(digest((ROOT / "data/raw/us-release-eval" / filename).read_bytes()) == expected,
                f"park source changed: {filename}")

    parser_sample = json.loads((PARSER / "sample.json").read_text())
    parser_checks = parser_sample["checks"]
    require(parser_sample["filters"]["countries"] == ["US"] and parser_checks["passed"],
            "US parser sample audit did not pass")
    require(all(value == 0 for key, value in parser_checks["counts"].items()
                if key != "postcode_and_number_shared"),
            "US parser sample has an unresolved split or label error")
    for split, expected in (("train", 62549), ("valid", 3000), ("test", 3000)):
        parser_file = parquet.ParquetFile(PARSER / f"{split}.parquet")
        require(parser_file.metadata.num_rows == expected,
                f"wrong US parser {split} size")
        for batch in parser_file.iter_batches(batch_size=10000, columns=["text"]):
            for row in batch.to_pylist():
                require(not address_keys(row["text"]) & addresses,
                        f"evaluation physical address in parser {split}: {row['text']}")
                require(normalized(row["text"]) not in surfaces["address"],
                        f"evaluation address surface in parser {split}: {row['text']}")

    seen = set()
    near = NearDuplicateIndex()
    near_evaluation = NearTextIndex()
    real_counts = collections.Counter()
    real_surfaces = collections.defaultdict(set)
    real_people = set()
    real_addresses = set()
    reviewed_candidates = {row["id"]: row for row in
                           load(SILVER / "us-reviewed-candidate-v1.jsonl")}
    r23_manifest = json.loads((SILVER / "us-r23-reviewed-org-v1.manifest.json").read_text())
    r23_agreed = {}
    for version in ("v1", "v2"):
        packet_path = SILVER / f"r23/us-org-blind-{version}.jsonl"
        expected_packet = (R23_FIRST_PACKET_SHA256 if version == "v1"
                           else R23_SECOND_PACKET_SHA256)
        require(r23_manifest["packet_sha256"][packet_path.name] == expected_packet,
                f"R23 blind packet {version} is not pinned")
        review_paths = [SILVER / f"r23/us-org-review-{reviewer}-{version}.jsonl"
                        for reviewer in ("a", "b")]
        agreed = agreed_blind_rows(packet_path, r23_manifest["packet_sha256"][packet_path.name],
                                  review_paths, r23_manifest["review_sha256"])
        require(not set(r23_agreed) & set(agreed), "duplicate R23 blind case across packets")
        r23_agreed.update(agreed)
    raw_manifest = json.loads((SILVER / "us-federal-org-reviewed-v2.manifest.json").read_text())
    require(raw_manifest["packet_sha256"] == RAW_ORG_PACKET_SHA256,
            "raw blind packet is not pinned")
    raw_packet_path = SILVER / "r23/us-raw-org-blind-v1.jsonl"
    raw_review_paths = [SILVER / f"r23/us-raw-org-review-{reviewer}-v1.jsonl"
                        for reviewer in ("a", "b")]
    raw_agreed = agreed_blind_rows(raw_packet_path, RAW_ORG_PACKET_SHA256,
                                  raw_review_paths, raw_manifest["review_sha256"])
    sources = V3_SILVER if args.v3 else ACTIVE_SILVER
    safe_sources = {replacement: source for source, replacement in SAFE_REPLACEMENTS.items()}
    for filename, key in sources:
        origin_name = safe_sources.get(filename, filename)
        path = SILVER / f"{filename}.jsonl"
        manifest = json.loads((SILVER / f"{filename}.manifest.json").read_text())
        require(manifest["sha256"] == digest(path.read_bytes()), f"{filename} changed")
        rows = load(path)
        require(len(rows) == manifest["documents"], f"{filename} document count changed")
        lineage_manifest = manifest
        source_rows = rows
        if origin_name != filename:
            from build_eval_disjoint_silver import EXCLUDED_IDS, filtered_rows
            source = SILVER / f"{origin_name}.jsonl"
            source_manifest = source.with_suffix(".manifest.json")
            rebuilt, lineage_manifest = filtered_rows(origin_name)
            source_rows = load(source)
            require(rows == rebuilt and
                    manifest["kind"] == "us_silver_eval_disjoint_v1" and
                    manifest["source_name"] == origin_name and
                    manifest["source_sha256"] == digest(source.read_bytes()) and
                    manifest["source_manifest_sha256"] == digest(source_manifest.read_bytes()) and
                    manifest["filter_evaluation_sha256"] == {
                        item.name: digest(item.read_bytes()) for item in
                        (exclusion_path, contact_holdout, year_holdout)} and
                    manifest["excluded_ids"] == sorted(EXCLUDED_IDS[origin_name]),
                    f"{filename} differs from its frozen silver source")
        if origin_name in {"us-2025-year-contacts-v2", "us-2025-year-acronym-prose-v3",
                           "us-2025-year-contact-expansion-v1",
                           "us-2025-targeted-contacts-v1"}:
            require(manifest[key] == {path.name: digest(path.read_bytes()) for path in
                    (exclusion_path, contact_holdout, year_holdout)},
                    f"{filename} uses stale evaluation exclusions")
        elif origin_name in {"us-r25-reviewed-contacts-v1",
                             "us-official-reviewed-contacts-v1",
                             "us-military-reviewed-addresses-v1",
                             "us-nrcs-reviewed-field-offices-v1",
                             "us-dol-whd-reviewed-offices-v1",
                             "us-dol-whd-reviewed-remaining-v1",
                             "us-environmental-hard-negatives-v1",
                             "us-nrcs-maine-reviewed-offices-v1"}:
            require(manifest[key] == {
                name: digest((ROOT / name).read_bytes())
                for name in manifest[key]
            }, "R25 contacts use stale evaluation exclusions")
        else:
            require(manifest[key] == legacy_exclusion_hash,
                    f"{filename} uses stale evaluation exclusions")
        if origin_name == "us-reviewed-strict-v1":
            errata = ROOT / "bench/us/fixtures/us-strict-label-errata-v1.json"
            require(lineage_manifest["reviewed_errata_sha256"] == digest(errata.read_bytes()),
                    "strict silver uses stale reviewed label corrections")
        if origin_name.startswith("us-house-"):
            require(lineage_manifest["source_sha256"] == digest(
                (ROOT / "data/raw/us-staff/house-phonebook-2023.pdf").read_bytes()),
                "House directory source changed")
        if origin_name in {"us-reviewed-contact-snippets-v1", "us-reviewed-org-paragraphs-v1"}:
            candidate = SILVER / "us-reviewed-candidate-v1.jsonl"
            require(lineage_manifest["source_candidate_sha256"] == digest(candidate.read_bytes()),
                    "reviewed paragraphs use a stale candidate")
        if origin_name in {"us-2025-year-contacts-v2", "us-2025-year-acronym-prose-v3",
                           "us-2025-year-contact-expansion-v1",
                           "us-2025-targeted-contacts-v1"}:
            if origin_name == "us-2025-year-contacts-v2":
                from build_2025_year_contacts_silver import silver_rows as rebuild_silver
                candidate_path = SILVER / "r24/us-2025-year-contact-train-candidate-v6.jsonl"
                require(lineage_manifest["holdout_sha256"] == digest(year_holdout.read_bytes()),
                        "year contact silver uses stale holdout")
            elif origin_name == "us-2025-year-acronym-prose-v3":
                from build_2025_year_acronym_silver import silver_rows as rebuild_silver
                candidate_path = SILVER / "r24/us-2025-year-acronym-prose-train-candidate-v3.jsonl"
            elif origin_name == "us-2025-targeted-contacts-v1":
                from build_2025_targeted_contact_silver import silver_rows as rebuild_silver
                candidate_path = SILVER / "r24/us-2025-targeted-contact-train-candidate-v1.jsonl"
            else:
                from build_2025_year_contact_expansion_silver import silver_rows as rebuild_silver
                candidate_path = SILVER / "r24/us-2025-year-contact-expansion-train-candidate-v1.jsonl"
            rebuilt, prior_hashes, raw_hashes = rebuild_silver()
            require(source_rows == rebuilt and
                    lineage_manifest["candidate_sha256"] == digest(candidate_path.read_bytes()) and
                    lineage_manifest["prior_silver_sha256"] == prior_hashes and
                    lineage_manifest["raw_sha256"] == raw_hashes,
                    f"{filename} differs from pinned raw sources or blind reviews")
        if origin_name == "us-park-contacts-v1":
            require(lineage_manifest["source_sha256"] == park_manifest["source_sha256"],
                    "NPS silver uses stale source files")
            source_rows = json.loads((ROOT / "data/raw/us-release-eval/nps-parks.json").read_text())["DATA"]
        if origin_name == "us-or-districts-v1":
            source_path = ROOT / "bench/us/fixtures/or-district-directory-v1.jsonl"
            require(lineage_manifest["source_sha256"] == digest(source_path.read_bytes()),
                    "Oregon district source changed")
            or_source_rows = load(source_path)
        if origin_name == "us-r25-reviewed-contacts-v1":
            from build_r25_reviewed_contacts_silver import silver_rows
            rebuilt, candidate_hashes, review_hashes, evaluation_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_r25_reviewed_contacts_silver_v1" and
                    manifest["candidate_sha256"] == candidate_hashes and
                    manifest["review_sha256"] == review_hashes and
                    manifest["evaluation_sha256"] == evaluation_hashes,
                    "R25 contacts differ from pinned reviews or sources")
        if origin_name == "us-official-reviewed-contacts-v1":
            from build_us_official_contact_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_official_reviewed_contacts_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes,
                    "official contacts differ from pinned reviews or sources")
        if origin_name == "us-military-reviewed-addresses-v1":
            from build_us_military_address_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_military_reviewed_addresses_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes,
                    "military addresses differ from pinned reviews or sources")
        if origin_name == "us-nrcs-reviewed-field-offices-v1":
            from build_us_nrcs_field_office_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_nrcs_reviewed_field_offices_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_v1_sha256"] == candidate_manifest["review_a_v1_sha256"] and
                    manifest["review_a_v2_sha256"] == candidate_manifest["review_a_v2_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes and
                    manifest["heldout_names"] == candidate_manifest["heldout_names"],
                    "USDA field offices differ from pinned reviews or sources")
        if origin_name == "us-dol-whd-reviewed-offices-v1":
            from build_us_dol_whd_office_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_dol_whd_reviewed_offices_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes and
                    manifest["heldout_names"] == candidate_manifest["heldout_names"],
                    "DOL offices differ from pinned reviews or sources")
        if origin_name == "us-dol-whd-reviewed-remaining-v1":
            from build_us_dol_whd_remaining_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_dol_whd_reviewed_remaining_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["reserved_sha256"] == candidate_manifest["reserved_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes,
                    "DOL remaining offices differ from pinned reviews or sources")
        if origin_name == "us-environmental-hard-negatives-v1":
            from build_us_environmental_hard_negative_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_environmental_hard_negatives_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes,
                    "environmental negatives differ from pinned reviews or sources")
        if origin_name == "us-nrcs-maine-reviewed-offices-v1":
            from build_us_nrcs_maine_office_silver import silver_rows
            rebuilt, candidate_manifest, evaluation_hashes, prior_hashes = silver_rows()
            require(rows == rebuilt and
                    manifest["kind"] == "us_nrcs_maine_reviewed_offices_silver_v1" and
                    manifest["candidate_sha256"] == candidate_manifest["sha256"] and
                    manifest["review_a_sha256"] == candidate_manifest["review_a_sha256"] and
                    manifest["review_b_sha256"] == candidate_manifest["review_b_sha256"] and
                    manifest["evaluation_sha256"] == evaluation_hashes and
                    manifest["prior_silver_sha256"] == prior_hashes and
                    manifest["heldout_names"] == candidate_manifest["heldout_names"],
                    "USDA Maine offices differ from pinned reviews or sources")
        file_counts = collections.Counter()
        all_counts = collections.Counter()
        for row in rows:
            require(row["country"] == "US", "non-US training document")
            if origin_name == "us-reviewed-strict-v1":
                raw = json.loads((ROOT / f"data/raw/silver/federal-register/{row['id']}.json").read_text())
                require(not legacy_cutoff(raw),
                        f"legacy source cutoff in strict training document: {row['id']}")
            if origin_name in {"us-reviewed-contact-snippets-v1", "us-reviewed-org-paragraphs-v1"}:
                marker = "-contact-" if origin_name == "us-reviewed-contact-snippets-v1" else "-org-paragraph-"
                source_id = row["id"].split(marker, 1)[0]
                raw = json.loads((ROOT / f"data/raw/silver/federal-register/{source_id}.json").read_text())
                require(row["source_url"] == raw["url"] and row["text"] in raw["text"],
                        f"reviewed paragraph lost source lineage: {row['id']}")
                require(source_id in reviewed_candidates and
                        matches_reviewed_source(reviewed_candidates[source_id], row),
                        f"reviewed paragraph labels differ from two-pass source: {row['id']}")
            if origin_name == "us-r23-reviewed-org-v1":
                require(row["id"] in r23_agreed, f"R23 label was not agreed: {row['id']}")
                case, agreed = r23_agreed[row["id"]]
                source_id = row["id"].removeprefix("r23-us-org-").rsplit("-", 1)[0]
                raw = json.loads((ROOT / f"data/raw/silver/federal-register/{source_id}.json").read_text())
                require(row["text"] == case["input"] and
                        row["source_url"] == case["source_url"] == raw["url"] and
                        row["text"] in raw["text"],
                        f"R23 paragraph lost source lineage: {row['id']}")
                require(sorted((span["kind"], span["start"], span["end"])
                               for span in row["entities"]) == agreed,
                        f"R23 paragraph differs from two blind reviews: {row['id']}")
            if origin_name == "us-federal-org-reviewed-v2":
                require(row["id"] in raw_agreed, f"raw label was not agreed: {row['id']}")
                case, agreed = raw_agreed[row["id"]]
                source_id = row["id"].removeprefix("raw-us-org-").rsplit("-", 1)[0]
                raw = json.loads((ROOT / f"data/raw/silver/federal-register/{source_id}.json").read_text())
                require(row["text"] == case["input"] and
                        row["source_url"] == case["source_url"] == raw["url"] and
                        row["text"] in raw["text"] and row["source_group"] == case["group"],
                        f"raw paragraph lost source lineage: {row['id']}")
                require(sorted((span["kind"], span["start"], span["end"])
                               for span in row["entities"]) == agreed,
                        f"raw paragraph differs from two blind reviews: {row['id']}")
            if origin_name == "us-park-contacts-v1":
                index = row["source_json_index"]
                case = park_case_for(index, source_rows[index])
                require(case is not None and row["text"] == case["input"],
                        f"NPS contact lost source lineage: {row['id']}")
                require(row["source_url"] == case["source_url"],
                        f"NPS source page mismatch: {row['id']}")
                require(row["source_data_url"].endswith("aboutus-NPSParkandSuperintendentList2023.csv"),
                        f"NPS data URL mismatch: {row['id']}")
                require(row["entities"] == [
                    {"kind": span["kind"], "start": span["start"], "end": span["end"]}
                    for span in case["expected"] if span["kind"] in KIND],
                    f"NPS labels lost source lineage: {row['id']}")
            if origin_name == "us-or-districts-v1":
                index = int(row["id"].rsplit("-", 1)[1])
                source_row = or_source_rows[index]
                district = source_row["district"]
                require(row["text"] == f"District: {district}" and
                        row["source_url"] == lineage_manifest["source_url"] and
                        row["source_line"] == source_row["source_line"] and
                        row["entities"] == [{"kind": "org", "start": 10,
                                             "end": 10 + len(district.encode())}],
                        f"Oregon district lost source lineage: {row['id']}")
            text_hash = digest(row["text"].encode())
            require(text_hash not in seen and text_hash not in texts,
                    "duplicate or evaluation training text")
            require(not SOURCE_ARTIFACT.search(row["text"]), "HTML artifact in real training text")
            require(not near.prior(row), f"near-duplicate real training text: {row['id']}")
            seen.add(text_hash)
            near.add(row)
            near_evaluation.add(row["id"], row["text"])
            for kind, _, _, value in spans(row):
                all_counts[kind] += 1
                if kind not in KIND:
                    continue
                require(normalized(value) not in surfaces[kind],
                        f"evaluation entity in real training: {value}")
                if kind == "address":
                    require(not address_keys(value) & addresses,
                            f"evaluation physical address in real training: {value}")
                if kind == "person":
                    require(person_key(value) not in people,
                            f"evaluation person alias in real training: {value}")
                if kind == "org":
                    require(normalized(value) not in org_aliases,
                            f"evaluation organization alias in real training: {value}")
                real_counts[kind] += 1
                file_counts[kind] += 1
                real_surfaces[kind].add(normalized(value))
                if kind == "person" and person_key(value):
                    real_people.add(person_key(value))
                if kind == "address":
                    real_addresses.update(address_keys(value))
        if origin_name == "us-house-staff-v1":
            require(file_counts["person"] == manifest["counts"]["person_labels"],
                    "House label count changed")
        elif filename in {"us-2025-year-contacts-v2", "us-2025-year-contact-expansion-v1",
                          "us-2025-targeted-contacts-v1", "us-r25-reviewed-contacts-v1",
                          "us-official-reviewed-contacts-v1",
                          "us-military-reviewed-addresses-v1",
                          "us-nrcs-reviewed-field-offices-v1",
                          "us-dol-whd-reviewed-offices-v1",
                          "us-dol-whd-reviewed-remaining-v1",
                          "us-environmental-hard-negatives-v1",
                          "us-nrcs-maine-reviewed-offices-v1"}:
            require(dict(all_counts) == manifest["labels"] and
                    dict(file_counts) == manifest["model_labels"] and
                    {kind: all_counts[kind] for kind in ("email", "phone")} ==
                    manifest["rule_labels"],
                    "reviewed year contact label count changed")
        else:
            require(dict(file_counts) == manifest["labels"],
                    f"{filename} label count changed")

    for row in exclusions:
        prior = near_evaluation.prior(row["input"])
        require(prior is None,
                f"evaluation text nearly duplicates real training: {row['name']} / {prior}")

    if args.v3:
        from build_us_reserved_office_diagnostic import (
            MANIFEST as RESERVED_MANIFEST, OUT as RESERVED, diagnostic_rows,
        )
        reserved_rows, reserved_excluded = diagnostic_rows()
        reserved_bytes = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in reserved_rows).encode()
        reserved_manifest = json.loads(RESERVED_MANIFEST.read_text())
        require(RESERVED.read_bytes() == reserved_bytes and
                reserved_manifest["kind"] == "us_reserved_office_diagnostic_v1" and
                reserved_manifest["cases"] == len(reserved_rows) and
                reserved_manifest["excluded_uncertain"] == reserved_excluded and
                reserved_manifest["sha256"] == digest(reserved_bytes) and
                reserved_manifest["source_sha256"] == {
                    path: digest((ROOT / path).read_bytes())
                    for path in reserved_manifest["source_sha256"]} and
                not reserved_manifest["training_eligible"],
                "reserved office diagnostic changed or entered training")
        from build_us_nrcs_maine_reserved_gold import (
            MANIFEST as MAINE_MANIFEST, OUT as MAINE_GOLD, gold_rows,
        )
        maine_rows = gold_rows()
        maine_bytes = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in maine_rows).encode()
        maine_manifest = json.loads(MAINE_MANIFEST.read_text())
        require(MAINE_GOLD.read_bytes() == maine_bytes and
                maine_manifest["kind"] == "us_nrcs_maine_reserved_gold_v1" and
                maine_manifest["cases"] == len(maine_rows) and
                maine_manifest["sha256"] == digest(maine_bytes) and
                maine_manifest["packet_sha256"] == digest((ROOT /
                    "data/raw/candidates/us-nrcs-maine-offices-v1/blind-v1.jsonl").read_bytes()) and
                maine_manifest["review_a_sha256"] == digest((REVIEW /
                    "us-nrcs-maine-office-review-a-v1.jsonl").read_bytes()) and
                maine_manifest["review_b_sha256"] == digest((REVIEW /
                    "us-nrcs-maine-office-review-b-v1.jsonl").read_bytes()) and
                not maine_manifest["training_eligible"],
                "USDA Maine reserved gold changed or entered training")

    generator = json.loads(generator_path.read_text())
    require(generator.get("version") == (3 if args.v3 else 2) and
            generator.get("template_policy") == (
                "source_backed_us_unit_parent_v1" if args.v3 else
                "exclude_unverified_us_unit_parent_v1"),
            "synthetic data predates the US organization coherence rule")
    require(generator.get("synthetic_parquet_sha256") == {
        split: digest((processed / f"{split}.parquet").read_bytes())
        for split in ("train", "valid", "test")
    }, "synthetic parquet files differ from their manifest")
    require(generator["exclude_gold_sha256"] == generator_exclusion_hash,
            "synthetic data uses stale evaluation exclusions")
    roster = ROOT / "trainer/data/us-public-bodies.json"
    require(generator["public_bodies_sha256"] == digest(roster.read_bytes()),
            "synthetic data uses stale public bodies")
    expected_real_hashes = {f"data/interim/silver/{name}.jsonl":
                            digest((SILVER / f"{name}.jsonl").read_bytes())
                            for name, _ in sources}
    require(generator["real_silver_sha256"] == expected_real_hashes,
            "synthetic validation/test uses stale real training exclusions")
    hierarchy_ids = set()
    hierarchy_pairs = set()
    if args.v3:
        roster_path = ROOT / "trainer/data/us-org-hierarchy-pairs.json"
        roster = json.loads(roster_path.read_text())
        hierarchy_ids = set(generator["hierarchy_template_ids"])
        hierarchy_pairs = {(normalized(item["child"]), normalized(item["parent"]))
                           for item in roster["pairs"]}
        require(generator["hierarchy_roster_sha256"] == digest(roster_path.read_bytes()) and
                generator["hierarchy_roster_pairs"] == len(hierarchy_pairs) and
                generator["hierarchy_train_rows"] > 0 and hierarchy_ids,
                "synthetic hierarchy roster or template membership changed")
    real_org_aliases = active_aliases([{"expected": [{"kind": "org", "text": value}
                                                  for value in real_surfaces["org"]]}],
                                      include_roster=True)
    synthetic_counts = collections.Counter()
    synthetic_mix = collections.Counter()
    synthetic_templates = collections.Counter()
    hierarchy_train_rows = 0
    synthetic_prior = collections.defaultdict(set)
    synthetic_current = collections.defaultdict(set)
    synthetic_prior_people = set()
    synthetic_current_people = set()
    synthetic_prior_addresses = set()
    synthetic_current_addresses = set()
    for split, expected in (("train", 170000), ("valid", 10000), ("test", 20000)):
        if args.v3 and split != "train":
            for kind, values in synthetic_current.items():
                synthetic_prior[kind].update(values)
            synthetic_prior_people.update(synthetic_current_people)
            synthetic_prior_addresses.update(synthetic_current_addresses)
            synthetic_current = collections.defaultdict(set)
            synthetic_current_people = set()
            synthetic_current_addresses = set()
        file = parquet.ParquetFile(processed / f"{split}.parquet")
        require(file.metadata.num_rows == expected, f"wrong {split} size")
        for batch in file.iter_batches(batch_size=10000,
                                       columns=["country", "text", "entities_json", "template_id"]):
            for row in batch.to_pylist():
                require(row["country"] == "US", "non-US synthetic document")
                if split == "train":
                    synthetic_templates[row["template_id"]] += 1
                labels = json.loads(row["entities_json"])
                if row["template_id"] in hierarchy_ids:
                    require(split == "train", "hierarchy template entered validation or test")
                    org_values = {normalized(value) for kind, _, _, value in
                                  spans({"text": row["text"], "entities": labels})
                                  if kind == "org"}
                    require(any(child in org_values and parent in org_values
                                for child, parent in hierarchy_pairs),
                            "hierarchy template used an unreviewed organization pair")
                    hierarchy_train_rows += 1
                for kind, _, _, value in spans({"text": row["text"], "entities": labels}):
                    if kind in KIND:
                        surface = normalized(value)
                        if args.v3:
                            require(surface not in synthetic_prior[kind],
                                    f"synthetic {split} {kind} repeats an earlier split: {value}")
                            if kind == "person":
                                require(person_key(value) not in synthetic_prior_people,
                                        f"synthetic {split} person alias repeats an earlier split: {value}")
                                if person_key(value):
                                    synthetic_current_people.add(person_key(value))
                            if kind == "address":
                                physical = address_keys(value)
                                require(not physical & synthetic_prior_addresses,
                                        f"synthetic {split} address repeats an earlier physical location: {value}")
                                synthetic_current_addresses.update(physical)
                            synthetic_current[kind].add(surface)
                        if split == "train":
                            if kind == "org":
                                synthetic_mix["org"] += 1
                                synthetic_mix["acronym"] += bool(ACRONYM.fullmatch(value))
                            if kind == "address":
                                synthetic_mix["address"] += 1
                                synthetic_mix["address_prefix"] += bool(
                                    ADDRESS_PREFIX.search(value))
                        require(normalized(value) not in surfaces[kind],
                                f"evaluation entity in synthetic {split}: {value}")
                        if kind == "address":
                            require(not address_keys(value) & addresses,
                                    f"evaluation physical address in synthetic {split}: {value}")
                        if kind == "person":
                            require(person_key(value) not in people,
                                    f"evaluation person alias in synthetic {split}: {value}")
                        if kind == "org":
                            require(normalized(value) not in org_aliases,
                                    f"evaluation organization alias in synthetic {split}: {value}")
                        if split != "train":
                            require(normalized(value) not in real_surfaces[kind],
                                    f"real training entity in synthetic {split}: {value}")
                            if kind == "person":
                                require(person_key(value) not in real_people,
                                        f"real training person alias in synthetic {split}: {value}")
                            if kind == "org":
                                require(normalized(value) not in real_org_aliases,
                                        f"real training organization alias in synthetic {split}: {value}")
                            if kind == "address":
                                require(not address_keys(value) & real_addresses,
                                        f"real training physical address in synthetic {split}: {value}")
                        synthetic_counts[kind] += 1
    if args.v3:
        require(hierarchy_train_rows == generator["hierarchy_train_rows"],
                "hierarchy row count differs from synthetic manifest")
    for subset, total in (("acronym", "org"), ("address_prefix", "address")):
        require(development_mix[total] > 0 and synthetic_mix[total] > 0,
                f"missing {total} labels for data mix check")
        reference = development_mix[subset] / development_mix[total]
        training = synthetic_mix[subset] / synthetic_mix[total]
        require(abs(reference - training) <= 0.15,
                f"synthetic {subset} share {training:.3f} differs from real development "
                f"share {reference:.3f}")
    top_ten_share = sum(count for _, count in synthetic_templates.most_common(10)) / 170000
    require(top_ten_share <= 0.30,
            f"synthetic training repeats ten templates too often: {top_ten_share:.3f}")
    config = config_path.read_text()
    detector_section = re.search(r"(?ms)^\[detector\]\s*\n(.*?)(?=^\[|\Z)", config)
    require(detector_section is not None, "detector config has no detector section")
    def config_array(name):
        match = re.search(rf"(?m)^{name}\s*=\s*(\[[^\n]*\])", detector_section.group(1))
        require(match is not None, f"detector config has no {name} array")
        return json.loads(match.group(1))
    expected_silver = [f"data/interim/silver/{filename}.jsonl" for filename, _ in sources]
    require(config_array("silver") == expected_silver and
            len(config_array("silver_repeats")) == len(expected_silver),
            "detector config does not match the gated real training sources")
    print(f"US input integrity gate passed: {len(exclusions)} evaluation cases "
          f"({staff_manifest['cases'] + park_manifest['cases'] + ky_manifest['cases'] + ca_manifest['cases'] + or_manifest['cases'] + pa_manifest['cases'] + pa_acronym_manifest['cases'] + pa_staff_manifest['cases'] + len(load(contact_holdout)) + len(load(year_holdout)) + (32 if args.v3 else 0)} frozen), "
          f"{len(seen)} real training documents, "
          f"{sum(synthetic_counts.values())} synthetic model labels; "
          f"real labels {dict(real_counts)}; "
          f"synthetic acronym {synthetic_mix['acronym']}/{synthetic_mix['org']}, "
          f"address prefix {synthetic_mix['address_prefix']}/{synthetic_mix['address']}, "
          f"top ten templates {top_ten_share:.3f}")


if __name__ == "__main__":
    main()
