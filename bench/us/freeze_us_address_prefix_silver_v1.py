"""Freeze independently agreed US address passages outside reserved evaluation."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/raw/candidates/federal-register-us-address-prefix-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
REVIEW = ROOT / "data/interim/review"
REVIEWS = tuple(REVIEW / f"us-address-prefix-review-{pass_id}-v1.jsonl"
                for pass_id in "abc")
NOTES = (
    REVIEW / "us-address-prefix-review-a-v1.uncertainties.json",
    REVIEW / "us-address-prefix-review-b-v1-uncertainty.md",
    REVIEW / "us-address-prefix-review-c-v1-notes.json",
)
OUT = ROOT / "data/interim/silver/us-address-prefix-reviewed-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
EXPECTED_UNRESOLVED = {"2025-22532", "2026-00122", "2026-06068"}
EXPECTED_HELDOUT = {"2024-26393", "2026-05037", "2026-16399"}


def digest(path):
    """Hash the exact bytes of a frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    """Load frozen JSONL without reordering it."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def labels(review, packet, *, full_fields):
    """Validate a review against the frozen packet and return its complete spans."""
    spans = review["expected"] if full_fields else review["entities"]
    if full_fields:
        if {key: value for key, value in review.items() if key != "expected"} != packet:
            raise ValueError(f"review changed packet metadata: {packet['name']}")
    elif {key: review[key] for key in ("name", "input", "source_id")} != {
            key: packet[key] for key in ("name", "input", "source_id")}:
        raise ValueError(f"review changed packet text: {packet['name']}")
    source = packet["input"].encode()
    ordered = sorted(spans, key=lambda span: span["start"])
    previous = 0
    for span in ordered:
        if (span["kind"] not in {"person", "org", "address", "email", "phone"} or
                not previous <= span["start"] < span["end"] <= len(source) or
                source[span["start"]:span["end"]].decode() != span["text"]):
            raise ValueError(f"invalid review span: {packet['name']}")
        previous = span["end"]
    return tuple((span["kind"], span["start"], span["end"], span["text"])
                 for span in ordered)


def silver_rows():
    """Require full-row agreement, then remove reserved-source label overlap."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET) or packet_manifest["cases"] != 21 or
            packet_manifest["training_eligible"] or packet_manifest["evaluation_eligible"]):
        raise ValueError("blind address packet changed")
    packet = read_rows(PACKET)
    by_name = {row["name"]: row for row in packet}
    if len(by_name) != 21:
        raise ValueError("duplicate or missing address packet case")
    review_maps = []
    for index, path in enumerate(REVIEWS):
        review = read_rows(path)
        mapped = {row["name"]: row for row in review}
        if len(mapped) != len(review) or (index < 2 and set(mapped) != set(by_name)):
            raise ValueError(f"incomplete independent review: {path.name}")
        if index == 2 and not set(mapped) <= set(by_name):
            raise ValueError("third review contains an unknown case")
        for name, row in mapped.items():
            labels(row, by_name[name], full_fields=index > 0)
        review_maps.append(mapped)
    first, second, third = review_maps
    disputed = {name for name, source in by_name.items()
                if labels(first[name], source, full_fields=False) !=
                labels(second[name], source, full_fields=True)}
    if set(third) != disputed:
        raise ValueError("third review must cover exactly the disputed cases")
    agreed = []
    unresolved = set()
    for source in packet:
        name = source["name"]
        a = labels(first[name], source, full_fields=False)
        b = labels(second[name], source, full_fields=True)
        if a == b:
            chosen = second[name]
        else:
            c = labels(third[name], source, full_fields=True)
            if c == a:
                chosen = {**source, "expected": first[name]["entities"]}
            elif c == b:
                chosen = second[name]
            else:
                unresolved.add(source["source_id"])
                continue
        agreed.append(chosen)
    if unresolved != EXPECTED_UNRESOLVED or len(agreed) != 18:
        raise ValueError("address review resolution changed")
    gold = [row for path in EVALUATION for row in read_rows(path)]
    heldout = heldout_reasons(agreed, gold)
    heldout_ids = {row["source_id"] for row in agreed if row["name"] in heldout}
    if heldout_ids != EXPECTED_HELDOUT:
        raise ValueError(f"address heldout overlap changed: {heldout_ids}")
    kept = [row for row in agreed if row["name"] not in heldout]
    if (len(kept) != 15 or len({row["source_group"] for row in kept}) != 15 or
            any(sum(span["kind"] == "address" for span in row["expected"]) != 1
                for row in kept)):
        raise ValueError("address silver diversity or labels changed")
    silver = []
    for row in kept:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "source_document_key": f"fr:{row['source_id']}",
                       "entities": [{key: span[key] for key in ("kind", "start", "end")}
                                    for span in row["expected"]]})
    return silver, unresolved, heldout


def main():
    """Pin clean address labels without activating them in a training config."""
    silver, unresolved, heldout = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in silver).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS, *NOTES, *EVALUATION,
              ROOT / "bench/us/check_us_blind_packet_holdout.py",
              ROOT / "bench/us/org_aliases.py",
              ROOT / "bench/us/address_keys.py")
    manifest = {
        "kind": "us_address_prefix_reviewed_v1",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(silver),
        "source_documents": len({row["source_document_key"] for row in silver}),
        "source_groups": len({row["source_group"] for row in silver}),
        "spans_by_kind": dict(sorted(collections.Counter(
            span["kind"] for row in silver for span in row["entities"]).items())),
        "unresolved_excluded": sorted(unresolved),
        "heldout_overlap_excluded": heldout,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("address silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("address silver changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("address silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(silver)} reviewed address passages frozen; "
          f"{manifest['spans_by_kind']}")


if __name__ == "__main__":
    main()
