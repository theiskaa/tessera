"""Freeze long US person prose with complete independent label agreement."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons, validate_spans
from check_us_next_data_split import EVALUATION, rows
from source_identity import document_key


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/raw/candidates/us-person-long-v3/blind-v3.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
REVIEW = ROOT / "data/interim/review"
REVIEWS = tuple(REVIEW / f"us-person-long-v3-review-{name}.jsonl" for name in "abc")
NOTES = REVIEW / "us-person-long-v3-review-c-uncertainties.md"
OUT = ROOT / "data/interim/silver/us-person-long-reviewed-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
UNRESOLVED = {"berkeley-keltner"}


def digest(path):
    """Hash the exact frozen input file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_key(row):
    """Compare all reviewed spans and their boundaries."""
    return tuple(sorted((span["kind"], span["start"], span["end"], span["text"])
                        for span in row["expected"]))


def read_review(path, packet, expected_ids):
    """Require complete labels without any change to the blind source."""
    reviewed = rows(path)
    mapped = {row["source_id"]: row for row in reviewed}
    if len(mapped) != len(reviewed) or set(mapped) != expected_ids:
        raise ValueError(f"review membership changed: {path.name}")
    for source_id, row in mapped.items():
        if {key: value for key, value in row.items() if key != "expected"} != packet[source_id]:
            raise ValueError(f"review changed blind source: {source_id}")
        validate_spans(row, row["expected"])
    return mapped


def silver_rows():
    """Keep only source-clean rows with complete review consensus."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET) or packet_manifest["cases"] != 5 or
            packet_manifest["training_eligible"] or packet_manifest["evaluation_eligible"]):
        raise ValueError("person v3 blind packet changed")
    packet_rows = rows(PACKET)
    packet = {row["source_id"]: row for row in packet_rows}
    if len(packet) != 5:
        raise ValueError("person v3 packet repeats a source")
    first = read_review(REVIEWS[0], packet, set(packet))
    second = read_review(REVIEWS[1], packet, set(packet))
    third = read_review(REVIEWS[2], packet, UNRESOLVED)
    selected = []
    disputed = set()
    for source in packet_rows:
        source_id = source["source_id"]
        if label_key(first[source_id]) == label_key(second[source_id]):
            selected.append(first[source_id])
        elif (source_id in third and
              label_key(third[source_id]) != label_key(first[source_id]) and
              label_key(third[source_id]) != label_key(second[source_id])):
            disputed.add(source_id)
        else:
            raise ValueError(f"unexpected person v3 review resolution: {source_id}")
    if disputed != UNRESOLVED or len(selected) != 4:
        raise ValueError(f"person v3 review resolution changed: {disputed}")
    gold = [row for path in EVALUATION for row in rows(path)]
    heldout = heldout_reasons(selected, gold)
    if heldout:
        raise ValueError(f"person v3 labels overlap evaluation: {heldout}")
    silver = []
    for row in selected:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "entities": [{key: span[key] for key in ("kind", "start", "end")}
                                    for span in row["expected"]]})
    if (len({document_key(row) for row in silver}) != 4 or
            sum(span["kind"] == "person" for row in silver
                for span in row["entities"]) != 27):
        raise ValueError("person v3 distinct-source supervision changed")
    return silver, disputed


def main():
    """Pin reviewed labels without activating a training run."""
    silver, disputed = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in silver).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS, NOTES, *EVALUATION,
              ROOT / "bench/us/check_us_blind_packet_holdout.py")
    manifest = {
        "kind": "us_person_long_reviewed_v3",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(silver),
        "source_documents": len({document_key(row) for row in silver}),
        "spans_by_kind": dict(sorted(collections.Counter(
            span["kind"] for row in silver for span in row["entities"]).items())),
        "unresolved_labels_excluded": sorted(disputed),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("person v3 silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("person v3 silver changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("person v3 silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(silver)} long person passages frozen; "
          f"{manifest['spans_by_kind']}")


if __name__ == "__main__":
    main()
