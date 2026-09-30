"""Freeze independently reviewed official US mail code contact examples."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons, validate_spans
from check_us_next_data_split import EVALUATION, rows


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/raw/candidates/us-official-mail-code-v1/blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
REVIEW = ROOT / "data/interim/review"
REVIEWS = tuple(REVIEW / f"us-official-mail-code-review-{pass_id}-v1.jsonl"
                for pass_id in "abc")
NOTES = (
    REVIEW / "us-official-mail-code-review-a-v1-notes.md",
    REVIEW / "us-official-mail-code-review-b-v1.uncertainties.json",
    REVIEW / "us-official-mail-code-review-c-v1.notes.md",
)
OUT = ROOT / "data/interim/silver/us-official-mail-code-reviewed-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def digest(path):
    """Hash a frozen packet, review, or checker input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_key(row):
    """Compare entire labeled rows, including address boundaries."""
    return tuple(sorted((span["kind"], span["start"], span["end"])
                        for span in row["expected"]))


def silver_rows():
    """Resolve the one disputed page and reject any reserved overlap."""
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET) or packet_manifest["cases"] != 4 or
            packet_manifest["training_eligible"] or packet_manifest["evaluation_eligible"]):
        raise ValueError("official mail code blind packet changed")
    packet = rows(PACKET)
    by_name = {row["name"]: row for row in packet}
    if len(by_name) != 4:
        raise ValueError("official mail code packet has duplicate cases")
    review_maps = []
    for index, path in enumerate(REVIEWS):
        review = rows(path)
        mapped = {row["name"]: row for row in review}
        expected = (set(by_name) if index < 2 else
                    {"us-official-mail-code-nasa-goddard-abshire"})
        if len(mapped) != len(review) or set(mapped) != expected:
            raise ValueError(f"incomplete independent review: {path.name}")
        for name, row in mapped.items():
            source = by_name[name]
            if {key: value for key, value in row.items() if key != "expected"} != source:
                raise ValueError(f"review changed blind page: {name}")
            validate_spans(row, row["expected"])
        review_maps.append(mapped)
    first, second, third = review_maps
    selected = []
    for source in packet:
        name = source["name"]
        if label_key(first[name]) == label_key(second[name]):
            if name in third:
                raise ValueError("third review included an agreed page")
            selected.append(first[name])
        elif name in third and label_key(third[name]) == label_key(second[name]):
            selected.append(second[name])
        else:
            raise ValueError(f"mail code review lacks full-row agreement: {name}")
    gold = [row for path in EVALUATION for row in rows(path)]
    heldout = heldout_reasons(selected, gold)
    if heldout:
        raise ValueError(f"mail code review overlaps evaluation: {heldout}")
    silver = []
    for row in selected:
        metadata = {key: value for key, value in row.items()
                    if key not in {"name", "input", "expected"}}
        silver.append({**metadata, "id": row["name"], "text": row["input"],
                       "entities": [{key: span[key] for key in ("kind", "start", "end")}
                                    for span in row["expected"]]})
    if len({row["source_document_key"] for row in silver}) != 4:
        raise ValueError("mail code reviewed pages repeat a source")
    return silver


def main():
    """Pin the reviewed examples without changing any training config."""
    silver = silver_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in silver).encode()
    inputs = (PACKET, PACKET_MANIFEST, *REVIEWS, *NOTES, *EVALUATION,
              ROOT / "bench/us/check_us_blind_packet_holdout.py",
              ROOT / "bench/us/org_aliases.py")
    manifest = {
        "kind": "us_official_mail_code_reviewed_v1",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": len(silver),
        "source_documents": len({row["source_document_key"] for row in silver}),
        "spans_by_kind": dict(sorted(collections.Counter(
            span["kind"] for row in silver for span in row["entities"]).items())),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("official mail code silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("official mail code silver changed")
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded_manifest:
        raise ValueError("official mail code silver inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
        MANIFEST.write_bytes(encoded_manifest)
    print(f"{len(silver)} official mail code passages frozen; "
          f"{manifest['spans_by_kind']}")


if __name__ == "__main__":
    main()
