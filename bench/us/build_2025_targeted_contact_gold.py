"""Reconcile five new contact layouts from two independent source reviews."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from build_silver import family
from freeze_2025_contact_packet import RAW
from freeze_2025_targeted_contact_packet import OUT as PACKET


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-2025-targeted-contact-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-2025-targeted-contact-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2025-targeted-contact-adjudication-v1.json"
OUT = REVIEW / "us-2025-targeted-contact-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"targeted contact input changed: {path.name}")
    packet_manifest = json.loads(PACKET.with_suffix(".manifest.json").read_text())
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"]):
        raise ValueError("targeted contact packet changed")
    packet, left, right = map(read_rows, (PACKET, REVIEW_A, REVIEW_B))
    names = [row["name"] for row in packet]
    if (len(set(names)) != len(names) or
            names != [row["name"] for row in left] or
            names != [row["name"] for row in right]):
        raise ValueError("targeted contact blind review order differs")
    retained = set(adjudication["retained_source_ids"])
    if len(retained) != len(adjudication["retained_source_ids"]) or len(retained) != 5:
        raise ValueError("targeted contact retained source list changed")
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate targeted contact adjudication")
    disputed = set()
    rows = []
    for source, a, b in zip(packet, left, right):
        if source["source_id"] not in retained:
            continue
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"targeted contact differs from source: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if (a_spans != b_spans or a.get("uncertain") or b.get("uncertain") or
                a.get("uncertain_reason") or b.get("uncertain_reason")):
            disputed.add(name)
            choice = choices.get(name)
            if choice is None or choice.get("take") not in {"a", "b"} or not choice.get("reason"):
                raise ValueError(f"unadjudicated targeted contact: {name}")
            spans = a_spans if choice["take"] == "a" else b_spans
        else:
            spans = a_spans
        rows.append({"name": name, "country": "US", "doc_type": "notice_contact",
                     "input": source["input"], "expected": spans,
                     "source_id": source["source_id"], "source_url": source["source_url"],
                     "source_group": family(source["source_url"])})
    if len(rows) != len(retained) or set(choices) != disputed:
        raise ValueError("targeted contact selection or adjudications changed")
    return rows, len(disputed)


def main():
    rows, adjudications = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_targeted_contact_gold_v1",
        "cases": len(rows), "adjudications": adjudications,
        "entity_counts": dict(sorted(counts.items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data), "source_group_key": "source_group",
        "training_eligible": False,
        "intended_use": "reviewed_candidate_pending_split_audit",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("targeted contact gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("targeted contact gold changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("targeted contact gold inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} new targeted contacts reconciled, {adjudications} adjudications")


if __name__ == "__main__":
    main()
