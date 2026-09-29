"""Adjudicate two blind reviews of source-backed 2025 contact passages."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans
from build_silver import family
from freeze_2025_ready_contact_packet import OUT as PACKET
from freeze_2025_ready_contact_packet import MANIFEST as PACKET_MANIFEST
from freeze_2025_contact_packet import RAW


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-2025-ready-contact-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-2025-ready-contact-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2025-ready-contact-adjudication-v1.json"
OUT = REVIEW / "us-2025-ready-contact-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"blind input changed: {path.name}")
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != adjudication["packet_sha256"] or
            packet_manifest["training_eligible"]):
        raise ValueError("ready contact packet manifest changed")
    packet, review_a, review_b = map(read_rows, (PACKET, REVIEW_A, REVIEW_B))
    names = [row["name"] for row in packet]
    if (len(names) != len(set(names)) or
            names != [row["name"] for row in review_a] or
            names != [row["name"] for row in review_b]):
        raise ValueError("blind reviews do not match the frozen packet order")
    choices = {row["name"]: row for row in adjudication["decisions"]}
    audited = {row["name"]: row for row in adjudication["agreed_audits"]}
    if (len(choices) != len(adjudication["decisions"]) or
            len(audited) != len(adjudication["agreed_audits"])):
        raise ValueError("duplicate adjudication or agreement audit")
    disputed = set()
    agreed = set()
    result = []
    for source, a, b in zip(packet, review_a, review_b):
        name = source["name"]
        raw_bytes = (RAW / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                not raw["date"].startswith("2025-") or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"contact case differs from raw source: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a_spans != b_spans:
            disputed.add(name)
            choice = choices.get(name)
            if choice is None or choice.get("take") not in {"a", "b"} or not choice.get("reason"):
                raise ValueError(f"unadjudicated blind review: {name}")
            spans = a_spans if choice["take"] == "a" else b_spans
        else:
            agreed.add(name)
            spans = a_spans
        result.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"], "source_url": source["source_url"],
            "source_group": family(source["source_url"]),
        })
    if set(choices) != disputed or not set(audited) <= agreed:
        raise ValueError("adjudications differ from disputed or agreed cases")
    if (not set(adjudication["training_excluded"]) <= set(names) or
            len(adjudication["training_excluded"]) != len(set(adjudication["training_excluded"]))):
        raise ValueError("invalid training exclusion list")
    if any(not row.get("reason") for row in audited.values()):
        raise ValueError("agreement audit lacks reason")
    return result, len(disputed), adjudication["training_excluded"]


def main():
    rows, adjudications, training_excluded = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2025_ready_contact_gold_v1",
        "cases": len(rows), "adjudications": adjudications,
        "entity_counts": dict(sorted(counts.items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_group",
        "training_eligible": False,
        "strict_entity_holdout": False,
        "training_excluded": training_excluded,
        "intended_use": "split_candidate_pending_overlap_audit",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("frozen contact gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2025 ready contact gold changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2025 ready contact gold manifest changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reconciled 2025 contacts, {adjudications} adjudications: {digest(data)}")


if __name__ == "__main__":
    main()
