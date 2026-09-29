"""Freeze source-backed 2026 US contacts after independent review and adjudication."""

import collections
import json
from pathlib import Path

from build_full_notice_gold import digest, read_rows, validate_spans


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "data/raw/candidates/federal-register-2026-contacts-v1"
PACKET = CAPTURE / "screened-contact-blind-v1.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
REVIEW = ROOT / "data/interim/review"
REVIEW_A = REVIEW / "us-2026-contact-review-a-v1.jsonl"
REVIEW_B = REVIEW / "us-2026-contact-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-2026-contact-adjudication-v1.json"
OUT = REVIEW / "us-2026-contact-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def gold_rows():
    """Validate every frozen input and resolve only recorded blind-review disputes."""
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"2026 contact review input changed: {path.name}")
    packet_manifest = json.loads(PACKET_MANIFEST.read_text())
    if (packet_manifest["sha256"] != digest(PACKET.read_bytes()) or
            packet_manifest["training_eligible"] or
            packet_manifest["source_capture_sha256"] !=
            digest((CAPTURE / "manifest.json").read_bytes())):
        raise ValueError("2026 contact packet provenance changed")
    capture = json.loads((CAPTURE / "manifest.json").read_text())
    sources = {row["source_id"]: row for row in capture["sources"]}
    if len(sources) != len(capture["sources"]) or len(sources) != 2000:
        raise ValueError("2026 contact source capture changed")
    packet, left, right = map(read_rows, (PACKET, REVIEW_A, REVIEW_B))
    names = [row["name"] for row in packet]
    if (len(names) != 40 or len(set(names)) != len(names) or
            names != [row["name"] for row in left] or
            names != [row["name"] for row in right]):
        raise ValueError("2026 contact blind review order differs")
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate 2026 contact adjudication")
    disputed = set()
    result = []
    for source, a, b in zip(packet, left, right):
        name = source["name"]
        original = sources[source["source_id"]]
        raw_bytes = (CAPTURE / "sources" / f"{source['source_id']}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if (digest(raw_bytes) != source["raw_sha256"] or
                original["raw_sha256"] != source["raw_sha256"] or
                original["source_url"] != source["source_url"] or
                original["source_group"] != source["source_group"] or
                raw["id"] != source["source_id"] or
                raw["url"] != source["source_url"] or
                not raw["date"].startswith(source["month"]) or
                raw["text"].count(source["input"]) != 1):
            raise ValueError(f"2026 contact differs from frozen source: {name}")
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if (a_spans != b_spans or a.get("uncertain") or b.get("uncertain") or
                a.get("uncertain_reason") or b.get("uncertain_reason")):
            disputed.add(name)
            choice = choices.get(name)
            if choice is None or choice.get("take") not in {"a", "b"} or not choice.get("reason"):
                raise ValueError(f"unadjudicated 2026 contact: {name}")
            primary = a_spans if choice["take"] == "a" else b_spans
            other = b_spans if choice["take"] == "a" else a_spans
            spans = list(primary)
            for removal in choice.get("remove", []):
                if removal not in spans:
                    raise ValueError(f"adjudication removal absent: {name}")
                spans.remove(removal)
            for addition in choice.get("add", []):
                if addition not in other or addition in spans:
                    raise ValueError(f"adjudication addition lacks independent review: {name}")
                spans.append(addition)
            spans = validate_spans(name, source["input"], spans)
        else:
            spans = a_spans
        result.append({
            "name": name, "country": "US", "doc_type": "notice_contact",
            "input": source["input"], "expected": spans,
            "source_id": source["source_id"], "source_url": source["source_url"],
            "source_group": source["source_group"], "month": source["month"],
            "agency": source["agency"],
        })
    if set(choices) != disputed:
        raise ValueError("2026 adjudications differ from disputed cases")
    return result, len(disputed)


def main():
    """Write reproducible gold and its input hashes once."""
    rows, adjudications = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows for span in row["expected"])
    manifest = {
        "kind": "us_2026_contact_gold_v1",
        "cases": len(rows), "adjudications": adjudications,
        "entity_counts": dict(sorted(counts.items())),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "source_capture_sha256": digest((CAPTURE / "manifest.json").read_bytes()),
        "sha256": digest(data), "source_group_key": "source_id",
        "training_eligible": False,
        "intended_use": "reviewed_candidate_pending_split_audit",
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("2026 contact gold or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("2026 contact gold changed after freezing")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("2026 contact gold inputs changed after freezing")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} source-backed 2026 contacts reconciled, {adjudications} adjudications")


if __name__ == "__main__":
    main()
