"""Build held-out US notice gold from two independent blind label passes."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "data/interim/silver/r23/us-full-notice-blind-v1.jsonl"
REVIEW_A = ROOT / "data/interim/review/us-full-notice-review-a-v1.jsonl"
REVIEW_B = ROOT / "data/interim/review/us-full-notice-review-b-v1.jsonl"
DECISIONS = ROOT / "bench/us/fixtures/us-full-notice-adjudication-v1.json"
OUT = ROOT / "data/interim/review/us-full-notice-gold-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
KINDS = {"person", "org", "address", "email", "phone"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_spans(name, text, spans):
    source = text.encode()
    ordered = sorted(spans, key=lambda span: (span["start"], span["end"]))
    for span in ordered:
        start, end = span["start"], span["end"]
        if (span["kind"] not in KINDS or start < 0 or start >= end or
                source[start:end].decode() != span["text"]):
            raise ValueError(f"invalid label in {name}: {span}")
    for left, right in zip(ordered, ordered[1:]):
        if left["end"] > right["start"]:
            raise ValueError(f"overlapping labels in {name}")
    return ordered


def gold_rows():
    adjudication = json.loads(DECISIONS.read_text())
    for path, key in ((PACKET, "packet_sha256"), (REVIEW_A, "review_a_sha256"),
                      (REVIEW_B, "review_b_sha256")):
        if digest(path.read_bytes()) != adjudication[key]:
            raise ValueError(f"blind input changed: {path.name}")
    packet, review_a, review_b = map(read_rows, (PACKET, REVIEW_A, REVIEW_B))
    packet_names = [row["name"] for row in packet]
    if (len(packet_names) != len(set(packet_names)) or
            [row["name"] for row in review_a] != packet_names or
            [row["name"] for row in review_b] != packet_names):
        raise ValueError("blind reviews do not match the frozen packet order")
    choices = {row["name"]: row for row in adjudication["decisions"]}
    if len(choices) != len(adjudication["decisions"]):
        raise ValueError("duplicate adjudication")
    needed = set()
    result = []
    for source, a, b in zip(packet, review_a, review_b):
        name = source["name"]
        a_spans = validate_spans(name, source["input"], a["entities"])
        b_spans = validate_spans(name, source["input"], b["entities"])
        if a_spans != b_spans or a.get("uncertain") or b.get("uncertain"):
            needed.add(name)
            if name not in choices:
                raise ValueError(f"unadjudicated blind review: {name}")
            decision = choices[name]
            if not decision.get("reason") or decision["take"] not in {"a", "b"}:
                raise ValueError(f"incomplete adjudication: {name}")
            spans = list(a_spans if decision["take"] == "a" else b_spans)
            for removal in decision.get("remove", []):
                if removal not in spans:
                    raise ValueError(f"adjudication removal absent: {name}")
                spans.remove(removal)
            spans = validate_spans(name, source["input"], spans)
        else:
            spans = a_spans
        result.append({"name": name, "country": "US", "doc_type": "notice",
                       "input": source["input"], "expected": spans,
                       "source_id": source["source_id"],
                       "source_url": source["source_url"]})
    if set(choices) != needed:
        raise ValueError("adjudications differ from disputed or uncertain cases")
    return result, adjudication


def main():
    rows, adjudication = gold_rows()
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("held-out notice gold changed after freezing")
    OUT.write_bytes(data)
    MANIFEST.write_text(json.dumps({
        "kind": "us_full_notice_gold",
        "cases": len(rows),
        "adjudications": len(adjudication["decisions"]),
        "packet_sha256": digest(PACKET.read_bytes()),
        "review_a_sha256": digest(REVIEW_A.read_bytes()),
        "review_b_sha256": digest(REVIEW_B.read_bytes()),
        "decisions_sha256": digest(DECISIONS.read_bytes()),
        "sha256": digest(data),
        "source_group_key": "source_id",
        "training_eligible": False,
        "strict_entity_holdout": False,
        "intended_use": "source_distinct_diagnostic_only",
    }, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} held-out full notices: {digest(data)}")


if __name__ == "__main__":
    main()
