"""Freeze a previously unresolved biography after two fresh policy-based reviews."""

import collections
import hashlib
import json
from pathlib import Path

from check_us_blind_packet_holdout import heldout_reasons
from check_us_next_data_split import EVALUATION, ROOT, rows
from freeze_us_person_long_v3_silver import digest, label_key, read_review


PACKET = ROOT / "data/raw/candidates/us-person-long-v3/blind-v3.jsonl"
PACKET_MANIFEST = PACKET.with_suffix(".manifest.json")
PACKET_SHA256 = "bcdb9e2ae27ca78d2535706b2ce24162360243e4cb0231b7396aaa4ed1696b25"
REVIEW = ROOT / "data/interim/review"
REVIEWS = tuple(REVIEW / f"us-person-keltner-policy-v1-review-{name}.jsonl" for name in "de")
NOTES = tuple(REVIEW / f"us-person-keltner-policy-v1-review-{name}-uncertain.json" for name in "de")
POLICY = ROOT / "internal/bench/review/GUIDELINES.md"
OUT = ROOT / "data/interim/silver/us-person-keltner-policy-reviewed-v1.jsonl"


def main():
    """Require complete independent agreement and preserve all original source bytes."""
    manifest = json.loads(PACKET_MANIFEST.read_text())
    if (digest(PACKET) != PACKET_SHA256 or manifest["sha256"] != PACKET_SHA256 or
            manifest["cases"] != 5 or manifest["training_eligible"] or
            manifest["evaluation_eligible"]):
        raise ValueError("Keltner source packet changed")
    for name, expected in manifest["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Keltner screening input changed: {name}")
    original = rows(PACKET)
    packet = {row["source_id"]: row for row in original}
    if len(packet) != 5 or len(original) != 5:
        raise ValueError("Keltner source packet repeats or omits cases")
    first, second = [read_review(path, packet, {"berkeley-keltner"}) for path in REVIEWS]
    selected = first["berkeley-keltner"]
    raw_source = ROOT / selected["source_file"]
    if digest(raw_source) != selected["raw_sha256"]:
        raise ValueError("Keltner captured source HTML changed")
    if label_key(selected) != label_key(second["berkeley-keltner"]):
        raise ValueError("Keltner policy reviews disagree")
    counts = dict(sorted(collections.Counter(span["kind"] for span in selected["expected"]).items()))
    if counts != {"org": 10, "person": 12} or len(selected["input"].split()) < 200:
        raise ValueError("Keltner reviewed supervision changed")
    hits = heldout_reasons([selected], [row for path in EVALUATION for row in rows(path)])
    if hits:
        raise ValueError(f"Keltner labels overlap evaluation: {hits}")
    metadata = {key: value for key, value in selected.items()
                if key not in {"name", "input", "expected", "training_eligible", "evaluation_eligible"}}
    silver = {**metadata, "id": selected["name"], "text": selected["input"],
              "entities": [{key: span[key] for key in ("kind", "start", "end")}
                           for span in selected["expected"]]}
    data = (json.dumps(silver, ensure_ascii=False, sort_keys=True) + "\n").encode()
    inputs = (PACKET, PACKET_MANIFEST, raw_source, *REVIEWS, *NOTES, POLICY, *EVALUATION,
              Path(__file__), ROOT / "bench/us/freeze_us_person_long_v3_silver.py",
              ROOT / "bench/us/check_us_blind_packet_holdout.py")
    output_manifest = {
        "kind": "us_person_keltner_policy_reviewed_v1",
        "training_eligible": True,
        "active_in_training_config": False,
        "cases": 1,
        "source_documents": 1,
        "spans_by_kind": counts,
        "resolution": {
            "person": "Names inside the award title are excluded; the later James reference names the human and is included. Surname references and names before possessives follow the written policy.",
            "org": "Pixar's attribution identifies the studio responsible for the films, rather than a company string embedded in a product name. Both fresh reviews include it. Named Berkeley references identify the university in this academic context.",
        },
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    encoded = (json.dumps(output_manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = OUT.with_suffix(".manifest.json")
    if OUT.exists() != manifest_path.exists():
        raise ValueError("Keltner silver or manifest is missing")
    if OUT.exists() and (OUT.read_bytes() != data or manifest_path.read_bytes() != encoded):
        raise ValueError("frozen Keltner labels or inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
        manifest_path.write_bytes(encoded)
    print("1 long biography frozen from two fresh reviews; 12 person and 10 organization spans")


if __name__ == "__main__":
    main()
