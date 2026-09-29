"""Remove independently reviewed duplicates and weak 2024 contact cases."""

import json
from pathlib import Path

from build_contact_snippets import lines
from freeze_2024_contact_packet import OUT as CANDIDATE, digest


ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / "bench/us/fixtures/us-2024-contact-triage-v1.json"
OUT = ROOT / "data/interim/silver/r24/us-2024-contact-blind-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")


def main():
    source = CANDIDATE.read_bytes()
    rows = lines(CANDIDATE)
    decisions = json.loads(DECISIONS.read_text())
    dropped = decisions["drop"]
    if (set(dropped) - {row["source_id"] for row in rows} or
            len(rows) != len({row["source_id"] for row in rows}) or
            any(not reason for reason in dropped.values())):
        raise ValueError("2024 contact triage differs from the frozen candidate packet")
    kept = [row for row in rows if row["source_id"] not in dropped]
    data = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                      for row in kept) + "\n").encode()
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("frozen 2024 contact review packet changed")
    manifest = {
        "kind": "us_2024_contact_blind_review_packet",
        "intended_use": "training_candidate_after_two_blind_reviews",
        "training_eligible": False,
        "label_status": "unlabeled",
        "source_group_key": "source_id",
        "source_scope": "complete_contact_or_addresses_paragraph_only",
        "candidate_sha256": digest(source),
        "triage_sha256": digest(DECISIONS.read_bytes()),
        "cases": len(kept),
        "dropped": len(dropped),
        "sha256": digest(data),
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("frozen 2024 contact review manifest changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(kept)} blind 2024 contact cases after source quality review")


if __name__ == "__main__":
    main()
