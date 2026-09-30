"""Promote reviewed US notices with source-faithful long passage splits."""

import collections
import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_two_paragraph_org_train_candidates_v1 import MANIFEST as CANDIDATE_MANIFEST
from freeze_2024_2025_two_paragraph_org_train_candidates_v1 import OUT as CANDIDATE
from freeze_2024_2025_two_paragraph_org_train_candidates_v1 import ROOT, candidate_rows
from source_identity import document_key


OUT = ROOT / "data/interim/silver/us-reviewed-two-paragraph-org-prose-v3.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
SPLITS = {
    "2025-18831": " 10. Project Sponsor: Graymont (PA) Inc.",
    "2025-21332": " Section 1905(y) of the Act, as added by section 2001(a)(3)",
}


def silver_piece(candidate, key, start, end, part):
    """Rebase complete reviewed spans to a source-contiguous piece."""
    spans = candidate["expected"]
    source = candidate["input"].encode()
    if any(span["start"] < end and span["end"] > start and not
           (start <= span["start"] and span["end"] <= end) for span in spans):
        raise ValueError(f"split crosses a reviewed label: {candidate['name']}")
    return {
        "id": f"{candidate['name']}-part-{part}" if part else candidate["name"],
        "country": "US",
        "source": "federal-register",
        "source_id": candidate["source_id"],
        "source_url": candidate["source_url"],
        "source_group": candidate["source_group"],
        "source_document_key": key,
        "raw_sha256": candidate["raw_sha256"],
        "text": source[start:end].decode(),
        "entities": [{"kind": span["kind"], "start": span["start"] - start,
                      "end": span["end"] - start}
                     for span in spans if start <= span["start"] and span["end"] <= end],
    }


def main():
    """Freeze eligible silver derived from the 16 agreed source passages."""
    candidates, _, _ = candidate_rows()
    frozen = [json.loads(line) for line in CANDIDATE.read_text().splitlines()]
    if (len(candidates) != 16 or candidates != frozen or
            json.loads(CANDIDATE_MANIFEST.read_text())["training_eligible"] is not False):
        raise ValueError("two-paragraph reviewed candidate changed")
    rows = []
    split_sources = []
    for candidate in candidates:
        key = document_key(candidate)
        if key is None:
            raise ValueError(f"source identity missing: {candidate['name']}")
        source = candidate["input"].encode()
        if candidate["source_id"] in SPLITS:
            boundary = SPLITS[candidate["source_id"]].encode()
            if source.count(boundary) != 1:
                raise ValueError(f"reviewed source boundary changed: {candidate['name']}")
            cut = source.index(boundary)
            chunks = ((0, cut, 1), (cut, len(source), 2))
            split_sources.append(candidate["source_id"])
        else:
            chunks = ((0, len(source), None),)
        pieces = [silver_piece(candidate, key, start, end, part)
                  for start, end, part in chunks]
        if b"".join(piece["text"].encode() for piece in pieces) != source:
            raise ValueError(f"source text lost in split: {candidate['name']}")
        original = sorted((span["kind"], span["start"], span["end"])
                          for span in candidate["expected"])
        rebuilt = sorted((span["kind"], span["start"] + start,
                          span["end"] + start)
                         for (start, _, _), piece in zip(chunks, pieces)
                         for span in piece["entities"])
        if original != rebuilt:
            raise ValueError(f"reviewed labels lost in split: {candidate['name']}")
        rows.extend(pieces)
    if split_sources != list(SPLITS):
        raise ValueError("expected long sources were not split exactly once")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    manifest = {
        "kind": "us_reviewed_two_paragraph_org_prose_v3",
        "training_eligible": True,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "split_boundaries": SPLITS,
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in rows for span in row["entities"]).items())),
        "candidate_sha256": digest(CANDIDATE.read_bytes()),
        "candidate_manifest_sha256": digest(CANDIDATE_MANIFEST.read_bytes()),
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("two-paragraph silver or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("two-paragraph silver changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("two-paragraph silver inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} reviewed pieces from 16 US passages are training eligible")


if __name__ == "__main__":
    main()
