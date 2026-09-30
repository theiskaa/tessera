"""Remove an adjectival Army label that cannot align to a detector token."""

import collections
import json

from build_us_full_eval_exclusions import digest
from freeze_2024_2025_extra_long_org_train_candidates_v1 import MANIFEST as FIRST_MANIFEST
from freeze_2024_2025_extra_long_org_train_candidates_v1 import OUT as FIRST
from freeze_2024_2025_extra_long_org_train_candidates_v1 import ROOT
from freeze_2024_2025_extra_long_org_train_candidates_v1 import main as verify_first
from source_identity import document_key


OUT = ROOT / "data/interim/silver/r25/us-extra-long-org-candidate-v2.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
ERRATUM = {
    "name": "us-long-org-prose-2025-01757-525",
    "kind": "org",
    "start": 1077,
    "end": 1081,
    "text": "Army",
    "context": "Army-directed",
    "reason": "Army is adjectival inside one tokenizer word, not an independently decodable organization mention.",
}


def main():
    """Pin the single reviewed label correction before silver promotion."""
    rows = verify_first()
    corrected = []
    removed = 0
    for row in rows:
        row = {**row, "expected": list(row["expected"])}
        if row["name"] == ERRATUM["name"]:
            source = row["input"].encode()
            if source[ERRATUM["start"]:
                      ERRATUM["start"] + len(ERRATUM["context"])].decode() != ERRATUM["context"]:
                raise ValueError("Army adjective source context changed")
            matches = [span for span in row["expected"] if all(
                span[key] == ERRATUM[key] for key in ("kind", "start", "end", "text"))]
            if len(matches) != 1:
                raise ValueError("Army adjective label changed")
            row["expected"].remove(matches[0])
            removed += 1
        corrected.append(row)
    if len(corrected) != 29 or removed != 1:
        raise ValueError("extra-long candidate erratum membership changed")
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in corrected).encode()
    manifest = {
        "kind": "us_extra_long_org_train_candidate_v2",
        "training_eligible": False,
        "cases": len(corrected),
        "source_documents": len({document_key(row) for row in corrected}),
        "entity_counts": dict(sorted(collections.Counter(
            span["kind"] for row in corrected for span in row["expected"]).items())),
        "erratum": ERRATUM,
        "input_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (FIRST, FIRST_MANIFEST)
        },
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("corrected candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("corrected candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("corrected candidate inputs changed")
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(corrected)} extra-long candidates frozen with one token-boundary correction")
    return corrected


if __name__ == "__main__":
    main()
