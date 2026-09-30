"""Create reviewable PDF-style line wraps from agreed US silver spans."""

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "data/interim/silver/us-v5-reviewed-long-org-prose-v1.jsonl",
    ROOT / "data/interim/silver/us-reviewed-extra-long-org-prose-v2.jsonl",
    ROOT / "data/interim/silver/us-reviewed-two-paragraph-org-prose-v3.jsonl",
)
OUT = ROOT / "data/interim/silver/r25/us-pdf-line-wrap-candidate-v1.jsonl"
MANIFEST = OUT.with_suffix(".manifest.json")
WIDTHS = tuple(range(54, 99, 4))


def digest(data):
    """Hash the exact source or frozen output bytes."""
    return hashlib.sha256(data).hexdigest()


def wrapped(row, width):
    """Replace source spaces with newlines without shifting any UTF-8 span offset."""
    source = row["text"].encode()
    data = bytearray(source)
    protected = tuple((span["start"], span["end"])
                      for span in row["entities"] if span["kind"] != "org")
    cuts = []
    line_start = 0
    while line_start < len(data):
        line_end = data.find(10, line_start)
        if line_end < 0:
            line_end = len(data)
        while line_end - line_start > width:
            near = range(line_start + max(1, width - 16),
                         min(line_end, line_start + width) + 1)
            spaces = [index for index in near if data[index] == 32 and
                      not any(start <= index < end for start, end in protected)]
            if not spaces:
                after = range(line_start + width + 1,
                              min(line_end, line_start + width + 17))
                spaces = [index for index in after if data[index] == 32 and
                          not any(start <= index < end for start, end in protected)]
            if not spaces:
                break
            cut = spaces[-1] if spaces[-1] <= line_start + width else spaces[0]
            data[cut] = 10
            cuts.append(cut)
            line_start = cut + 1
        line_start = line_end + 1
    if any(source[index] != 32 for index in cuts):
        raise ValueError(f"wrap changed a non-space byte: {row['id']}")
    restored = bytearray(data)
    for index in cuts:
        restored[index] = 32
    if bytes(restored) != source:
        raise ValueError(f"wrap failed source reconstruction: {row['id']}")
    transformed = bytes(data).decode()
    org_breaks = sum(any(span["start"] <= index < span["end"] for index in cuts)
                     for span in row["entities"] if span["kind"] == "org")
    return transformed, cuts, org_breaks


def candidate_rows():
    """Choose one reproducible wrap per source piece with an organization line break."""
    sources = [json.loads(line) for path in SOURCES
               for line in path.read_text().splitlines() if line.strip()]
    if len(sources) != 82:
        raise ValueError("reviewed long silver membership changed")
    rows = []
    for source in sources:
        first = int(digest(source["id"].encode()), 16) % len(WIDTHS)
        widths = WIDTHS[first:] + WIDTHS[:first]
        for width in widths:
            text, cuts, org_breaks = wrapped(source, width)
            if org_breaks:
                rows.append({**source,
                             "id": f"{source['id']}-pdf-wrap-v1",
                             "text": text,
                             "augmented_from_id": source["id"],
                             "wrap_width": width,
                             "changed_space_byte_offsets": cuts})
                break
    if len(rows) != 51 or sum(
            sum("\n" in row["text"].encode()[span["start"]:span["end"]].decode()
                for span in row["entities"] if span["kind"] == "org")
            for row in rows) != 78:
        raise ValueError("PDF-style organization line-break coverage changed")
    return rows


def main():
    """Freeze a technical candidate without enabling it for model training."""
    rows = candidate_rows()
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode()
    counts = collections.Counter(span["kind"] for row in rows
                                 for span in row["entities"])
    manifest = {
        "kind": "us_pdf_line_wrap_candidate_v1",
        "training_eligible": False,
        "cases": len(rows),
        "source_documents": len({row["source_document_key"] for row in rows}),
        "org_line_break_spans": 78,
        "entity_counts": dict(sorted(counts.items())),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                          for path in SOURCES},
        "sha256": digest(data),
    }
    if OUT.exists() != MANIFEST.exists():
        raise ValueError("PDF-style candidate or manifest is missing")
    if OUT.exists() and OUT.read_bytes() != data:
        raise ValueError("PDF-style candidate changed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != manifest:
        raise ValueError("PDF-style candidate inputs changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(data)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} PDF-style US pieces frozen with 78 line-broken organizations")


if __name__ == "__main__":
    main()
