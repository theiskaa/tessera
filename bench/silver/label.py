"""Label person, organization, and address spans with GLiNER; write byte-offset spans.

Silver labels for detector training: real text labelled by a larger teacher model. Emails
and phone numbers are left to the rules layer. Run from the repository root:

    python bench/silver/label.py SOURCE_DIR OUT_JSONL [threshold]

and a manifest `data/manifests/silver-<source>.json` is written next to the labels.
"""

import datetime
import hashlib
import json
import pathlib
import sys

TEACHER = "urchade/gliner_multi-v2.1"
LABELS = ["person", "organization", "address"]
KIND = {"person": "person", "organization": "org", "address": "address"}

# GLiNER reads about 384 words and silently drops the rest, so windows stay near 1,600
# characters: a new one starts every STRIDE characters (at a space) and reaches OVERLAP past
# the next start. Each window owns the entities starting in its middle, so every character
# is owned once and every owned entity has at least OVERLAP / 2 characters of context.
STRIDE = 1200
OVERLAP = 300

SOURCES = {
    "federal-register": {
        "url": "https://www.federalregister.gov/api/v1/documents.json (list); https://www.govinfo.gov/content/pkg/FR-<date>/html/<number>.htm (text)",
        "license": "public-domain-us-government",
        "attribution": "Federal Register, Office of the Federal Register and the Government Publishing Office",
    },
    "govuk": {
        "url": "https://www.gov.uk/api/search.json (list); https://www.gov.uk/api/content<link> (text)",
        "license": "OGL-3.0",
        "attribution": "Contains public sector information licensed under the Open Government Licence v3.0.",
    },
}


def to_bytes(text, i):
    """The UTF-8 byte offset of code-point index `i`."""
    return len(text[:i].encode("utf-8"))


def windows(text):
    """(start, window, own_from, own_to) in code points; owned ranges tile the text."""
    starts = [0]
    while starts[-1] + STRIDE < len(text):
        nxt = starts[-1] + STRIDE
        space = text.find(" ", nxt, nxt + 100)
        starts.append(space + 1 if space >= 0 else nxt)
    out = []
    for i, start in enumerate(starts):
        last = i + 1 == len(starts)
        end = len(text) if last else starts[i + 1] + OVERLAP
        own_from = start + OVERLAP // 2 if i else 0
        own_to = len(text) if last else starts[i + 1] + OVERLAP // 2
        out.append((start, text[start:end], own_from, own_to))
    return out


def merge(entities):
    """Sorted by start, longer first on ties; a span overlapping a kept one is dropped."""
    out = []
    for e in sorted(entities, key=lambda e: (e["start"], -e["end"])):
        if out and e["start"] < out[-1]["end"]:
            continue
        out.append(e)
    return out


def plausible(value):
    """Drops spans that cannot be a name or an address: no letter at all (phone numbers the
    teacher calls addresses), or Latin letters that are all lowercase ("government")."""
    letters = [c for c in value if c.isalpha()]
    if not letters:
        return False
    latin = all(c.isascii() or "\u00c0" <= c <= "\u024f" for c in letters)
    return not (latin and value == value.lower())


def label(model, text, threshold):
    found = []
    for start, window, own_from, own_to in windows(text):
        for e in model.predict_entities(window, LABELS, threshold=threshold):
            s, t = start + e["start"], start + e["end"]
            if not own_from <= s < own_to or not plausible(text[s:t]):
                continue
            found.append({"kind": KIND[e["label"]], "start": to_bytes(text, s),
                          "end": to_bytes(text, t), "score": round(float(e["score"]), 4)})
    return merge(found)


def main(src, out, threshold=0.5):
    import torch
    from gliner import GLiNER

    model = GLiNER.from_pretrained(TEACHER)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    src, out = pathlib.Path(src), pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*.json"))
    per_kind = {k: 0 for k in KIND.values()}
    source = None
    with out.open("w") as f:
        for n, path in enumerate(paths, 1):
            doc = json.loads(path.read_text())
            source = doc["source"]
            ents = label(model, doc["text"], threshold)
            for e in ents:
                per_kind[e["kind"]] += 1
            f.write(json.dumps({"id": doc["id"], "source": source, "url": doc["url"],
                                "date": doc.get("date", ""), "text": doc["text"],
                                "entities": ents}, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"{source}: {n}/{len(paths)} labelled", flush=True)
    if source is None:
        sys.exit(f"no documents in {src}")
    manifest = {
        "source": f"silver-{source}",
        "kind": "silver",
        "use": "detector training (train 90%, valid 10%); never test, never versioned",
        **SOURCES[source],
        "date_range": "2024",
        "documents": len(paths),
        "labels": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "teacher": TEACHER,
        "threshold": threshold,
        "window_stride_chars": STRIDE,
        "window_overlap_chars": OVERLAP,
        "filters": "overlapping spans keep the earlier; spans with no letter or with only lowercase Latin letters are dropped",
        "entities_per_kind": per_kind,
        "labelled": datetime.date.today().isoformat(),
    }
    path = pathlib.Path("data/manifests") / f"silver-{source}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{source}: {len(paths)} documents, {per_kind}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
