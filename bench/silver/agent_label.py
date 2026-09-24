"""Labels from model reviewers on real public documents, under internal/bench/review/GUIDELINES.md:
silver rounds for detector training (person, org, and address; emails and phones come from the
rules layer at training time) and evaluation rounds (all five kinds, exported as gold).

    python bench/silver/agent_label.py sample ROUND
    python bench/silver/agent_label.py slices ROUND
    python bench/silver/agent_label.py apply ROUND PASS
    python bench/silver/agent_label.py export ROUND PASS
    python bench/silver/agent_label.py gold ROUND PASS

`sample` draws documents into data/interim/silver/r<ROUND>/sample.jsonl. `slices` writes
slice-NN.md, ten documents each, for the labelling pass. A labeller writes
pass<P>/slice-NN.json:

    {"<doc id>": [{"kind": "person|org|address", "text": "exact text"}, ...], ...}

listing every distinct entity string once. An entry may add `"within": "longer text"` to label
the string only inside occurrences of that context, as a one- or two-character Japanese name
must (`{"kind": "person", "text": "中", "within": "中 国際食料情報特別分析官"}`). A checking pass reads pass<P>-view/slice-NN.md
(the documents with the labels inline) and writes pass<P+1>/slice-NN.json in the same shape,
the full corrected list. `apply` turns a pass into byte spans: every occurrence of each string
that is not glued to a word, longest strings first, never overlapping. It reports strings it
cannot find and writes the views. `export` writes the pass's spans as
data/interim/silver/r<ROUND>/train.jsonl, and `gold` writes an evaluation round in the review
set's format as data/interim/review/gold-r<ROUND>.jsonl.
"""

import glob
import json
import pathlib
import random
import re
import sys

KINDS = {"person", "org", "address", "email", "phone"}
PER_SLICE = 10
MAX_CHARS = 6000
SOURCES = {
    "federal-register": ("data/interim/silver/federal-register.jsonl", "US"),
    "govuk": ("data/interim/silver/govuk.jsonl", "GB"),
}
# Collected documents, one JSON file each (internal/bench/review/fetch_sites.py).
COLLECTED = {
    **{s: (f"data/raw/review/{s}/*.json", s[:2].upper())
       for s in ("jp-soumu", "jp-maff", "jp-caa", "ge-economy", "ge-civil", "ge-tsu", "ge-contacts",
                 "de-berlin", "de-impressum", "gb-contacts")},
    **{s: (f"data/raw/silver/{s}/*.json", s[:2].upper())
       for s in ("jp-env", "ge-mepa", "ge-tbilisi", "ge-parliament")},
}
# Rounds 1 and 3 are silver (training) rounds; round 2 is the GE and JP evaluation set and round
# 4 the DE and GB additions to the review set. A count of None takes every document of the source.
PLAN = {
    1: {"federal-register": 150, "govuk": 150},
    2: {s: None for s in ("jp-soumu", "jp-maff", "jp-caa", "ge-economy", "ge-civil", "ge-tsu",
                          "ge-contacts")},
    3: {s: None for s in ("jp-env", "ge-mepa", "ge-tbilisi", "ge-parliament")},
    4: {s: None for s in ("de-berlin", "de-impressum", "gb-contacts")},
}
CONTACT = re.compile(r"@|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|\b0\d{2,4} ?\d{3} ?\d{3,4}\b|\b(Street|Avenue|Road|Room|Suite)\b")


def root(round_no):
    return pathlib.Path(f"data/interim/silver/r{round_no}")


def cut(text):
    """`text` cut at the last paragraph break before MAX_CHARS characters."""
    if len(text) <= MAX_CHARS:
        return text
    head = text[:MAX_CHARS]
    at = head.rfind("\n\n")
    return head[:at] if at > MAX_CHARS // 2 else head[: head.rfind("\n")]


def sample(round_no):
    rng = random.Random(1000 + round_no)
    out = []
    for source, n in PLAN[round_no].items():
        if source in COLLECTED:
            pattern, country = COLLECTED[source]
            docs = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(pattern))]
            if n is None:
                out.extend({"id": d["id"], "source": source, "country": country, "url": d["url"],
                            "date": d["date"], "doc_type": d.get("doc_type", ""), "text": d["text"]}
                           for d in docs)
                continue
        else:
            path, country = SOURCES[source]
            docs = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l]
        # Two thirds from documents with contact details, where the entities are.
        contact = [d for d in docs if CONTACT.search(d["text"])]
        rest = [d for d in docs if not CONTACT.search(d["text"])]
        k = min(len(contact), n * 2 // 3)
        chosen = rng.sample(contact, k) + rng.sample(rest, min(len(rest), n - k))
        for d in chosen:
            out.append({"id": d["id"], "source": source, "country": country, "url": d["url"],
                        "date": d["date"], "text": cut(d["text"])})
    root(round_no).mkdir(parents=True, exist_ok=True)
    (root(round_no) / "sample.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out))
    print(f"{len(out)} documents in {root(round_no) / 'sample.jsonl'}")


def load(round_no):
    return [json.loads(l) for l in (root(round_no) / "sample.jsonl").read_text().splitlines() if l]


def chunks(docs):
    for n in range(0, len(docs), PER_SLICE):
        yield n // PER_SLICE, docs[n:n + PER_SLICE]


def slices(round_no):
    docs = load(round_no)
    for idx, chunk in chunks(docs):
        md = [f"# Slice {idx:02d}\n"]
        for d in chunk:
            md.append(f"\n## Document `{d['id']}` ({d['source']}, {d['country']})\n\n```text\n{d['text']}\n```\n")
        (root(round_no) / f"slice-{idx:02d}.md").write_text("\n".join(md))
    print(f"{(len(docs) + PER_SLICE - 1) // PER_SLICE} slices in {root(round_no)}")


def script(ch):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or ch in "々〆〇":
        return "han"
    if 0x3040 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9D:
        return "katakana"
    return "word"


def joins(a, b):
    """Whether two adjacent letters are one token to the tokenizer: every Han character is a
    token of its own, and a kana run ends where the script changes (`消費者庁が`)."""
    if not (a.isalnum() and b.isalnum()):
        return False
    sa, sb = script(a), script(b)
    return sa == sb and sa != "han"


def glued(b, at, end):
    """Whether bytes `at..end` of `b` continue a word, an identifier, or an address on
    either side: `NRC` in `NRC-2025-0012`, `Smith` in `jsmith@x.gov`. A slash separates
    names (`DOL-OWCP/DFEC` is one org per segment), so it does not glue."""
    before = b[:at].decode("utf-8", "ignore")[-2:]
    after = b[end:].decode("utf-8", "ignore")[:2]
    value = b[at:end].decode("utf-8")
    if before and joins(before[-1], value[0]):
        return True
    if after and joins(value[-1], after[0]):
        return True
    if len(before) == 2 and before[-1] in "-@._" and before[0].isalnum() and value[0].isalnum():
        return True
    if len(after) == 2 and after[0] in "-@_" and after[1].isalnum() and value[-1].isalnum():
        return True
    return False


def occurrences(b, e):
    """Byte offsets of the entity's text in `b`; with `within`, only its first place inside
    each occurrence of that longer context string."""
    v = e["text"].encode("utf-8")
    if e.get("within"):
        w = e["within"].encode("utf-8")
        inner = w.find(v)
        if inner < 0:
            return []
        at, out = b.find(w), []
        while at >= 0:
            out.append(at + inner)
            at = b.find(w, at + 1)
        return out
    at, out = b.find(v), []
    while at >= 0:
        out.append(at)
        at = b.find(v, at + 1)
    return out


def short_cjk(text):
    return len(text) <= 2 and all(script(c) != "word" for c in text)


def spans(text, entities):
    """Byte spans for `entities` (kind, text, optional within), every free occurrence,
    longest first. Returns the spans, the entities not found, and short CJK strings without
    `within` that matched more than once, which usually label unrelated characters."""
    b = text.encode("utf-8")
    taken = []
    out = []
    missing = []
    risky = []
    for e in sorted(entities, key=lambda e: -len(e["text"].encode("utf-8"))):
        v = e["text"].encode("utf-8")
        if not v.strip():
            continue
        places = occurrences(b, e)
        if not places:
            missing.append(e)
        if short_cjk(e["text"]) and not e.get("within") and len(places) > 1:
            risky.append(e)
        for at in places:
            end = at + len(v)
            if not glued(b, at, end) and not any(at < te and ts < end for ts, te in taken):
                taken.append((at, end))
                out.append({"kind": e["kind"], "start": at, "end": end})
    out.sort(key=lambda s: s["start"])
    return out, missing, risky


def read_pass(round_no, pass_no):
    labels = {}
    problems = []
    for p in sorted((root(round_no) / f"pass{pass_no}").glob("slice-*.json")):
        for doc_id, ents in json.loads(p.read_text()).items():
            good = []
            for e in ents:
                if e.get("kind") not in KINDS or not isinstance(e.get("text"), str):
                    problems.append(f"{p.name}: {doc_id}: bad entry {e}")
                else:
                    good.append(e)
            labels[doc_id] = good
    return labels, problems


def render(text, doc_spans):
    b = text.encode("utf-8")
    out, at = [], 0
    for s in doc_spans:
        out.append(b[at:s["start"]].decode("utf-8"))
        out.append(f"⟦{s['kind']}: {b[s['start']:s['end']].decode('utf-8')}⟧")
        at = s["end"]
    out.append(b[at:].decode("utf-8"))
    return "".join(out)


def apply(round_no, pass_no):
    docs = load(round_no)
    labels, problems = read_pass(round_no, pass_no)
    view = root(round_no) / f"pass{pass_no}-view"
    view.mkdir(exist_ok=True)
    counts = {k: 0 for k in KINDS}
    for idx, chunk in chunks(docs):
        md = [f"# Slice {idx:02d}, labels of pass {pass_no}\n"]
        for d in chunk:
            if d["id"] not in labels:
                problems.append(f"slice {idx:02d}: {d['id']} has no labels")
                continue
            doc_spans, missing, risky = spans(d["text"], labels[d["id"]])
            for s in doc_spans:
                counts[s["kind"]] += 1
            for m in missing:
                problems.append(f"{d['id']}: not found: {m['kind']} {m['text']!r}")
            for m in risky:
                problems.append(f"{d['id']}: short CJK string matches several places, give "
                                f"`within`: {m['kind']} {m['text']!r}")
            listed = "\n".join(f"- {e['kind']}: `{e['text']}`" for e in labels[d["id"]]) or "- (none)"
            md.append(f"\n## Document `{d['id']}` ({d['source']}, {d['country']})\n\n"
                      f"```text\n{render(d['text'], doc_spans)}\n```\n\nPass {pass_no} list:\n\n{listed}\n")
        (view / f"slice-{idx:02d}.md").write_text("\n".join(md))
    print(f"pass {pass_no}: {counts}, views in {view}")
    for p in problems:
        print("PROBLEM", p)


def gold(round_no, pass_no):
    docs = load(round_no)
    labels, problems = read_pass(round_no, pass_no)
    rows = []
    for d in docs:
        if d["id"] not in labels:
            problems.append(f"{d['id']} has no labels")
            continue
        b = d["text"].encode("utf-8")
        doc_spans, _, _ = spans(d["text"], labels[d["id"]])
        rows.append({"name": d["id"], "input": d["text"], "country": d["country"],
                     "doc_type": d.get("doc_type", ""),
                     "expected": [s | {"text": b[s["start"]:s["end"]].decode("utf-8")} for s in doc_spans]})
    out = pathlib.Path(f"data/interim/review/gold-r{round_no}.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} documents in {out}")
    for p in problems:
        print("PROBLEM", p)


def export(round_no, pass_no):
    docs = load(round_no)
    labels, problems = read_pass(round_no, pass_no)
    rows = []
    for d in docs:
        if d["id"] not in labels:
            continue
        doc_spans, _, _ = spans(d["text"], labels[d["id"]])
        rows.append({"id": d["id"], "source": d["source"], "country": d["country"], "text": d["text"],
                     "entities": doc_spans})
    out = root(round_no) / "train.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} documents in {out}")
    for p in problems:
        print("PROBLEM", p)


if __name__ == "__main__":
    cmd, round_no = sys.argv[1], int(sys.argv[2])
    if cmd == "sample":
        sample(round_no)
    elif cmd == "slices":
        slices(round_no)
    elif cmd == "apply":
        apply(round_no, int(sys.argv[3]))
    elif cmd == "export":
        export(round_no, int(sys.argv[3]))
    elif cmd == "gold":
        gold(round_no, int(sys.argv[3]))
