"""Freeze source-faithful US congressional office contact cases."""

import collections
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/us-offices"
OUTPUT = ROOT / "data/interim/review/us-office-eval-v1.jsonl"
MANIFEST = ROOT / "data/interim/review/us-office-eval-v1.manifest.json"
DEV = ROOT / "data/interim/review/us-dev-v1.jsonl"
EXCLUSIONS = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
PHONE = re.compile(r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}")

# Each tuple gives the inclusive source line range and the first and last address lines.
OFFICES = {
    "cantwell": ((101, 105, 102, 103), (107, 111, 108, 109),
                 (113, 117, 114, 115), (119, 123, 120, 121),
                 (125, 129, 126, 127), (130, 134, 131, 132)),
    "case": ((68, 72, 70, 71), (77, 81, 79, 80)),
    "ron-johnson": ((52, 57, 54, 55), (61, 66, 63, 64),
                    (70, 75, 72, 73), (78, 83, 80, 81)),
    "van-hollen": ((41, 46, 43, 44), (50, 55, 52, 54),
                   (59, 65, 61, 63), (69, 73, 70, 72),
                   (78, 83, 80, 82), (88, 93, 90, 92),
                   (98, 102, 99, 101)),
    "whitehouse": ((11, 14, 11, 12), (20, 23, 20, 21)),
}

ADDITIONAL = (
    ("doe-legacy-mail", "doe-legacy", 4, 11, (
        ("org", "U.S. Department of Energy"),
        ("org", "Office of Legacy Management"),
        ("address", "1000 Independence Avenue, SW\nWashington, DC 20585"),
        ("phone", "(202) 586-7550"),
        ("phone", "(202) 586-8403"),
    )),
    ("doe-legacy-other", "doe-legacy", 12, 19, (
        ("email", "LM@hq.doe.gov"),
        ("email", "LMWebsiteSupport@lm.doe.gov"),
        ("org", "DOE Public Affairs Office"),
        ("phone", "(202) 586-4940"),
        ("email", "doenews@hq.doe.gov"),
    )),
    ("doe-nuclear-mail", "doe-nuclear", 10, 19, (
        ("org", "Office of Nuclear Energy"),
        ("org", "U.S. Department of Energy"),
        ("address", "1000 Independence Ave., SW\nWashington, DC 20585"),
        ("phone", "(202) 586-2240"),
    )),
    ("doe-nuclear-email", "doe-nuclear", 6, 8, (
        ("email", "NECommunications@Nuclear.Energy.gov"),
    )),
    ("hal-rogers-staff", "hal-rogers", 11, 19, (
        *(("person", name) for name in (
            "Karen Kelly", "Kelley Kurtz", "Danielle Smoot", "Will Tener",
            "Will Reynolds", "Katie Hessenius", "Emily Hale", "Heath Maynard",
        )),
        ("email", "schedule.rogers@mail.house.gov"),
        ("email", "danielle.smoot@mail.house.gov"),
    )),
    ("house-ethics-leadership", "house-ethics", 56, 67, (
        *(("person", name) for name in (
            "Tom Rust", "Allison Chipman", "Jordan Downs", "Kate Seibert", "David Arrojo",
        )),
    )),
    ("house-ethics-advice", "house-ethics", 68, 85, (
        *(("person", name) for name in (
            "Sarah Myers-Mutschall", "Tamar Nedzar", "Emily Eisenrauch",
            "Katherine Fitzpatrick", "Arlinda Rouse", "Roshan Patel", "Nicholas Long",
        )),
    )),
    ("house-ethics-disclosure", "house-ethics", 86, 100, (
        *(("person", name) for name in (
            "Stephanie Richards", "Chris Pierce", "Annie Hudgins", "Deborah Bethea",
            "Kate McLane",
        )),
    )),
    ("house-ethics-investigations", "house-ethics", 101, 118, (
        *(("person", name) for name in (
            "Brittney Pescatore", "Sydney Bellwoar", "Melissa Chong", "Nicqelle Fleming",
            "Christine Gwinn", "Ray Rhatican", "Jennifer Seeba",
        )),
    )),
    ("house-ethics-administration", "house-ethics", 119, 136, (
        *(("person", name) for name in (
            "Melanie Cohan", "Dorian Mitchell", "Peyton Wilmer", "Denia Alcoser",
            "Jack Hovland", "Justin Lopez-Beltran", "Sara Ryan",
        )),
    )),
    ("nasa-headquarters", "nasa", 416, 422, (
        ("org", "Mary W. Jackson NASA Headquarters"),
        ("address", "300 E. Street SW, Suite 5R30\nWashington, DC 20546"),
        ("phone", "(202) 358-0001"),
        ("phone", "(202) 358-4338"),
    )),
    ("nasa-page-staff", "nasa", 472, 478, (
        ("person", "Brittany A. Brown"),
        ("person", "Abigail Bowman"),
    )),
    ("nist-itl-mail", "nist-itl", 129, 138, (
        ("org", "National Institute of Standards and Technology (NIST)"),
        ("org", "Information Technology Laboratory (ITL)"),
        ("org", "Software and Systems Division (775)"),
        ("address", "100 Bureau Drive, Mail Stop 8970\nGaithersburg, MD 20899-8970"),
        ("phone", "301-975-3345"),
        ("phone", "301-975-6097"),
    )),
    ("nih-ods-mail", "nih-ods", 52, 56, (
        ("org", "Office of Dietary Supplements"),
        ("org", "National Institutes of Health"),
        ("address", "6705 Rockledge Dr., Room 730, MSC 7991\nBethesda, MD 20817"),
    )),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def byte_span(text, start, end, kind):
    value = text[start:end]
    return {
        "kind": kind,
        "start": len(text[:start].encode()),
        "end": len(text[:end].encode()),
        "text": value,
    }


def main():
    dev = [json.loads(line) for line in DEV.read_text().splitlines()]
    dev_names = {row["name"] for row in dev}
    dev_texts = {digest(row["input"].encode()) for row in dev}
    cases = []
    source_hashes = {}
    counts = collections.Counter()
    for name, specs in OFFICES.items():
        path = SOURCE / f"{name}.json"
        source_hashes[name] = digest(path.read_bytes())
        source = json.loads(path.read_text())
        lines = {row["line"]: row["text"] for row in source["rendered_lines"]}
        if len(lines) != len(source["rendered_lines"]):
            raise ValueError(f"duplicate rendered source line: {name}")
        for index, (first, last, address_first, address_last) in enumerate(specs, 1):
            block = [lines[line] for line in range(first, last + 1)]
            if any("\ue200" in line or "\ue201" in line for line in block):
                raise ValueError(f"tool citation marker in selected source: {name}:{first}")
            text = "\n".join(block)
            address = "\n".join(lines[line] for line in range(address_first, address_last + 1))
            if not address or text.count(address) != 1:
                raise ValueError(f"ambiguous address: {name}:{first}")
            start = text.index(address)
            expected = [byte_span(text, start, start + len(address), "address")]
            expected.extend(byte_span(text, match.start(), match.end(), "phone") for match in PHONE.finditer(text))
            if not any(span["kind"] == "phone" for span in expected):
                raise ValueError(f"missing contact number: {name}:{first}")
            expected.sort(key=lambda span: span["start"])
            case_name = f"us-office-{name}-{index:02}"
            if case_name in dev_names or digest(text.encode()) in dev_texts:
                raise ValueError(f"development evaluation overlap: {case_name}")
            raw = text.encode()
            for span in expected:
                if raw[span["start"]:span["end"]].decode() != span["text"]:
                    raise ValueError(f"bad byte offset: {case_name}")
                counts[span["kind"]] += 1
            cases.append({
                "name": case_name,
                "input": text,
                "expected": expected,
                "country": "US",
                "doc_type": "congressional_office_contact",
                "source_host": source["source_url"].split("/")[2],
                "source_url": source["source_url"],
                "source_lines": [first, last],
            })
    for case_name, source_name, first, last, labels in ADDITIONAL:
        path = SOURCE / f"{source_name}.json"
        source_hashes[source_name] = digest(path.read_bytes())
        source = json.loads(path.read_text())
        lines = {row["line"]: row["text"] for row in source["rendered_lines"]}
        if len(lines) != len(source["rendered_lines"]):
            raise ValueError(f"duplicate rendered source line: {source_name}")
        block = [lines[line] for line in range(first, last + 1)]
        if any("\ue200" in line or "\ue201" in line for line in block):
            raise ValueError(f"tool citation marker in selected source: {case_name}")
        text = "\n".join(block)
        expected = []
        for kind, value in labels:
            matches = list(re.finditer(re.escape(value), text))
            if not matches:
                raise ValueError(f"missing {kind} surface in {case_name}: {value}")
            expected.extend(byte_span(text, match.start(), match.end(), kind) for match in matches)
        expected.sort(key=lambda span: span["start"])
        for left, right in zip(expected, expected[1:]):
            if left["end"] > right["start"]:
                raise ValueError(f"overlapping labels: {case_name}")
        raw = text.encode()
        for span in expected:
            if raw[span["start"]:span["end"]].decode() != span["text"]:
                raise ValueError(f"bad byte offset: {case_name}")
            counts[span["kind"]] += 1
        cases.append({
            "name": f"us-office-{case_name}",
            "input": text,
            "expected": expected,
            "country": "US",
            "doc_type": "government_contact_or_staff",
            "source_host": source["source_url"].split("/")[2],
            "source_url": source["source_url"],
            "source_lines": [first, last],
        })
    if len({digest(row["input"].encode()) for row in cases}) != len(cases):
        raise ValueError("duplicate office evaluation text")
    if any(row["name"] in dev_names or digest(row["input"].encode()) in dev_texts for row in cases):
        raise ValueError("office evaluation overlaps development cases")
    output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases).encode()
    exclusions = DEV.read_bytes() + output
    OUTPUT.write_bytes(output)
    EXCLUSIONS.write_bytes(exclusions)
    MANIFEST.write_text(json.dumps({
        "kind": "us_office_development_evaluation",
        "status": "source_and_offset_verified; one_label_review_complete",
        "cases": len(cases),
        "labels": dict(counts),
        "sources_sha256": source_hashes,
        "dev_sha256": digest(DEV.read_bytes()),
        "exclusions_sha256": digest(exclusions),
        "sha256": digest(output),
        "use": "development evaluation only; never training or final release gating",
    }, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(cases)} office cases with {dict(counts)} labels")


if __name__ == "__main__":
    main()
