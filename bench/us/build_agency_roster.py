"""Build a conservative US public-body roster from the captured USAGov index."""

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/us-agencies/usa-gov-index.json"
EVALUATION = ROOT / "data/interim/review/us-eval-exclusions-v1.jsonl"
OUTPUT = ROOT / "trainer/data/us-public-bodies.json"
ACRONYM = re.compile(r"^(.*?) \(([A-Z0-9]{2,12})\)$")
NOT_BODIES = {
    "CPIR", "EJI", "FCSM", "FEDLINK", "HFP", "IAQ", "JFSP", "NFIP",
    "NHIC", "NIFC", "NISIC", "NPIN", "OFR", "PIH",
}
PREFERRED = {
    "CFPB": "Consumer Financial Protection Bureau",
    "FHEO": "Office of Fair Housing and Equal Opportunity",
    "IER": "Office of Immigrant and Employee Rights",
    "NAL": "National Agricultural Library",
    "TTB": "Alcohol and Tobacco Tax and Trade Bureau",
    "USCIS": "U.S. Citizenship and Immigration Services",
    "WHD": "Wage and Hour Division",
}


def normalized(value):
    value = value.casefold().replace("u.s.", "us")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def evaluation_orgs():
    orgs = set()
    for line in EVALUATION.read_text().splitlines():
        for span in json.loads(line)["expected"]:
            if span["kind"] == "org":
                orgs.add(normalized(span["text"]))
    return orgs


def touches_evaluation(acronym, aliases, orgs):
    surfaces = {normalized(alias) for alias in aliases}
    surfaces.add(normalized(acronym))
    for surface in surfaces:
        if len(surface) < 3:
            continue
        for org in orgs:
            if f" {surface} " in f" {org} " or f" {org} " in f" {surface} ":
                return True
    return False


def canonical(acronym, entries):
    if acronym in PREFERRED:
        chosen = next((entry for entry in entries if entry["name"] == PREFERRED[acronym]), None)
        if chosen is None:
            raise ValueError(f"missing preferred source name: {acronym}")
        return chosen
    return max(entries, key=lambda entry: (
        entry["name"].startswith("U.S. Department of "),
        entry["name"].startswith("Department of "),
        entry["name"].startswith("U.S. "),
        len(entry["name"]),
    ))


def main():
    source = json.loads(SOURCE.read_text())
    if source["source"] != "USAGov A-Z agency index":
        raise ValueError("unexpected agency source")
    groups = defaultdict(list)
    for page in source["pages"]:
        for item in page["names"]:
            match = ACRONYM.fullmatch(item["name"])
            if match is None or "," in match[1]:
                continue
            groups[match[2]].append({
                "name": match[1],
                "source_url": page["source_url"],
                "source_line": item["source_line"],
            })
    orgs = evaluation_orgs()
    rows = []
    for acronym, entries in sorted(groups.items()):
        if acronym in NOT_BODIES:
            continue
        if touches_evaluation(acronym, [entry["name"] for entry in entries], orgs):
            continue
        chosen = canonical(acronym, entries)
        rows.append({"name": chosen["name"], "acronym": acronym,
                     "source_url": chosen["source_url"], "source_line": chosen["source_line"]})
    if len(rows) < 150 or len({row["name"].casefold() for row in rows}) != len(rows):
        raise ValueError("agency roster is too small or contains duplicate names")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "source": source["source"],
        "captured_on": source["captured_on"],
        "selection": "one reviewed acronymed body per source alias group; evaluation bodies excluded",
        "bodies": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} public bodies written to {OUTPUT}")


if __name__ == "__main__":
    main()
