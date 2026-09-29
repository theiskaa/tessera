"""Curated names for the same US body in strict name-disjoint evaluation."""

import json
import unicodedata
from pathlib import Path


ROSTER = Path(__file__).resolve().parents[2] / "trainer/data/us-public-bodies.json"


FAMILIES = (
    ("Commerce", "Commerce Department", "Department of Commerce",
     "U.S. Department of Commerce"),
    ("Department of Labor", "U.S. Department of Labor", "DOL"),
    ("National Park Service", "U.S. National Park Service", "NPS"),
    ("Department of Defense", "U.S. Department of Defense", "DoD", "DOD"),
    ("Department of Homeland Security", "U.S. Department of Homeland Security", "DHS"),
    ("Department of Transportation", "U.S. Department of Transportation", "DOT", "USDOT"),
    ("Postal Service", "U.S. Postal Service", "United States Postal Service", "USPS"),
    ("U.S. Coast Guard", "Coast Guard", "USCG"),
    ("Fish and Wildlife Service", "U.S. Fish and Wildlife Service", "FWS", "USFWS"),
    ("Office of Management and Budget", "OMB"),
    ("Food and Drug Administration", "FDA"),
    ("Department of Veterans Affairs", "VA", "DVA"),
    ("Environmental Protection Agency", "U.S. Environmental Protection Agency", "EPA"),
    ("U.S. Department of Agriculture", "USDA"),
    ("U.S. Department of Energy", "DOE"),
    ("General Services Administration", "U.S. General Services Administration", "GSA"),
    ("Bureau of Justice Statistics", "BJS"),
    ("Federal Energy Regulatory Commission", "FERC"),
    ("Federal Highway Administration", "FHWA"),
    ("Federal Insurance Office", "FIO"),
    ("Food and Nutrition Service", "FNS"),
    ("National Nuclear Security Administration", "NNSA"),
    ("National Marine Fisheries Service", "NMFS"),
    ("Public Health Service", "PHS"),
    ("Forest Service", "U.S. Forest Service", "USFS"),
    ("U.S. International Trade Commission", "United States International Trade Commission",
     "ITC", "USITC"),
    ("U.S. Customs and Border Protection", "CBP"),
    ("Department of the Treasury", "U.S. Department of the Treasury", "Treasury"),
    ("Department of Justice", "U.S. Department of Justice",
     "United States Department of Justice", "DOJ"),
    ("Social Security Administration", "SSA"),
    ("Securities and Exchange Commission", "SEC"),
    ("State, Private and Tribal Forestry", "State, Private, and Tribal Forestry"),
    ("Office of Regulatory Affairs and Collaborative Action",
     "Office of Regulatory Affairs and Collaborative Action--Indian Affairs"),
    ("International Trade Administration", "ITA"),
    ("Administrative Office of the U.S. Courts",
     "Administrative Office of the United States Courts", "AOUSC"),
)

READINESS_FAMILIES = (
    ("Small Business Administration", "SBA"),
    ("Federal Emergency Management Agency", "FEMA"),
    ("Federal Aviation Administration", "FAA"),
    ("Federal Communications Commission", "FCC"),
    ("Bureau of Land Management", "BLM"),
    ("U.S. Merchant Marine Academy", "USMMA"),
    ("U.S. Army Corps of Engineers", "US Army Corps. of Engineers"),
)


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def roster_families():
    bodies = json.loads(ROSTER.read_text())["bodies"]
    families = []
    seen_names = set()
    seen_acronyms = set()
    for body in bodies:
        name = normalized(body["name"])
        acronym = normalized(body["acronym"])
        if (not name or not acronym or name == acronym or
                name in seen_names or acronym in seen_acronyms):
            raise ValueError("public body roster has an invalid or duplicate alias pair")
        seen_names.add(name)
        seen_acronyms.add(acronym)
        families.append((name, acronym))
    if not families:
        raise ValueError("public body roster has no alias pairs")
    return families


def active_aliases(gold, *, include_roster=False):
    observed = {normalized(span["text"]) for row in gold for span in row["expected"]
                if span["kind"] == "org"}
    active = set()
    families = FAMILIES + (READINESS_FAMILIES + tuple(roster_families())
                          if include_roster else ())
    for family in families:
        if any(normalized(name) in observed for name in family):
            active.update(normalized(name) for name in family)
    return active
