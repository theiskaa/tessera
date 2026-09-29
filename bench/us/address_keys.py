"""Find conservative physical-address keys for US evaluation leakage checks."""

import re


STREET_TYPES = {
    "st", "street", "ave", "avenue", "rd", "road", "dr", "drive", "blvd",
    "boulevard", "ln", "lane", "way", "pl", "place", "pkwy", "parkway",
    "ct", "court", "cir", "circle", "pike", "hwy", "highway", "ter",
    "terrace", "plaza", "sq", "square", "trail", "trl",
}
DIRECTIONS = {
    "n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south",
    "east", "west", "northeast", "northwest", "southeast", "southwest",
}
NON_STREET_STARTS = {"of", "the", "room", "suite", "floor", "building", "code", "mail", "stop"}
POSTCODE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
NUMBER_WORD = re.compile(r"\b(\d{1,6})[A-Za-z]?[ \t]+([A-Za-z]+)\b")
NUMBERED_STREET_IN_TEXT = re.compile(
    r"\b\d{1,6}[ \t]+\d{1,3}(?:st|nd|rd|th)[ \t]+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Blvd|Boulevard|Pl|Place)\b", re.I)
MAX_POSTCODE_DISTANCE = 180


def first_street_word(parts):
    index = 0
    while index < len(parts) and parts[index] in DIRECTIONS:
        index += 1
    if index >= len(parts) or parts[index] in NON_STREET_STARTS:
        return None
    return parts[index]


def address_keys(value):
    """Return ZIP, street number, and first street word for plausible addresses."""
    postcodes = list(POSTCODE.finditer(value))
    if not postcodes:
        return set()
    streets = {}
    offset = 0
    for line in value.splitlines(keepends=True):
        number = re.match(r"^[ \t]*(\d{1,6})[A-Za-z]?[ \t]+", line)
        if number is None:
            offset += len(line)
            continue
        parts = re.findall(r"[a-z]+|\d+", line[number.end():].casefold())
        word = first_street_word(parts)
        if word:
            streets[offset + number.start(1)] = (number[1], word, offset + number.end())
        offset += len(line)
    for match in NUMBER_WORD.finditer(value):
        tail = re.split(r"[,;\n]", value[match.start(2):match.start() + 90], maxsplit=1)[0]
        parts = re.findall(r"[a-z]+|\d+", tail.casefold())
        word = first_street_word(parts)
        if word and any(part in STREET_TYPES for part in parts[:10]):
            streets[match.start()] = (match[1], word, match.end())
    keys = set()
    ordered = sorted(streets.items())
    for index, (start, (number, word, end)) in enumerate(ordered):
        next_street = ordered[index + 1][0] if index + 1 < len(ordered) else len(value)
        for postcode in postcodes:
            if postcode.start() < end:
                continue
            if postcode.start() >= min(start + MAX_POSTCODE_DISTANCE, next_street):
                break
            keys.add((postcode[1], number, word))
            break
    return keys


def address_keys_in_text(value):
    """Also find ordinal street names when an address appears inside prose."""
    keys = address_keys(value)
    for match in NUMBERED_STREET_IN_TEXT.finditer(value):
        keys.update(address_keys(value[match.start():match.start() + MAX_POSTCODE_DISTANCE]))
    return keys
