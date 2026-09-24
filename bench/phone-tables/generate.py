"""Generate tessera/src/rules/phone_tables.rs from libphonenumber's metadata XML.

The library validates phone numbers by calling code, possible national lengths,
and leading-digit prefixes of up to four digits. This script derives those from
libphonenumber's per-type regular expressions, so the library needs no regex
engine and no copy of the full metadata.

Usage, from this directory:

    python3 generate.py [--tag v9.0.33] [--regions US,CA,...]
                        [--out ../../tessera/src/rules/phone_tables.rs]
                        [--report ../../internal/reports/phone-tables.md]

The XML is downloaded once per tag into cache/, which is gitignored. Python
standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REGIONS = "US,CA,GB,IE,DE,AT,CH,NL,BE,GE,JP"
# Ordinary subscriber numbers. A length that only special types use (for example Canada's
# 7-digit universal access numbers) is accepted only when a known prefix also matches.
CORE_TYPES = {"fixedLine", "mobile"}
TYPES = [
    "fixedLine", "mobile", "tollFree", "premiumRate", "sharedCost",
    "voip", "personalNumber", "uan", "pager", "voicemail",
]
MAX_LEN = 4
MIN_LEN = 2
MAX_SET = 2000
ANY = "x"


class Unsupported(Exception):
    pass


def concat(left: set[str], right: set[str]) -> set[str]:
    out = set()
    for a in left:
        if len(a) >= MAX_LEN:
            out.add(a)
            continue
        for b in right:
            out.add((a + b)[:MAX_LEN])
    if len(out) > MAX_SET:
        raise Unsupported("prefix set too large")
    return out


def repeat(inner: set[str], lo: int, hi: int | None) -> set[str]:
    """Concatenate `inner` between lo and hi times; hi None means unbounded."""
    acc = {""}
    for _ in range(lo):
        acc = concat(acc, inner)
    result = set(acc)
    extra = (hi - lo) if hi is not None else MAX_LEN
    for _ in range(extra):
        acc = concat(acc, inner)
        before = len(result)
        result |= acc
        if hi is None and len(result) == before:
            break
    if len(result) > MAX_SET:
        raise Unsupported("prefix set too large")
    return result


class Parser:
    """Recursive descent over the regex subset libphonenumber's patterns use."""

    def __init__(self, pattern: str):
        self.s = pattern
        self.i = 0

    def peek(self) -> str | None:
        return self.s[self.i] if self.i < len(self.s) else None

    def take(self) -> str:
        c = self.s[self.i]
        self.i += 1
        return c

    def parse(self) -> set[str]:
        result = self.alternation()
        if self.i != len(self.s):
            raise Unsupported(f"unexpected {self.s[self.i]!r} at {self.i}")
        return result

    def alternation(self) -> set[str]:
        result = self.sequence()
        while self.peek() == "|":
            self.take()
            result |= self.sequence()
        return result

    def sequence(self) -> set[str]:
        result = {""}
        while self.peek() is not None and self.peek() not in "|)":
            result = concat(result, self.quantified())
        return result

    def quantified(self) -> set[str]:
        atom = self.atom()
        c = self.peek()
        if c == "?":
            self.take()
            return repeat(atom, 0, 1)
        if c == "*":
            self.take()
            return repeat(atom, 0, None)
        if c == "+":
            self.take()
            return repeat(atom, 1, None)
        if c == "{":
            m = re.match(r"\{(\d+)(?:(,)(\d*))?\}", self.s[self.i:])
            if not m:
                raise Unsupported("bad repeat")
            self.i += m.end()
            lo = int(m.group(1))
            if m.group(2) is None:
                hi = lo
            else:
                hi = int(m.group(3)) if m.group(3) else None
            return repeat(atom, lo, hi)
        return atom

    def atom(self) -> set[str]:
        c = self.take()
        if c.isdigit():
            return {c}
        if c == "\\":
            e = self.take()
            if e == "d":
                return {ANY}
            raise Unsupported(f"escape \\{e}")
        if c == "[":
            return self.char_class()
        if c == "(":
            if self.s.startswith("?:", self.i):
                self.i += 2
            elif self.peek() == "?":
                raise Unsupported("group modifier")
            inner = self.alternation()
            if self.take() != ")":
                raise Unsupported("unclosed group")
            return inner
        raise Unsupported(f"construct {c!r}")

    def char_class(self) -> set[str]:
        if self.peek() == "^":
            raise Unsupported("negated class")
        digits = set()
        while self.peek() != "]":
            if self.peek() is None:
                raise Unsupported("unclosed class")
            c = self.take()
            if c == "\\" and self.peek() == "d":
                self.take()
                digits |= set("0123456789")
                continue
            if not c.isdigit():
                raise Unsupported(f"class member {c!r}")
            if self.peek() == "-" and self.s[self.i + 1:self.i + 2].isdigit():
                self.take()
                hi = self.take()
                digits |= {str(d) for d in range(int(c), int(hi) + 1)}
            else:
                digits.add(c)
        self.take()
        return {ANY} if len(digits) == 10 else digits


def prefixes(pattern: str) -> tuple[set[str], int]:
    """Prefixes of the longest length from MAX_LEN down to MIN_LEN whose set stays under MAX_SET."""
    global MAX_LEN
    top = MAX_LEN
    try:
        for length in range(top, MIN_LEN - 1, -1):
            MAX_LEN = length
            try:
                return Parser(pattern).parse(), length
            except Unsupported as err:
                if "too large" not in str(err):
                    raise
        raise Unsupported(f"prefix set too large even at {MIN_LEN} digits")
    finally:
        MAX_LEN = top


def covers(short: str, long: str) -> bool:
    return len(short) < len(long) and all(s == ANY or s == l for s, l in zip(short, long))


def minimize(ps: set[str]) -> list[str]:
    if "" in ps:
        return [""]
    kept = [p for p in ps if not any(covers(q, p) for q in ps)]
    return sorted(kept)


def parse_lengths(attr: str | None) -> set[int]:
    out: set[int] = set()
    for item in (attr or "").split(","):
        item = item.strip()
        if not item or item == "-1":
            continue
        m = re.fullmatch(r"\[(\d+)-(\d+)\]", item)
        if m:
            out |= set(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            out.add(int(item))
    return out


def load_xml(tag: str) -> ET.Element:
    cache = HERE / "cache" / f"{tag}.xml"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/google/libphonenumber/{tag}/resources/PhoneNumberMetadata.xml"
        with urllib.request.urlopen(url, timeout=120) as resp:
            cache.write_bytes(resp.read())
    return ET.parse(cache).getroot()


def rust_str_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v9.0.33")
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--out", default=str(HERE.parents[1] / "tessera" / "src" / "rules" / "phone_tables.rs"))
    ap.add_argument("--report", default=str(HERE.parents[1] / "internal" / "reports" / "phone-tables.md"))
    args = ap.parse_args()

    wanted = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    root = load_xml(args.tag)
    territories = {t.get("id"): t for t in root.iter("territory")}
    missing = [r for r in wanted if r not in territories]
    if missing:
        print(f"unknown regions: {missing}", file=sys.stderr)
        return 1

    rows, report_rows, fallbacks, shortened = [], [], [], []
    for rid in sorted(wanted):
        t = territories[rid]
        lengths: set[int] = set()
        core_lengths: set[int] = set()
        prefix_set: set[str] = set()
        for ty in TYPES:
            e = t.find(ty)
            if e is None:
                continue
            pattern = re.sub(r"\s", "", e.findtext("nationalNumberPattern") or "")
            pl = e.find("possibleLengths")
            found_lengths = parse_lengths(pl.get("national") if pl is not None else None)
            lengths |= found_lengths
            if ty in CORE_TYPES:
                core_lengths |= found_lengths
            if not pattern:
                continue
            try:
                found, used = prefixes(pattern)
                prefix_set |= found
                if used < MAX_LEN:
                    shortened.append((rid, ty, used))
            except Unsupported as err:
                fallbacks.append((rid, ty, str(err)))
                prefix_set.add("")
        final = minimize(prefix_set)
        np = t.get("nationalPrefix")
        rows.append(
            f'    Region {{ id: "{rid}", code: {int(t.get("countryCode"))}, '
            f'national_prefix: {f"Some({chr(34)}{np}{chr(34)})" if np else "None"}, '
            f'lengths: &{sorted(lengths)}, core_lengths: &{sorted(core_lengths)}, '
            f'main: {"true" if t.get("mainCountryForCode") == "true" else "false"}, '
            f'prefixes: &{rust_str_list(final)} }},'
        )
        report_rows.append(f"| {rid} | {t.get('countryCode')} | {np or ''} | {', '.join(map(str, sorted(lengths)))} | {len(final)} |")

    # Every other country's calling code, main region, and possible national lengths, so a
    # number written with its code is still found where no full table is shipped.
    covered = {int(territories[r].get("countryCode")) for r in wanted}
    others: dict[int, tuple[str, set[int]]] = {}
    for rid, t in territories.items():
        code = int(t.get("countryCode"))
        if code in covered or not re.fullmatch(r"[A-Z]{2}", rid or ""):
            continue
        lengths: set[int] = set()
        for ty in TYPES:
            e = t.find(ty)
            pl = e.find("possibleLengths") if e is not None else None
            if pl is not None:
                lengths |= parse_lengths(pl.get("national"))
        main_id, main_lengths = others.get(code, (rid, set()))
        if t.get("mainCountryForCode") == "true":
            main_id = rid
        others[code] = (main_id, main_lengths | lengths)
    other_rows = [
        f"    ({code}, *b\"{rid}\", 0b{sum(1 << n for n in lengths):b}),"
        for code, (rid, lengths) in sorted(others.items())
        if lengths
    ]

    header = (
        f"//! Generated by bench/phone-tables/generate.py from libphonenumber {args.tag}. Do not edit.\n"
        "//!\n"
        "//! One entry per supported region: calling code, national prefix, possible national\n"
        "//! number lengths, and leading digits of valid national numbers. Then every other\n"
        "//! calling code with its main region and possible national lengths.\n\n"
        "pub(crate) struct Region {\n"
        "    pub id: &'static str,\n"
        "    pub code: u16,\n"
        "    pub national_prefix: Option<&'static str>,\n"
        "    /// Every possible national length, over all number types.\n"
        "    pub lengths: &'static [u8],\n"
        "    /// Lengths of fixed-line and mobile numbers; the only lengths accepted without a known prefix.\n"
        "    pub core_lengths: &'static [u8],\n"
        "    /// Whether this is libphonenumber's main country for its calling code (US for +1).\n"
        "    pub main: bool,\n"
        "    /// Leading digits of valid national numbers; `x` matches any digit; `\"\"` matches all.\n"
        "    pub prefixes: &'static [&'static str],\n"
        "}\n\n"
        "pub(crate) static REGIONS: &[Region] = &[\n"
    )
    footer = (
        "\n];\n\n"
        "/// Calling codes outside `REGIONS`, sorted: code, main region, and the possible national\n"
        "/// lengths as a mask, bit `n` set for length `n`.\n"
        "pub(crate) static OTHER_CODES: &[(u16, [u8; 2], u32)] = &[\n"
        + "\n".join(other_rows)
        + "\n];\n"
    )
    Path(args.out).write_text(header + "\n".join(rows) + footer, encoding="utf-8")

    report = [
        "# Phone tables", "",
        f"Generated from libphonenumber {args.tag} `PhoneNumberMetadata.xml` by `bench/phone-tables/generate.py`.",
        f"Regions: {', '.join(sorted(wanted))}. Prefix length: {MAX_LEN}.", "",
        "| Region | Code | National prefix | Lengths | Prefixes |",
        "| --- | ---: | --- | --- | ---: |",
        *report_rows, "",
        "## Shortened prefixes", "",
        *([f"- {r} {ty}: {n}-digit prefixes, because {MAX_LEN} digits exceeded {MAX_SET} entries." for r, ty, n in shortened] or ["None."]),
        "",
        "## Pattern fallbacks", "",
    ]
    if fallbacks:
        report += [f"- {r} {ty}: {why}; that type accepts any leading digits." for r, ty, why in fallbacks]
    else:
        report.append("None. Every pattern was expanded into prefixes.")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} regions, {len(other_rows)} other codes, {len(fallbacks)} fallbacks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
