"""Screen blind US packets against reserved labels before review or promotion."""

import argparse
import collections
import json
import re
from pathlib import Path

from address_keys import address_keys_in_text
from check_holdout_overlap import collisions, contact_collisions, index_gold
from check_ready import normalized
from check_us_next_data_split import EVALUATION, rows, digest
from screen_2026_contact_expansion_v3 import shingles
from source_identity import document_key, source_url_key


KINDS = ("person", "org", "address", "email", "phone")
EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
EMAIL_LOCAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.!#$%&'*+/=?^_`{|}~-@\\")
EMAIL_DOMAIN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-@")


def phrase_in_text(phrase, text, raw):
    """Keep two-letter acronym surfaces distinct from ordinary lowercase words."""
    if len(phrase) <= 2 and phrase.isalpha():
        phrase, text = phrase.upper(), raw
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def entity_surface_text(text):
    """Ignore complete rule-email regions when screening learned entity surfaces."""
    masked = list(text)
    for match in EMAIL.finditer(text):
        start, end = match.span()
        if (start and text[start - 1].casefold() in EMAIL_LOCAL_CHARS or
                end < len(text) and text[end].casefold() in EMAIL_DOMAIN_CHARS):
            continue
        if re.search(r"[^ \t\n\r\f\v]*://[^ \t\n\r\f\v]*$", text[:start]):
            continue
        local, domain = match.group().rsplit("@", 1)
        if (not match.group().isascii() or len(local) > 64 or
                local.startswith(".") or local.endswith(".") or
                ".." in local or len(domain) > 253 or
                any(not label or len(label) > 63 or label.startswith("-") or
                    label.endswith("-") for label in domain.split("."))):
            continue
        masked[start:end] = " " * (end - start)
    return "".join(masked)


def validate_spans(row, spans):
    """Ensure a reviewed row still labels its exact frozen UTF-8 text."""
    source = row["input"].encode()
    ordered = []
    for span in spans:
        start, end = span["start"], span["end"]
        if (span["kind"] not in KINDS or not 0 <= start < end <= len(source) or
                source[start:end].decode() != span["text"]):
            raise ValueError(f"invalid span in {row['name']}: {span}")
        ordered.append((start, end))
    ordered.sort()
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"overlapping spans in {row['name']}")


def heldout_reasons(packet, gold):
    """Return conservative text hits and exact reviewed-label hits by packet case."""
    surfaces, _, addresses, aliases = index_gold(gold, strict_aliases=True)
    eval_keys = {document_key(row) for row in gold}
    eval_urls = {source_url_key(row) for row in gold} - {None}
    eval_groups = {row["source_group"] for row in gold if row.get("source_group")}
    eval_fragments = set().union(*(shingles(row["input"], 12) for row in gold))
    contacts = collections.defaultdict(set)
    for row in gold:
        for span in row["expected"]:
            if span["kind"] == "email":
                contacts["email"].add(span["text"].casefold())
            elif span["kind"] == "phone":
                contacts["phone"].add(re.sub(r"\D", "", span["text"]))
    reasons = collections.defaultdict(set)
    labeled = []
    for row in packet:
        name = row["name"]
        entity_text = entity_surface_text(row["input"])
        text = normalized(entity_text)
        key = document_key(row)
        if (key is None or key in eval_keys or source_url_key(row) in eval_urls or
                row.get("source_group") in eval_groups):
            reasons[name].add("source")
        if any(phrase_in_text(value, text, entity_text)
               for value in surfaces["person"]):
            reasons[name].add("person surface")
        if any(phrase_in_text(value, text, entity_text) for value in
               set(surfaces["org"]) | set(aliases)):
            reasons[name].add("org referent")
        if address_keys_in_text(row["input"]) & set(addresses):
            reasons[name].add("physical address")
        if any(value.casefold() in contacts["email"] for value in EMAIL.findall(row["input"])):
            reasons[name].add("email")
        if any(re.sub(r"\D", "", value) in contacts["phone"]
               for value in PHONE.findall(row["input"])):
            reasons[name].add("phone")
        if shingles(row["input"], 12) & eval_fragments:
            reasons[name].add("twelve-word prose")
        spans = row.get("expected", row.get("entities"))
        if spans is not None:
            validate_spans(row, spans)
            labeled.append((name, row["input"], spans))
    if labeled:
        for kinds in collisions(gold, labeled, strict_aliases=True).values():
            for kind, matches in kinds.items():
                for origin, _ in matches:
                    reasons[origin].add(f"labeled {kind}")
        for kinds in contact_collisions(gold, labeled).values():
            for kind, matches in kinds.items():
                for origin, value in matches:
                    if kind != "phone" or re.sub(r"\D", "", value) != "711":
                        reasons[origin].add(f"labeled {kind}")
    return {name: sorted(hits) for name, hits in reasons.items() if hits}


def main():
    """Fail closed when any blind passage overlaps reserved evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    packet = rows(args.packet)
    if len({row["name"] for row in packet}) != len(packet):
        raise ValueError("duplicate packet case")
    gold = [row for path in EVALUATION for row in rows(path)]
    hits = heldout_reasons(packet, gold)
    print(f"{len(packet)} cases, {len(hits)} heldout overlaps; "
          f"packet SHA-256 {digest(args.packet)}")
    for name, reasons in sorted(hits.items()):
        print(f"{name}: {', '.join(reasons)}")
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
