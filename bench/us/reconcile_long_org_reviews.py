"""Resolve independent long-notice reviews using complete-row agreement."""

import collections

from build_contact_snippets import lines
from build_full_notice_gold import validate_spans


BARE_ORG_NOUNS = {
    "Agency", "Board", "Commission", "Committee", "Council", "Department",
    "Institute", "Office", "Service", "Tribe", "University",
}


def reviewed_spans(source, review):
    """Verify review identity and remove bare organizational nouns."""
    if review["name"] != source["name"] or review["input"] != source["input"]:
        raise ValueError(f"review differs from blind source: {source['name']}")
    spans = validate_spans(source["name"], source["input"], review["expected"])
    return [span for span in spans
            if span["kind"] != "org" or span["text"] not in BARE_ORG_NOUNS]


def reconciled_rows(packet_path, a_path, b_path, dispute_path, c_path):
    """Accept only complete labels shared by at least two blind reviewers."""
    packet = lines(packet_path)
    a = lines(a_path)
    b = lines(b_path)
    dispute = lines(dispute_path)
    c = lines(c_path)
    names = [row["name"] for row in packet]
    if (len(names) != len(set(names)) or
            [row["name"] for row in a] != names or
            [row["name"] for row in b] != names):
        raise ValueError("first reviews differ from blind packet order")
    expected_disputes = [row["name"] for row, left, right in zip(packet, a, b)
                         if left["expected"] != right["expected"]]
    if ([row["name"] for row in dispute] != expected_disputes or
            [row["name"] for row in c] != expected_disputes):
        raise ValueError("third review does not cover every first-review dispute")
    third = dict(zip(expected_disputes, c))
    counts = collections.Counter()
    resolved = []
    unresolved = []
    for source, left, right in zip(packet, a, b):
        name = source["name"]
        first = reviewed_spans(source, left)
        second = reviewed_spans(source, right)
        if name in third:
            third_spans = reviewed_spans(source, third[name])
            if first == second:
                spans = first
                counts["first_review_agreement"] += 1
            elif first == third_spans:
                spans = first
                counts["third_agreed_a"] += 1
            elif second == third_spans:
                spans = second
                counts["third_agreed_b"] += 1
            else:
                unresolved.append(name)
                continue
        elif first == second:
            spans = first
            counts["first_review_agreement"] += 1
        else:
            raise ValueError(f"unreviewed disagreement: {name}")
        resolved.append({**source, "expected": spans})
    counts["resolved"] = len(resolved)
    counts["unresolved"] = len(unresolved)
    return resolved, unresolved, dict(sorted(counts.items()))
