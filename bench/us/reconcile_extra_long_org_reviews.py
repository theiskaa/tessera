"""Accept extra-long notice labels only when two blind reviews agree on a whole row."""

import collections

from build_contact_snippets import lines
from reconcile_long_org_reviews import reviewed_spans
from screen_2024_2025_extra_long_org_v1 import OUT as PACKET
from screen_2024_2025_extra_long_org_v1 import ROOT


REVIEWS = tuple(ROOT / f"data/interim/review/us-extra-long-org-review-{label}-v1.jsonl"
                for label in ("a", "b", "c"))


def reconciled_rows_for(packet_path, review_paths):
    """Return complete agreed labels and keep unresolved rows out of training."""
    if len(review_paths) != 3:
        raise ValueError("three independent reviews are required")
    packet = lines(packet_path)
    reviews = [lines(path) for path in review_paths]
    names = [row["name"] for row in packet]
    if len(names) != len(set(names)) or any(
            [row["name"] for row in review] != names for review in reviews):
        raise ValueError("extra-long blind reviews differ from the packet order")
    resolved = []
    unresolved = []
    decisions = collections.Counter()
    for index, source in enumerate(packet):
        spans = []
        for review in reviews:
            row = review[index]
            if {key: value for key, value in row.items() if key != "expected"} != source:
                raise ValueError(f"blind review changed source metadata: {source['name']}")
            spans.append(reviewed_spans(source, row))
        if spans[0] == spans[1]:
            chosen = spans[0]
            decisions["first_review_agreement"] += 1
        elif spans[0] == spans[2]:
            chosen = spans[0]
            decisions["third_agreed_a"] += 1
        elif spans[1] == spans[2]:
            chosen = spans[1]
            decisions["third_agreed_b"] += 1
        else:
            unresolved.append(source["name"])
            continue
        resolved.append({**source, "expected": chosen})
    decisions["resolved"] = len(resolved)
    decisions["unresolved"] = len(unresolved)
    return resolved, unresolved, dict(sorted(decisions.items()))


def reconciled_rows():
    """Resolve the frozen extra-long packet with three independent reviews."""
    return reconciled_rows_for(PACKET, REVIEWS)
