"""Tests for the silver labelling helpers: `python -m pytest bench/silver`."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from label import OVERLAP, STRIDE, merge, plausible, to_bytes, windows  # noqa: E402
from score_review import score  # noqa: E402


def test_code_points_become_utf8_bytes():
    assert to_bytes("ნინო x", 5) == len("ნინო ".encode("utf-8")) == 13


def test_merge_keeps_the_earlier_of_overlapping_spans():
    spans = [
        {"kind": "org", "start": 5, "end": 20},
        {"kind": "person", "start": 0, "end": 10},
        {"kind": "address", "start": 30, "end": 40},
    ]
    assert [e["kind"] for e in merge(spans)] == ["person", "address"]


def test_owned_ranges_tile_the_text_with_context():
    text = ("word " * 1000).strip()
    ws = windows(text)
    assert len(ws) > 1
    assert ws[0][2] == 0 and ws[-1][3] == len(text)
    for (_, _, _, own_to), (_, _, own_from, _) in zip(ws, ws[1:]):
        assert own_to == own_from
    for start, window, own_from, own_to in ws:
        assert text[start:start + len(window)] == window
        assert len(window) <= STRIDE + 100 + OVERLAP
        end = start + len(window)
        assert end - own_to >= min(OVERLAP // 2, len(text) - own_to)
        assert own_from == 0 or own_from - start >= OVERLAP // 2


def test_short_text_is_one_window():
    assert windows("short text") == [(0, "short text", 0, 10)]


def test_score_counts_boundary_errors_against_both_measures():
    rows = [{"kind": "person", "verdict": v} for v in
            ["correct"] * 8 + ["wrong_boundary", "not_entity", "missing"]]
    rows += [{"kind": k, "verdict": "correct"} for k in ["org", "address"]]
    s = score(rows)["person"]
    assert s["precision"] == 0.8
    assert s["recall_estimate"] == 0.8
    assert not s["accepted"]


def test_implausible_spans_are_dropped():
    assert not plausible("0303 444 5004")
    assert not plausible("government")
    assert plausible("Mansfield District Council")
    assert plausible("ნინო ბერიძე")
    assert plausible("MCA")


def test_gpo_character_codes_become_characters():
    from fetch_federal_register import gpo_entities
    assert gpo_entities("Mu[ntilde]oz, L[imacr]hu[revaps]e") == "Muñoz, Līhuʻe"
    assert gpo_entities("Postal Service[supreg] [sic] [reserved]") == "Postal Service® [sic] [reserved]"
