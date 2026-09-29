"""Guard exact UTF-8 spans before blind reviews become training labels."""

import unittest

from build_r23_reviewed_org import checked_labels
from silver_dedupe import NearDuplicateIndex, NearTextIndex


class BlindReviewTests(unittest.TestCase):
    def test_multibyte_span_uses_byte_offsets(self):
        text = "José works at NASA."
        labels = checked_labels(text, {"name": "one", "entities": [
            {"kind": "person", "start": 0, "end": 5, "text": "José"},
            {"kind": "org", "start": 15, "end": 19, "text": "NASA"},
        ]})
        self.assertEqual(labels, [("person", 0, 5), ("org", 15, 19)])

    def test_wrong_text_or_overlap_is_rejected(self):
        text = "Office of Health"
        wrong = {"name": "one", "entities": [
            {"kind": "org", "start": 0, "end": 6, "text": "Agency"},
        ]}
        with self.assertRaises(ValueError):
            checked_labels(text, wrong)
        overlapping = {"name": "one", "entities": [
            {"kind": "org", "start": 0, "end": 6, "text": "Office"},
            {"kind": "org", "start": 0, "end": 16, "text": text},
        ]}
        with self.assertRaises(ValueError):
            checked_labels(text, overlapping)

    def test_repeated_notice_with_same_labels_is_one_example(self):
        def row(source_id, text, label):
            start = text.index(label)
            return {"id": source_id, "text": text,
                    "entities": [{"kind": "org", "start": start,
                                  "end": start + len(label)}]}

        first = row("one", "The Council will hold its public meeting on Monday at the "
                    "same location. The Council will discuss permit applications.", "Council")
        repeated = row("two", "The Council will hold its public meeting on Monday at the "
                       "same location; the Council will discuss permit applications.", "Council")
        different = row("three", "The Agency will hold its public meeting on Monday at the "
                        "same location. The Agency will discuss permit applications.", "Agency")
        index = NearDuplicateIndex()
        index.add(first)
        self.assertEqual(index.prior(repeated), "one")
        self.assertIsNone(index.prior(different))

    def test_near_evaluation_text_is_detected_without_matching_labels(self):
        index = NearTextIndex()
        index.add("training", "The agency accepts written comments by mail at the public "
                  "office through the end of the month.")
        self.assertEqual(index.prior("The agency accepts written comments by mail at the public "
                                     "office through the end of this month."), "training")
        self.assertIsNone(index.prior("The project reviewed an unrelated set of local roads "
                                      "and buildings in the county last week."))


if __name__ == "__main__":
    unittest.main()
