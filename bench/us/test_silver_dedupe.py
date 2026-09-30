"""Regression tests for repeated training text and evaluation passage matching."""

import unittest

from silver_dedupe import NearDuplicateIndex, NearTextIndex


class SilverDedupeTests(unittest.TestCase):
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
