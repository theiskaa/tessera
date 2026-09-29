"""Keep development corrections limited to the reviewed cases and spans."""

import json
import unittest

from build_corrected_dev import corrected_rows


class CorrectedDevTests(unittest.TestCase):
    def test_only_reviewed_labels_change(self):
        corrected, base, errata = corrected_rows()
        original = [json.loads(line) for line in base.splitlines()]
        self.assertEqual(len(corrected), 196)
        self.assertEqual([row["name"] for row in corrected],
                         [row["name"] for row in original])
        changed = set()
        for before, after in zip(original, corrected):
            self.assertEqual(before["input"], after["input"])
            if before["expected"] != after["expected"]:
                changed.add(after["name"])
        self.assertEqual(changed, {row["name"] for row in errata["corrections"]})


if __name__ == "__main__":
    unittest.main()
