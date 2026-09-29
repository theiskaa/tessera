"""Check that audited source corrections survive the strict silver build."""

import json
import unittest

from build_silver import OUT, STRICT_OUT, apply_strict_errata


class StrictLabelErrataTests(unittest.TestCase):
    def test_audited_corrections_apply_to_original_source_rows(self):
        wanted = {"2024-29016", "2024-30490", "2024-31221"}
        original = [json.loads(line) for line in OUT.read_text().splitlines()
                    if json.loads(line)["id"] in wanted]
        self.assertEqual({row["id"] for row in original}, wanted)
        revised = apply_strict_errata(original)
        built = {row["id"]: row for row in
                 (json.loads(line) for line in STRICT_OUT.read_text().splitlines())
                 if row["id"] in wanted}
        self.assertEqual(set(built), {"2024-30490"})
        for row in revised:
            if row["id"] in built:
                self.assertEqual(row["entities"], built[row["id"]]["entities"])
        by_id = {row["id"]: row for row in revised}
        def org_texts(row):
            source = row["text"].encode()
            return [source[span["start"]:span["end"]].decode()
                    for span in row["entities"] if span["kind"] == "org"]
        self.assertIn("Program Administration Office", org_texts(by_id["2024-29016"]))
        self.assertNotIn("NITRD", org_texts(by_id["2024-31221"]))
        self.assertEqual(org_texts(by_id["2024-30490"]).count("Privacy Office"), 2)


if __name__ == "__main__":
    unittest.main()
