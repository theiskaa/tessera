"""Keep full recaptures separate from and comparable to frozen clipped sources."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from recapture_cutoff_sources import recapture


class RecaptureTests(unittest.TestCase):
    def test_full_text_extends_the_old_cutoff_without_changing_old_bytes(self):
        source_id = "2024-00001"
        old_text = "ADDRESSES:\nAn address.\n\nSUPPLEMENTARY INFORMATION:\n" + "x" * 2000
        full_text = old_text + " Additional source text."
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_dir, out = root / "old", root / "full"
            old_dir.mkdir()
            out.mkdir()
            old = {"id": source_id, "url": "https://www.federalregister.gov/example",
                   "text_url": "https://www.govinfo.gov/content/pkg/FR-2024-01-02/html/2024-00001.htm",
                   "text": old_text}
            old_path = old_dir / f"{source_id}.json"
            old_bytes = json.dumps(old).encode()
            old_path.write_bytes(old_bytes)
            with patch("recapture_cutoff_sources.OLD", old_dir):
                with patch("recapture_cutoff_sources.get", return_value=b"html"):
                    with patch("recapture_cutoff_sources.page_text", return_value="page"):
                        with patch("recapture_cutoff_sources.extract",
                                   return_value=full_text):
                            result = recapture(source_id, out)
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(json.loads((out / f"{source_id}.json").read_text())["text"],
                             full_text)
            self.assertGreater(result["full_chars"], result["old_chars"])
            self.assertEqual(list(out.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
