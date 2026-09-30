"""Pin separately downloaded official US contact pages for review."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/us-official-contact-pages-v1"
PAGES = {
    "epa-oil-spill-regional-contacts": "https://www.epa.gov/oil-spills-prevention-and-preparedness-regulations/contact-us-about-oil-spill-prevention-and",
    "nara-organization-telephone-list": "https://www.archives.gov/about/organization/telephone-list.html",
    "nara-seattle-contacts": "https://www.archives.gov/seattle/contacts",
    "nara-frc-directors": "https://www.archives.gov/frc/directors",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    """Freeze the page bytes once, then verify them on later runs."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if set(manifest["pages"]) != set(PAGES):
            raise ValueError("official contact page list changed")
        for name, url in PAGES.items():
            item = manifest["pages"][name]
            data = (OUT / f"{name}.html").read_bytes()
            if (item["source_url"] != url or item["sha256"] != digest(data) or
                    item["bytes"] != len(data)):
                raise ValueError(f"official contact source changed: {name}")
        print(f"verified {len(PAGES)} pinned official contact pages")
        return
    captured = {}
    for name, url in PAGES.items():
        data = (OUT / f"{name}.html").read_bytes()
        if len(data) < 1000 or b"<html" not in data[:5000].lower():
            raise ValueError(f"invalid official contact page: {name}")
        captured[name] = {"source_url": url, "bytes": len(data),
                          "sha256": digest(data)}
    manifest = {
        "kind": "us_official_contact_pages_v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "download_command": "curl --fail --location --max-time 30 --silent --show-error --remove-on-error --output <page.html> <source_url>",
        "training_eligible": False,
        "pages": dict(sorted(captured.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"pinned {len(PAGES)} official contact pages for review")


if __name__ == "__main__":
    main()
