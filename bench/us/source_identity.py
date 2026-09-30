"""Map US source rows to the underlying document used for split checks."""

import re
from urllib.parse import urlsplit, urlunsplit


DOCUMENT_NUMBER = re.compile(r"(?<!\d)(\d{4}-\d{5})(?!\d)")


def source_url_key(row):
    """Compare page URLs independently of caller-assigned source group names."""
    url = row.get("source_url") or ""
    parsed = urlsplit(url)
    if not parsed.hostname:
        return None
    host = parsed.hostname.casefold()
    if host.startswith("www."):
        host = host[4:]
    if parsed.port not in (None, 80, 443):
        host += f":{parsed.port}"
    return urlunsplit(("https", host, parsed.path.rstrip("/"), parsed.query, ""))


def document_key(row):
    """Prefer a Federal Register document number, then a reviewed source group or URL."""
    url = row.get("source_url") or ""
    if "federalregister.gov/documents/" in url:
        match = DOCUMENT_NUMBER.search(url)
        if match:
            return f"fr:{match[1]}"
    for field in ("source_id", "id", "name"):
        match = DOCUMENT_NUMBER.search(row.get(field) or "")
        if match:
            return f"fr:{match[1]}"
    if row.get("source_group"):
        return f"group:{row['source_group']}"
    if url:
        parsed = urlsplit(url)
        return "url:" + urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(),
                                    parsed.path.rstrip("/"), "", ""))
    return None
