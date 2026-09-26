"""Verify pinned HTML-to-text evidence, without deciding rights or labels.

The expected file hashes and source identity must come from a separately reviewed
manifest. Supplying values computed from the same untrusted files defeats that
part of the check. Source-specific projection profiles make text derivation
replayable; the SA profile also checks that no office/address was omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PUBLISHER_ID = re.compile(r"[a-z][a-z0-9_-]*(?::[a-z0-9][a-z0-9._-]*)+\Z")
PARAGRAPH = re.compile(rb"<p(?:\s+[^>]*)?>.*?</p>", re.DOTALL)
TAG = re.compile(rb"<[^>]+>")
SA_START = "<h5><b>სტრუქტურული ერთეულები:</b></h5>".encode("utf-8")
SA_END = "<h5><b>სამუშაო საათები:".encode("utf-8")


class ProvenanceError(ValueError):
    """The frozen raw HTML, mapping, or projected text failed verification."""


@dataclass(frozen=True)
class VerifiedProvenance:
    profile: str
    rows: int
    raw_sha256: str
    mapping_sha256: str
    packet_sha256: str
    source_url: str
    publisher_group: str


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def load_jsonl(raw: bytes, name: str) -> list[dict]:
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        require(bool(line.strip()), f"{name}:{line_no}: empty line")
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProvenanceError(f"{name}:{line_no}: invalid JSON") from exc
        require(isinstance(item, dict), f"{name}:{line_no}: object required")
        rows.append(item)
    require(bool(rows), f"{name}: no rows")
    return rows


def paragraph_text(raw_paragraph: bytes) -> str:
    require(bool(PARAGRAPH.fullmatch(raw_paragraph)), "source range is not one complete <p> paragraph")
    # The stored SA projection used HTML-unescape and edge trim after tag removal.
    return html.unescape(TAG.sub(b"", raw_paragraph).decode("utf-8")).strip()


def selected_paragraphs(raw: bytes, start: int = 0, end: int | None = None) -> dict[tuple[int, int], bytes]:
    if end is None:
        end = len(raw)
    return {(start + match.start(), start + match.end()): match.group()
            for match in PARAGRAPH.finditer(raw[start:end])}


def canonical_sa_rows(raw: bytes) -> list[dict]:
    require(raw.count(SA_START) == 1 and raw.count(SA_END) == 1,
            "SA section anchors missing or duplicated")
    start = raw.index(SA_START) + len(SA_START)
    end = raw.index(SA_END, start)
    paragraphs = selected_paragraphs(raw, start, end)
    offices = []
    current = None
    historical_removed = 0
    for (begin, finish), p in paragraphs.items():
        text = paragraph_text(p)
        if not text:
            continue
        evidence = {"byte_range": [begin, finish], "raw_sha256": digest(p)}
        if p.startswith(b"<p><b>"):
            current = {"heading": text, "addresses": [], "paragraphs": []}
            offices.append(current)
            current["paragraphs"].append({"kind": "heading", **evidence})
        elif text.startswith("მის.:"):
            require(current is not None, "SA address precedes first office heading")
            if " (ყოფილი მისამართი:" in text:
                require(current["heading"] == "რუსთავის ცენტრალური დეპარტამენტი",
                        "former-address clause outside Rustavi office")
                require(text.count(" (ყოფილი მისამართი:") == 1 and text.endswith(")"),
                        "unexpected former-address clause shape")
                text = text.split(" (ყოფილი მისამართი:", 1)[0]
                historical_removed += 1
            current["addresses"].append(text)
            current["paragraphs"].append({"kind": "current_address", **evidence})
    require(historical_removed == 1, "expected one removed Rustavi former-address clause")
    require(all(o["addresses"] for o in offices), "SA office without current address")
    require(len(offices) == len({o["heading"] for o in offices}), "duplicate SA office heading")
    return [{
        "row_id": f"SAE{i:03d}",
        "selected_text": "\n".join([o["heading"], *o["addresses"]]),
        "source_paragraphs": o["paragraphs"],
    } for i, o in enumerate(offices, 1)]


def canonical_generic_rows(raw: bytes, mapping: list[dict]) -> list[dict]:
    """Replay mapped complete HTML paragraphs; completeness is profile-specific."""
    paragraphs = selected_paragraphs(raw)
    result = []
    for row in mapping:
        pieces = []
        evidence = row.get("source_paragraphs")
        require(isinstance(evidence, list) and bool(evidence), "generic row needs source_paragraphs")
        for p in evidence:
            require(isinstance(p, dict) and p.get("kind") == "text",
                    "generic paragraph must have kind=text")
            byte_range = p.get("byte_range")
            require(isinstance(byte_range, list) and len(byte_range) == 2
                    and all(type(v) is int for v in byte_range), "invalid paragraph byte range")
            piece = paragraphs.get(tuple(byte_range))
            require(piece is not None, "mapped range is not an exact HTML paragraph")
            pieces.append(paragraph_text(piece))
        result.append({"row_id": row.get("row_id"), "selected_text": "\n".join(pieces),
                       "source_paragraphs": evidence})
    return result


def verify_packet(*, raw_path: Path, mapping_path: Path, packet_path: Path,
                  expected_raw_sha256: str, expected_mapping_sha256: str,
                  expected_packet_sha256: str, expected_source_url: str,
                  expected_publisher_group: str, profile: str) -> VerifiedProvenance:
    for field, value in (("raw SHA", expected_raw_sha256), ("mapping SHA", expected_mapping_sha256),
                         ("packet SHA", expected_packet_sha256)):
        require(isinstance(value, str) and bool(SHA256.fullmatch(value)), f"invalid expected {field}")
    source = urlsplit(expected_source_url) if isinstance(expected_source_url, str) else None
    require(source is not None and source.scheme == "https" and bool(source.hostname)
            and not source.username and not source.password,
            "expected source URL must be HTTPS with a host")
    require(isinstance(expected_publisher_group, str)
            and bool(PUBLISHER_ID.fullmatch(expected_publisher_group)),
            "expected publisher group is not a canonical ID")
    raw = raw_path.read_bytes()
    mapping_bytes = mapping_path.read_bytes()
    packet_bytes = packet_path.read_bytes()
    require(digest(raw) == expected_raw_sha256, "raw HTML SHA mismatch")
    require(digest(mapping_bytes) == expected_mapping_sha256, "mapping SHA mismatch")
    require(digest(packet_bytes) == expected_packet_sha256, "packet SHA mismatch")
    mapping = load_jsonl(mapping_bytes, "mapping")
    packet = load_jsonl(packet_bytes, "packet")
    require(len(mapping) == len(packet), "mapping/packet row count mismatch")
    if profile == "sa_offices_v1":
        canonical = canonical_sa_rows(raw)
    elif profile == "paragraphs_v1":
        canonical = canonical_generic_rows(raw, mapping)
    else:
        raise ProvenanceError(f"unsupported projection profile: {profile}")
    require(len(canonical) == len(mapping), "source/packet row count mismatch")
    source_paragraphs = selected_paragraphs(raw)
    seen_ids = set()
    for index, (expected, map_row, text_row) in enumerate(zip(canonical, mapping, packet), 1):
        row_id = expected["row_id"]
        require(isinstance(row_id, str) and row_id and row_id not in seen_ids,
                f"row {index}: missing/duplicate row ID")
        seen_ids.add(row_id)
        require(map_row.get("row_id") == text_row.get("row_id") == row_id,
                f"row {index}: row ID/order mismatch")
        require(map_row.get("source_url") == expected_source_url,
                f"{row_id}: source URL mismatch")
        require(map_row.get("publisher_group") == expected_publisher_group,
                f"{row_id}: publisher group mismatch")
        require(map_row.get("raw_html_sha256") == expected_raw_sha256,
                f"{row_id}: mapped raw SHA mismatch")
        text = text_row.get("selected_text")
        require(isinstance(text, str) and text == expected["selected_text"],
                f"{row_id}: parsed text projection mismatch")
        require(map_row.get("selected_text_sha256") == digest(text.encode("utf-8")),
                f"{row_id}: selected-text SHA mismatch")
        evidence = map_row.get("source_paragraphs")
        require(evidence == expected["source_paragraphs"],
                f"{row_id}: source paragraph sequence/range mismatch")
        for item in evidence:
            begin, finish = item["byte_range"]
            p = source_paragraphs.get((begin, finish))
            require(p is not None and digest(p) == item["raw_sha256"],
                    f"{row_id}: source paragraph byte/hash mismatch")
    return VerifiedProvenance(profile, len(packet), expected_raw_sha256,
                          expected_mapping_sha256, expected_packet_sha256,
                          expected_source_url, expected_publisher_group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "mapping", "packet"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--publisher-group", required=True)
    parser.add_argument("--profile", choices=("sa_offices_v1", "paragraphs_v1"), required=True)
    args = parser.parse_args()
    result = verify_packet(
        raw_path=args.raw, mapping_path=args.mapping, packet_path=args.packet,
        expected_raw_sha256=args.raw_sha256, expected_mapping_sha256=args.mapping_sha256,
        expected_packet_sha256=args.packet_sha256, expected_source_url=args.source_url,
        expected_publisher_group=args.publisher_group, profile=args.profile)
    print(json.dumps(result.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
