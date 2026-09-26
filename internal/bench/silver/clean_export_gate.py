"""Fail-closed private gate for a reviewed, source-qualified detector silver export.

This does not decide source rights or label correctness. It requires explicit,
versioned human decisions for every selected candidate row and asks the trainer
to check the exact staged export before an output can be written.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE = ROOT / "data/interim/silver/qa-dedup-20260926/train.jsonl"
DEFAULT_LINEAGE = ROOT / "data/interim/silver/qa-dedup-20260926/source-lineage-pending-v2.jsonl"
DEV_GOLD_SHA256 = "e283b6db04469fc67945f0204a597fe25da092fa2f91062073325f1213028887"
KINDS = {"person", "org", "address"}
BAD_VALUES = {"", "pending", "unknown", "todo", "tbd", "n/a", "none"}
PUBLISHER_ID = re.compile(r"^[a-z][a-z0-9_-]*(?::[a-z0-9][a-z0-9._-]*)+$")
HOST_ID = re.compile(r"^[a-z0-9.-]+$")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def rows(path):
    result = []
    for line_no, line in enumerate(path.read_bytes().splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return result


def filled(value):
    return isinstance(value, str) and value.strip().lower() not in BAD_VALUES


def url(value):
    return filled(value) and urlsplit(value).scheme in {"https", "http"} and bool(urlsplit(value).hostname)


def key(row):
    return (row.get("source"), row.get("id"))


def eval_gold_identity_text(row):
    """Read either gate-shaped or trainer-evaluator gold, rejecting mixed aliases."""
    if not isinstance(row, dict):
        raise ValueError("sealed evaluation gold row is not an object")
    native = "name" in row or "input" in row
    legacy = "id" in row or "text" in row
    if not native and not legacy:
        raise ValueError("sealed evaluation gold row missing identity and text")
    if native and (not filled(row.get("name")) or not isinstance(row.get("input"), str)):
        raise ValueError("sealed evaluation gold row missing name/input")
    if legacy and (not filled(row.get("id")) or not isinstance(row.get("text"), str)):
        raise ValueError("sealed evaluation gold row missing id/text")
    if native and legacy and (row["name"] != row["id"] or row["input"] != row["text"]):
        raise ValueError("sealed evaluation gold aliases disagree")
    return (row["name"], row["input"]) if native else (row["id"], row["text"])


def publisher_ids(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, str) and PUBLISHER_ID.fullmatch(v) for v in value)
            and len(value) == len(set(value)))


def normalized_host(value):
    if not isinstance(value, str):
        return None
    host = value.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host if HOST_ID.fullmatch(host) and "." in host else None


def load_eval_hosts(path, errors):
    if path is None:
        errors.append("missing sealed development host manifest")
        return None, None
    manifest = json.loads(path.read_text())
    if (manifest.get("status") != "sealed" or not filled(manifest.get("version"))
            or not filled(manifest.get("reviewer")) or not filled(manifest.get("reviewed_at"))):
        errors.append("development host manifest needs sealed status, version, reviewer, date")
    gold_path = manifest.get("gold_path")
    if manifest.get("gold_sha256") != DEV_GOLD_SHA256:
        errors.append("development host gold identity differs from pinned gold")
        return None, None
    if not isinstance(gold_path, str) or not Path(gold_path).is_file():
        errors.append("development host gold path missing")
        return None, None
    gold_bytes = Path(gold_path).read_bytes()
    if digest(gold_bytes) != manifest.get("gold_sha256"):
        errors.append("development host gold SHA mismatch")
        return None, None
    gold = rows(Path(gold_path))
    if not gold:
        errors.append("development host gold is empty")
        return None, None
    hosts = set()
    hashes = set()
    identities = set()
    for row in gold:
        try:
            identity, value = eval_gold_identity_text(row)
        except ValueError as exc:
            errors.append(str(exc))
            return None, None
        if identity in identities:
            errors.append(f"development gold duplicate identity {identity}")
            return None, None
        identities.add(identity)
        source_host = row.get("source_host")
        if not isinstance(source_host, str) or source_host.count("|") != 1:
            errors.append(f"development gold missing source_host {identity}")
            return None, None
        country, host = source_host.split("|", 1)
        host = normalized_host(host)
        if country != row.get("country") or host is None:
            errors.append(f"development gold invalid source_host {identity}")
            return None, None
        hosts.add(host)
        hashes.add(digest(value.encode("utf-8")))
    if len(gold) != manifest.get("rows"):
        errors.append("development host gold row count mismatch")
    return hosts, hashes


def indexed(items, name, errors):
    out = {}
    for item in items:
        k = key(item)
        if not all(filled(x) for x in k):
            errors.append(f"{name}: missing source/id")
        elif k in out:
            errors.append(f"{name}: duplicate source/id {k}")
        else:
            out[k] = item
    return out


def validate_spans(text, spans):
    if not isinstance(spans, list):
        return "entities must be a list"
    raw = text.encode("utf-8")
    last = 0
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"kind", "start", "end"}:
            return "entity must contain only kind/start/end"
        kind, start, end = span["kind"], span["start"], span["end"]
        if kind not in KINDS or type(start) is not int or type(end) is not int:
            return "invalid entity kind or offset type"
        if not last <= start < end <= len(raw):
            return "out-of-order, overlapping, or out-of-range entity"
        try:
            surface = raw[start:end].decode("utf-8")
        except UnicodeDecodeError:
            return "entity cuts a UTF-8 character"
        if surface != surface.strip():
            return "entity has surrounding whitespace"
        last = end
    return None


def validate_inputs(candidate_path, lineage_path, qualification_path, labels_path, eval_path,
                    root=ROOT, selection_path=None, eval_hosts_path=None):
    errors = []
    counts = Counter()
    candidate_bytes = candidate_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()
    manifest_path = lineage_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if digest(candidate_bytes) != manifest.get("candidate_sha256"):
        errors.append("candidate SHA differs from lineage manifest")
    if digest(lineage_bytes) != manifest.get("lineage_sha256"):
        errors.append("lineage SHA differs from manifest")
    candidate = rows(candidate_path)
    lineage = rows(lineage_path)
    if len(candidate) != manifest.get("rows") or len(lineage) != len(candidate):
        errors.append("candidate/lineage row count differs from manifest")
    qualifications = rows(qualification_path) if qualification_path else []
    reviewed = rows(labels_path) if labels_path else []
    qmap = indexed(qualifications, "qualification", errors)
    lmap = indexed(reviewed, "reviewed labels", errors)
    cmap = indexed(candidate, "candidate", errors)
    imap = indexed(lineage, "lineage", errors)
    if set(cmap) != set(imap):
        errors.append("candidate/lineage key sets differ")
    if set(qmap) - set(cmap):
        errors.append("qualification has keys outside candidate")
    if set(lmap) - set(cmap):
        errors.append("reviewed labels have keys outside candidate")
    selected = set(cmap)
    selection_sha = None
    if selection_path:
        selection_sha = digest(selection_path.read_bytes())
        selection = json.loads(selection_path.read_text())
        if (selection.get("candidate_sha256") != digest(candidate_bytes)
                or selection.get("lineage_sha256") != digest(lineage_bytes)):
            errors.append("selection candidate/lineage SHA mismatch")
        for field in ("version", "reviewer", "reason"):
            if not filled(selection.get(field)):
                errors.append(f"selection missing {field}")
        selected = set()
        choice_rows = selection.get("selected")
        if not isinstance(choice_rows, list) or not choice_rows:
            errors.append("selection needs at least one selected row")
            choice_rows = []
        for choice in choice_rows:
            k = key(choice)
            if k in selected:
                errors.append(f"selection duplicate source/id {k}")
            selected.add(k)
            doc = cmap.get(k)
            if doc is None:
                errors.append(f"selection source/id absent from candidate {k}")
            elif choice.get("candidate_text_sha256") != digest(doc["text"].encode("utf-8")):
                errors.append(f"selection text SHA mismatch {k}")
        if set(qmap) - selected:
            errors.append("qualification has keys outside selection")
        if set(lmap) - selected:
            errors.append("reviewed labels have keys outside selection")
    eval_publishers = None
    eval_text_hashes = None
    eval_hosts, dev_text_hashes = load_eval_hosts(eval_hosts_path, errors)
    if eval_path:
        ev = json.loads(eval_path.read_text())
        if (ev.get("status") != "sealed" or not filled(ev.get("version"))
                or not filled(ev.get("reviewer")) or not filled(ev.get("reviewed_at"))):
            errors.append("evaluation publisher manifest needs sealed status, version, reviewer, date")
        gold_path = ev.get("gold_path")
        gold = []
        if not isinstance(gold_path, str) or not Path(gold_path).is_file():
            errors.append("sealed evaluation gold path missing")
        elif digest(Path(gold_path).read_bytes()) != ev.get("gold_sha256"):
            errors.append("sealed evaluation gold SHA mismatch")
        else:
            gold = rows(Path(gold_path))
            try:
                gold_fields = [eval_gold_identity_text(g) for g in gold]
            except ValueError as exc:
                gold_fields = []
                errors.append(str(exc))
            if gold_fields:
                gold_hashes = [digest(text.encode("utf-8")) for _, text in gold_fields]
                if len(set(gold_hashes)) != len(gold_hashes):
                    errors.append("sealed evaluation gold has duplicate exact text")
                eval_text_hashes = set(gold_hashes)
        index_path = ev.get("publisher_index_path")
        if not isinstance(index_path, str) or not Path(index_path).is_file():
            errors.append("sealed evaluation publisher index missing")
        elif digest(Path(index_path).read_bytes()) != ev.get("publisher_index_sha256"):
            errors.append("sealed evaluation publisher index SHA mismatch")
        else:
            index = rows(Path(index_path))
            gold_ids = [identity for identity, _ in gold_fields] if gold else []
            index_ids = [i.get("id") for i in index]
            if (not gold or any(not filled(x) for x in gold_ids + index_ids)
                    or len(set(gold_ids)) != len(gold_ids)
                    or set(gold_ids) != set(index_ids) or len(index_ids) != len(gold_ids)
                    or any(not publisher_ids(i.get("publisher_groups"))
                           or i.get("publisher_group") not in i["publisher_groups"]
                           for i in index)):
                errors.append("sealed evaluation publisher index incomplete or invalid")
            else:
                eval_publishers = {group for i in index for group in i["publisher_groups"]}
    else:
        errors.append("missing sealed evaluation publisher manifest")
    output = []
    seen_text = set()
    for line_no, (doc, info) in enumerate(zip(candidate, lineage), 1):
        k = key(doc)
        prefix = f"candidate line {line_no} {k}: "
        if key(info) != k or info.get("candidate_line") != line_no or info.get("country") != doc.get("country"):
            errors.append(prefix + "lineage identity/order mismatch")
            continue
        text = doc.get("text")
        if not isinstance(text, str):
            errors.append(prefix + "missing text")
            continue
        text_hash = digest(text.encode("utf-8"))
        if text_hash != info.get("candidate_text_sha256"):
            errors.append(prefix + "candidate text SHA mismatch")
        if text_hash in seen_text:
            errors.append(prefix + "duplicate exact text")
        seen_text.add(text_hash)
        if k not in selected:
            counts["out_of_scope"] += 1
            continue
        counts["selected"] += 1
        if eval_text_hashes is not None and text_hash in eval_text_hashes:
            errors.append(prefix + "exact text overlaps sealed evaluation gold")
        if dev_text_hashes is not None and text_hash in dev_text_hashes:
            errors.append(prefix + "exact text overlaps development gold")
        candidate_host = normalized_host(info.get("url_host"))
        if candidate_host is None:
            errors.append(prefix + "invalid candidate URL host")
        if eval_hosts is not None and candidate_host in eval_hosts:
            errors.append(prefix + "development host overlap")
        raw_path_value = info.get("raw_path")
        raw_path = root / raw_path_value if isinstance(raw_path_value, str) else None
        if raw_path is None or not raw_path.is_file() or not raw_path.resolve().is_relative_to((root / "data/raw/silver").resolve()):
            errors.append(prefix + "raw path missing or outside raw silver tree")
        else:
            raw_bytes = raw_path.read_bytes()
            if digest(raw_bytes) != info.get("raw_file_sha256"):
                errors.append(prefix + "raw file SHA mismatch")
            try:
                raw = json.loads(raw_bytes)
                raw_text = raw["text"].encode("utf-8")
                start, end = info["raw_excerpt_start_byte"], info["raw_excerpt_end_byte"]
                if (key(raw) != k or digest(raw_text) != info.get("raw_text_sha256")
                        or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(raw_text)
                        or raw_text[start:end] != text.encode("utf-8")
                        or raw.get("url") != info.get("raw_url")
                        or urlsplit(raw.get("url", "")).hostname != info.get("url_host")
                        or (raw.get("country") is not None and raw.get("country") != doc.get("country"))):
                    errors.append(prefix + "raw identity/text/excerpt/URL mismatch")
            except (ValueError, KeyError, TypeError):
                errors.append(prefix + "invalid raw record or excerpt")
        decision = qmap.get(k)
        if decision is None:
            counts["missing_qualification"] += 1
            continue
        if (decision.get("candidate_text_sha256") != text_hash
                or decision.get("raw_file_sha256") != info.get("raw_file_sha256")
                or decision.get("raw_url") != info.get("raw_url")):
            errors.append(prefix + "qualification is not bound to candidate/raw version")
        status = decision.get("source_eligibility")
        if status == "excluded":
            if not filled(decision.get("exclusion_reason")) or not filled(decision.get("reviewer")):
                errors.append(prefix + "excluded row lacks reason/reviewer")
            counts["excluded"] += 1
            continue
        if status != "approved":
            counts["pending_or_held"] += 1
            continue
        for field in ("publisher_group", "publisher_evidence_url", "qualification_evidence_url",
                      "rights_scope", "reviewer", "reviewed_at", "decision_version"):
            value = decision.get(field)
            if (not url(value)) if field.endswith("_url") else (not filled(value)):
                errors.append(prefix + f"missing or invalid {field}")
        if not publisher_ids(decision.get("publisher_groups")):
            errors.append(prefix + "missing or invalid linked publisher_groups")
        elif decision.get("publisher_group") not in decision["publisher_groups"]:
            errors.append(prefix + "primary publisher_group absent from linked publisher_groups")
        if decision.get("publisher_group") == doc.get("source") and doc.get("source") == "govuk":
            errors.append(prefix + "GOV.UK source bucket is not an actual publisher group")
        if (eval_publishers is not None and publisher_ids(decision.get("publisher_groups"))
                and eval_publishers.intersection(decision["publisher_groups"])):
            errors.append(prefix + "evaluation publisher overlap")
        label = lmap.get(k)
        if label is None:
            counts["missing_reviewed_labels"] += 1
            continue
        if label.get("candidate_text_sha256") != text_hash:
            errors.append(prefix + "reviewed label text SHA mismatch")
        for field in ("label_version", "reviewer", "review_evidence"):
            if not filled(label.get(field)):
                errors.append(prefix + f"missing {field}")
        spans = label.get("entities")
        if label.get("entities_sha256") != digest(canonical(spans)):
            errors.append(prefix + "reviewed entity SHA mismatch")
        span_issue = validate_spans(text, spans)
        if span_issue:
            errors.append(prefix + span_issue)
            continue
        output.append({"id": doc["id"], "source": doc["source"], "country": doc["country"],
                       "text": text, "entities": spans})
        counts["approved_with_labels"] += 1
    if len(candidate) != len(lineage):
        errors.append("candidate/lineage length mismatch")
    if not output:
        errors.append("no approved reviewed rows")
    for category in ("missing_qualification", "pending_or_held", "missing_reviewed_labels"):
        if counts[category]:
            errors.append(f"{counts[category]} {category} rows")
    return output, {"candidate_rows": len(candidate), "counts": dict(counts),
                    "errors": errors[:25], "error_count": len(errors),
                    "candidate_sha256": digest(candidate_bytes),
                    "lineage_sha256": digest(lineage_bytes),
                    "selection_sha256": selection_sha}


def preflight(output, config, trainer):
    with tempfile.TemporaryDirectory(prefix="tessera-silver-gate-") as tmp:
        tmp = Path(tmp)
        staged = tmp / "train.jsonl"
        staged.write_bytes(b"".join(canonical(row) + b"\n" for row in output))
        config_text = config.read_text()
        replacement = 'silver = [' + json.dumps(str(staged)) + ']'
        config_text, changes = re.subn(r"(?m)^silver\s*=\s*\[[^\n]*\]$", replacement, config_text)
        if changes != 1:
            raise ValueError("detector config needs exactly one single-line silver array")
        staged_config = tmp / "config.toml"
        staged_config.write_text(config_text)
        result = subprocess.run([str(trainer), "check-silver", "--config", str(staged_config)],
                                cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise ValueError("trainer check-silver failed: " + (result.stderr or result.stdout)[:4000])
        counts = json.loads(result.stdout)
        if counts.get("unreachable_spans") != 0 or counts.get("documents") != len(output):
            raise ValueError("trainer preflight count/reachability mismatch")
        return staged.read_bytes(), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--eval-publishers", type=Path)
    parser.add_argument("--eval-hosts", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output, report = validate_inputs(args.candidate, args.lineage, args.qualification,
                                     args.labels, args.eval_publishers,
                                     selection_path=args.selection, eval_hosts_path=args.eval_hosts)
    if not report["errors"]:
        if not args.config or not args.trainer:
            report["errors"].append("config and trainer executable required for runtime preflight")
            report["error_count"] += 1
        else:
            try:
                data, counts = preflight(output, args.config, args.trainer)
                report["runtime_counts"] = counts
                report["export_sha256"] = digest(data)
                if args.output:
                    allowed = ROOT / "data/interim/silver"
                    if (args.output.suffix != ".jsonl"
                            or not args.output.resolve().is_relative_to(allowed.resolve())):
                        raise ValueError("output must be a versioned .jsonl under private data/interim/silver")
                    manifest_path = args.output.with_suffix(".manifest.json")
                    if args.output.exists() or manifest_path.exists():
                        raise FileExistsError("versioned output or manifest already exists")
                    manifest = {**report, "status": "source-qualified reviewed silver export",
                                "qualification_sha256": digest(args.qualification.read_bytes()),
                                "labels_sha256": digest(args.labels.read_bytes()),
                                "eval_publishers_sha256": digest(args.eval_publishers.read_bytes()),
                                "eval_hosts_sha256": digest(args.eval_hosts.read_bytes()),
                                "trainer_sha256": digest(args.trainer.read_bytes()),
                                "config_sha256": digest(args.config.read_bytes())}
                    with args.output.open("xb") as handle:
                        handle.write(data)
                    try:
                        with manifest_path.open("x") as handle:
                            handle.write(json.dumps(manifest, indent=2) + "\n")
                    except OSError:
                        args.output.unlink()
                        raise
            except (ValueError, OSError) as exc:
                report["errors"].append(str(exc))
                report["error_count"] += 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
