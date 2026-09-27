"""Copy synthetic detector shards, omitting documents with gold entity-name overlap.

The gold labels are used only to exclude training examples. Their text is never
added to a training shard. Run this after regenerating the synthetic corpus.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/processed/detector"
OUTPUT = ROOT / "data/processed/detector-no-gold-overlap"
GOLD = [
    ROOT / "data/interim/review/gold.jsonl",
    *(ROOT / f"data/interim/review/gold-r{round_no}.jsonl" for round_no in (2, 4, 6, 8)),
    ROOT / "data/interim/review/qa-sa-bounded-20260926/gold-v1.jsonl",
    ROOT / "data/interim/review/qa-ge-sda-confirmation-20260927/gold-v1.jsonl",
]
PINNED_GOLD = {
    "data/interim/review/gold.jsonl": "557487a12fb45499d1f60da929b22717f38a2072eb4660ce5a558815437ef237",
    "data/interim/review/gold-r2.jsonl": "f04591b4122c7e9fe52cb677c364f79038a4ceb8444f28a3eb5193c1c9c93010",
    "data/interim/review/gold-r4.jsonl": "212ba6fa3972bc75f91d8767a401b7e8a13409e3b22b2ed69eae2a59684adafb",
    "data/interim/review/gold-r6.jsonl": "20b1f0c7ab47d5000fbd40753d029434de94bbb3996440a48e5a7ecba0f649f2",
    "data/interim/review/gold-r8.jsonl": "32df57a191c0f8cf9350bb4614196decccaa7da214f9da4d261e2293c5b8ae81",
    "data/interim/review/qa-sa-bounded-20260926/gold-v1.jsonl": "2e1b7b80d68495c3efbc1db33b619fca4ada18eef91421dd75b331879d25f76c",
    "data/interim/review/qa-ge-sda-confirmation-20260927/gold-v1.jsonl": "35f76daa56fb2b6a061e5979d665291fc310677016a804086fde4ddbf39b47a4",
}
PINNED_SOURCE = {
    "train.parquet": "7dfc8af911bdd8aac69c35d35f1ed94e2969972ba2efdce57c69a51ad5336c97",
    "valid.parquet": "aba5b9a1bfc0ea0d3e41899aa313639f0c3958673ccfeb72d6c4a28648f19761",
    "test.parquet": "a8e92ef443ec588237396eacd1148be92101de3c76ee19a8cfbb1044c7c42fde",
}
KINDS = {"person", "org", "address"}
WORD = re.compile(r"\w+", re.UNICODE)
JP_HEAD = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]{2,}(?:省|庁)\Z")
JP_UNIT_END = re.compile(r"(?:部|局|課|室|班|係|本部|センター|委員会)\Z")
JP_UNIT_CHARS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff \u3000・]+\Z")
GE_CASE_SUFFIXES = {"ის", "ში", "ს", "თან", "დან", "ზე", "ით", "მა", "ად", "ისთვის"}
GENERIC_UNIT_ENDINGS = (" დეპარტამენტი", " department", " division", " branch")
GE_EXPANSIONS = {
    "state audit office of georgia": "state audit office",
    "ministry of internal affairs of georgia": "ministry of internal affairs",
    "revenue service of georgia": "revenue service",
    "supreme court of georgia": "supreme court",
}
LEADING_ARTICLE_ALIASES = {
    "the new york times": "new york times",
    "the washington post": "washington post",
}
GE_INFLECTION_ALIASES = {
    "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტი": (
        "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტის",
        "თბილისის სახელმწიფო სამედიცინო უნივერსიტეტში",
    ),
    "ეკონომიკისა და ბიზნესის ფაკულტეტი": (
        "ეკონომიკისა და ბიზნესის ფაკულტეტის",
        "ეკონომიკისა და ბიზნესის ფაკულტეტში",
    ),
    "ილიას სახელმწიფო უნივერსიტეტი": (
        "ილიას სახელმწიფო უნივერსიტეტის",
        "ილიას სახელმწიფო უნივერსიტეტში",
    ),
    "სახელმწიფო სერვისების განვითარების სააგენტო": (
        "სახელმწიფო სერვისების განვითარების სააგენტოს",
        "სახელმწიფო სერვისების განვითარების სააგენტომ",
        "სახელმწიფო სერვისების განვითარების სააგენტოსთან",
    ),
    "საქართველოს ეკონომიკისა და მდგრადი განვითარების სამინისტრო": (
        "საქართველოს ეკონომიკისა და მდგრადი განვითარების სამინისტროსთან",
    ),
    "თიბისი ბანკი": ("თიბისი ბანკთან",),
}
GE_LEGAL_PREFIX_ALIASES = {
    "სს თიბისი ბანკი": "თიბისი ბანკი",
    "სს პროკრედიტ ბანკი": "პროკრედიტ ბანკი",
}
GB_LEGAL_SUFFIX_ALIASES = {
    "royal mail group limited": "royal mail group",
    "punch taverns plc": "punch taverns",
}
ADDITIONAL_ATTESTED_ALIASES = {
    "საქართველოს აგრარული უნივერსიტეტი": "აგრარული უნივერსიტეტი",
    "procredit bank-თან": "procredit bank",
    "european commission-ის": "european commission",
}
KNOWN_PREFIX_ALIASES = {
    "tbilisi state university": ("ivane javakhishvili ",),
    "ეკონომიკისა და მდგრადი განვითარების სამინისტროს": ("საქართველოს ",),
    "ეკონომიკისა და მდგრადი განვითარების სამინისტრომ": ("საქართველოს ",),
    "ხალიკ ბანკი საქართველო": ("სს ",),
}
STRICT_CURATED_PROTECTED = (
    "tbilisi state university",
    "ministry of internal affairs",
    "სახელმწიფო სერვისების განვითარების სააგენტო",
    "state audit office",
    "revenue service",
    "supreme court",
)


def checked_source(path):
    expected = PINNED_SOURCE.get(path.name)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"pinned source changed: {path}")


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def gold_names():
    names = defaultdict(set)
    digest = hashlib.sha256()
    for path in GOLD:
        contents = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        expected = PINNED_GOLD.get(relative)
        if expected is None or hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"pinned gold changed: {path}")
        digest.update(relative.encode())
        digest.update(contents)
        for line in contents.splitlines():
            row = json.loads(line)
            for entity in row["expected"]:
                if entity["kind"] in KINDS:
                    names[entity["kind"]].add(normalized(entity["text"]))
    return names, digest.hexdigest()


def embedded_org_policy(names):
    multiword = defaultdict(list)
    japanese_heads = []
    strict_index = defaultdict(list)
    for name in names["org"]:
        words = name.split()
        first = WORD.match(name)
        if (len(name) >= 16 and len(words) >= 3 and first and len(first.group()) >= 3
                and not name.endswith(GENERIC_UNIT_ENDINGS)):
            multiword[first.group()].append(name)
        if JP_HEAD.fullmatch(name):
            japanese_heads.append(name)
        if len(name) >= 16 and len(words) >= 3:
            first_anywhere = WORD.search(name)
            if first_anywhere:
                strict_index[first_anywhere.group()].append((name, first_anywhere.start()))
    for candidates in multiword.values():
        candidates.sort(key=lambda name: (-len(name), name))
    japanese_heads.sort(key=lambda name: (-len(name), name))
    for candidates in strict_index.values():
        candidates.sort(key=lambda item: (-len(item[0]), item[0]))
    return multiword, japanese_heads, strict_index


def embedded_org_match(value, policy):
    multiword, japanese_heads, _ = policy
    for token in WORD.finditer(value):
        for name in multiword.get(token.group(), ()):
            end = token.start() + len(name)
            if not value.startswith(name, token.start()):
                continue
            suffix = value[end:]
            prefix = value[:token.start()]
            if (prefix == "" or prefix in KNOWN_PREFIX_ALIASES.get(name, ())):
                if (not suffix or suffix in ("'s", "’s") or
                        (suffix[:1] in ("-", "‐", "‑", "–", "—") and
                         suffix[1:] in GE_CASE_SUFFIXES)):
                    return "embedded_multiword_org", name
    for name in japanese_heads:
        if not value.startswith(name) or len(value) == len(name):
            continue
        suffix = value[len(name):].lstrip(" \u3000・/／-—")
        if len(suffix) >= 2 and JP_UNIT_CHARS.fullmatch(suffix) and JP_UNIT_END.search(suffix):
            return "jp_ministry_unit_compound", name
    return None


def word_bounded(value, name, at):
    end = at + len(name)
    return (at >= 0 and value.startswith(name, at)
            and (at == 0 or not WORD.match(value[at - 1]))
            and (end == len(value) or not WORD.match(value[end])))


def long_name_literal(value, policy):
    _, _, strict_index = policy
    for token in WORD.finditer(value):
        for name, offset in strict_index.get(token.group(), ()):
            if word_bounded(value, name, token.start() - offset):
                return name
    return None


def strict_lexical_match(value, names, policy):
    _, japanese_heads, _ = policy
    protected = long_name_literal(value, policy)
    if protected:
        return "strict_broad_protected_literal", protected
    for name in STRICT_CURATED_PROTECTED:
        if name not in names["org"]:
            continue
        for match in re.finditer(re.escape(name), value):
            if word_bounded(value, name, match.start()):
                return "strict_curated_protected_literal", name
    for head in japanese_heads:
        if head in value:
            return "strict_jp_head_literal", head
    sda = "სახელმწიფო სერვისების განვითარების სააგენტო"
    if sda in names["org"]:
        for form in GE_INFLECTION_ALIASES[sda]:
            for match in re.finditer(re.escape(form), value):
                if word_bounded(value, form, match.start()):
                    return "strict_sda_inflection_literal", sda
    return None


def full_text_overlap(text, names, policy):
    matched = strict_lexical_match(normalized(text), names, policy)
    if matched is None:
        return None
    rule, protected = matched
    full_text_rules = {
        "strict_broad_protected_literal": "strict_full_text_broad_literal",
        "strict_curated_protected_literal": "strict_full_text_curated_literal",
        "strict_jp_head_literal": "strict_full_text_jp_head_literal",
        "strict_sda_inflection_literal": "strict_full_text_sda_inflection_literal",
    }
    return full_text_rules[rule], protected


def overlap(value, kind, names, policy):
    value = normalized(value)
    if value in names[kind]:
        return "exact", value
    if kind == "org":
        protected = LEADING_ARTICLE_ALIASES.get(value)
        if protected in names["org"]:
            return "leading_article_alias", protected
        for protected, aliases in GE_INFLECTION_ALIASES.items():
            if protected in names["org"] and value in aliases:
                return "ge_attested_inflection", protected
        protected = GE_LEGAL_PREFIX_ALIASES.get(value)
        if protected in names["org"]:
            return "ge_attested_legal_prefix", protected
        protected = GB_LEGAL_SUFFIX_ALIASES.get(value)
        if protected in names["org"]:
            return "gb_attested_legal_suffix", protected
        protected = ADDITIONAL_ATTESTED_ALIASES.get(value)
        if protected in names["org"]:
            return "additional_attested_alias", protected
        for alias, protected in GE_EXPANSIONS.items():
            if protected in names["org"] and (value == alias or any(
                    value == alias + hyphen + suffix
                    for hyphen in ("-", "‐", "‑", "–", "—")
                    for suffix in GE_CASE_SUFFIXES)):
                return "ge_jurisdiction_alias", protected
        return embedded_org_match(value, policy) or strict_lexical_match(value, names, policy)
    return None


def filter_shard(path, names, policy):
    table = pq.read_table(path)
    keep = []
    removed = Counter()
    rules = Counter()
    for index, row in enumerate(table.select(["text", "entities_json"]).to_pylist()):
        encoded = row["text"].encode("utf-8")
        matching_kinds = set()
        for entity in json.loads(row["entities_json"]):
            kind = entity["kind"]
            if kind not in KINDS:
                continue
            value = encoded[entity["start"] : entity["end"]].decode("utf-8")
            matched = overlap(value, kind, names, policy)
            if matched:
                matching_kinds.add(kind)
                rules[matched[0]] += 1
        if not matching_kinds:
            matched = full_text_overlap(row["text"], names, policy)
            if matched:
                matching_kinds.add("full_text")
                rules[matched[0]] += 1
        if matching_kinds:
            removed.update(matching_kinds)
            removed["any"] += 1
        else:
            keep.append(index)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        pa.field(field.name, pa.string() if pa.types.is_string_view(field.type) else field.type)
        for field in table.schema
    )
    pq.write_table(table.cast(schema).take(pa.array(keep, type=pa.int64())), OUTPUT / path.name)
    return {"source_rows": table.num_rows, "kept_rows": len(keep),
            "removed": dict(removed), "matching_spans_by_rule": dict(rules)}


def main():
    names, gold_sha256 = gold_names()
    policy = embedded_org_policy(names)
    paths = [SOURCE / name for name in sorted(PINNED_SOURCE)]
    if {path.name for path in SOURCE.glob("*.parquet")} != set(PINNED_SOURCE):
        raise ValueError(f"source shard set changed: {SOURCE}")
    for path in paths:
        checked_source(path)
    report = {"gold_sha256": gold_sha256, "name_counts": {k: len(v) for k, v in names.items()},
              "embedded_policy_counts": {"multiword": sum(map(len, policy[0].values())),
                                         "japanese_heads": len(policy[1])}}
    for path in paths:
        report[path.name] = filter_shard(path, names, policy)
    (OUTPUT / "overlap-filter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
