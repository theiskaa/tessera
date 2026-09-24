"""Run Presidio's default analyzer over fixtures/detector/*.json, or over a reviewed gold file
with --gold, and write predictions in the trainer's external format with byte offsets.

ORGANIZATION is not enabled by default and is added through the spaCy recognizer; Presidio has
no address entity, so LOCATION is reported against address, a loose comparison."""

import argparse
import glob
import json
import os
import sys
from importlib.metadata import version

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

MAP = {
    "PERSON": "person",
    "ORGANIZATION": "org",
    "LOCATION": "address",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
}


def build_engine():
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp, languages=["en"])
    registry.remove_recognizer("SpacyRecognizer")
    registry.add_recognizer(SpacyRecognizer(supported_entities=["PERSON", "ORGANIZATION", "LOCATION"]))
    return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])


def byte_offset(text, i):
    return len(text[:i].encode("utf-8"))


def entities(engine, text, dropped):
    results = engine.analyze(
        text=text,
        language="en",
        entities=["PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        score_threshold=0.35,
    )
    ents = []
    for r in results:
        kind = MAP.get(r.entity_type)
        if kind is None:
            dropped[r.entity_type] = dropped.get(r.entity_type, 0) + 1
            continue
        ents.append({"kind": kind, "start": byte_offset(text, r.start), "end": byte_offset(text, r.end), "score": round(r.score, 3)})
    ents.sort(key=lambda e: (e["start"], e["end"]))
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "detector"))
    ap.add_argument("--gold", help="a JSONL file of reviewed cases; predictions are keyed by case name")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "presidio.json"))
    args = ap.parse_args()

    engine = build_engine()
    dropped = {}
    predictions = []
    if args.gold:
        with open(args.gold, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    predictions.append({"name": case["name"], "entities": entities(engine, case["input"], dropped)})
    else:
        for path in sorted(glob.glob(os.path.join(args.fixtures, "*.json"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(path, encoding="utf-8") as f:
                cases = json.load(f)["cases"]
            for case in cases:
                predictions.append({"fixture": stem, "case": case["name"], "entities": entities(engine, case["input"], dropped)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = {
        "system": "presidio",
        "presidio_analyzer": version("presidio-analyzer"),
        "spacy_model": "en_core_web_lg " + version("en_core_web_lg"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({**header, "predictions": predictions}, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}; dropped types: {dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
