"""Silver labels from the Anthropic Message Batches API instead of interactive agents: the same
slices, guidelines, and output shape as bench/silver/agent_label.py, sent one request per slice.

    python bench/silver/api_label.py submit ROUND FROM TO [--model MODEL] [--prompt FILE]
    python bench/silver/api_label.py collect ROUND

`submit` sends slices FROM..TO of round ROUND as one batch and records its id in
data/interim/silver/r<ROUND>/batches.json. The system prompt is the round's labelling task
(`--prompt`, default internal/bench/silver/prompts/silver<ROUND>_label.txt) followed by the
labelling guidelines, cached across requests. `collect` waits for every recorded batch to end,
writes each slice's labels to pass1/slice-NN.json, and prints the token usage; run
`agent_label.py apply ROUND 1` afterwards to check the labels against the documents.

The API key is read from ~/.config/anthropic/key and never written anywhere else.
"""

import argparse
import json
import pathlib
import re
import sys
import time

import anthropic

KEY = pathlib.Path.home() / ".config" / "anthropic" / "key"
GUIDELINES = pathlib.Path("internal/bench/review/GUIDELINES.md")
PROMPTS = pathlib.Path("internal/bench/silver/prompts")
DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000

INSTRUCTIONS = """You are running without tools: you cannot read or write files or run commands.
The task below describes files and commands; ignore those steps. The slice's documents are in
the user message. Reply with only the JSON object the task says to write for that slice
(`{"<document id>": [{"kind": ..., "text": ...}, ...], ...}`), with every document id of the
slice present, and nothing else: no code fences, no commentary."""


def root(round_no):
    return pathlib.Path(f"data/interim/silver/r{round_no}")


def client():
    if not KEY.exists():
        sys.exit(f"no API key: save it to {KEY} (chmod 600)")
    return anthropic.Anthropic(api_key=KEY.read_text().strip())


def system_prompt(prompt_file):
    task = pathlib.Path(prompt_file).read_text()
    return f"{INSTRUCTIONS}\n\n# Task\n\n{task}\n\n# Guidelines\n\n{GUIDELINES.read_text()}"


def submit(round_no, first, last, model, prompt_file):
    system = system_prompt(prompt_file)
    requests = []
    for n in range(first, last + 1):
        path = root(round_no) / f"slice-{n:02d}.md"
        requests.append({
            "custom_id": f"r{round_no}-s{n:02d}",
            "params": {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user",
                              "content": f"Slice {n:02d}:\n\n{path.read_text()}"}],
            },
        })
    batch = client().messages.batches.create(requests=requests)
    record = root(round_no) / "batches.json"
    ids = json.loads(record.read_text()) if record.exists() else []
    ids.append(batch.id)
    record.write_text(json.dumps(ids, indent=1) + "\n")
    print(f"batch {batch.id}: slices {first:02d}..{last:02d} of round {round_no} with {model}")


def labels_from(text):
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def collect(round_no):
    api = client()
    record = root(round_no) / "batches.json"
    ids = json.loads(record.read_text())
    out = root(round_no) / "pass1"
    out.mkdir(exist_ok=True)
    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    failed = []
    for batch_id in ids:
        while (batch := api.messages.batches.retrieve(batch_id)).processing_status != "ended":
            print(f"{batch_id}: {batch.request_counts}", flush=True)
            time.sleep(60)
        for result in api.messages.batches.results(batch_id):
            slice_no = result.custom_id.rsplit("-s", 1)[1]
            if result.result.type != "succeeded":
                failed.append(f"slice {slice_no}: {result.result.type}")
                continue
            message = result.result.message
            u = message.usage
            usage["input"] += u.input_tokens
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            usage["output"] += u.output_tokens
            text = "".join(block.text for block in message.content if block.type == "text")
            try:
                labels = labels_from(text)
            except json.JSONDecodeError as e:
                failed.append(f"slice {slice_no}: invalid JSON ({e})")
                continue
            (out / f"slice-{slice_no}.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=1) + "\n")
    print("tokens:", usage)
    for f in failed:
        print("FAILED", f)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("round", type=int)
    s.add_argument("first", type=int)
    s.add_argument("last", type=int)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--prompt")
    c = sub.add_parser("collect")
    c.add_argument("round", type=int)
    args = parser.parse_args()
    if args.command == "submit":
        prompt = args.prompt or PROMPTS / f"silver{args.round}_label.txt"
        submit(args.round, args.first, args.last, args.model, prompt)
    else:
        collect(args.round)


if __name__ == "__main__":
    main()
