---
language:
- en
license: cc-by-4.0
pipeline_tag: token-classification
library_name: tessera
tags:
- tessera
- contact-extraction
- named-entity-recognition
- wasm
- experimental
---

# Tessera

A small experimental model for finding contact details in text, locally.

This bundle contains an int8 entity detector and address parser. Tessera's Rust/WebAssembly runtime combines them with deterministic email and phone rules, and can assemble the extracted details into contacts. Text stays in the calling process or browser worker.

## Scope and limits

The current learned detector focuses on English United States material. It can miss or mislabel people, organization names, and addresses, especially in unfamiliar prose. Other countries and languages have no demonstrated general accuracy from this evaluation. Confidence is not a guarantee of correctness.

The checkpoint completed 4,000 optimizer updates. It failed the automatic quality-promotion gates and was accepted for an experimental release. Fresh unseen release accuracy has not been established.

Eight of thirteen small public detector fixture cases remain documented failures, including missed people in prose and tables. Their expected labels are retained in the code repository. These examples are not a general accuracy benchmark.

| Native checkpoint evaluation | Documents | PERSON exact F1 | ORG exact F1 | ADDRESS exact F1 |
| --- | ---: | ---: | ---: | ---: |
| Training-seen real documents | 1,799 | 96.47% | 75.39% | 97.33% |
| Reused real development documents | 196 | 76.86% | 53.76% | 89.22% |

These measurements were taken before int8 packaging. The training-seen set measures fitting; the development set was reused during selection. Neither establishes fresh unseen accuracy. They do not measure email/phone rules, address-component accuracy, or contact grouping.

Separately, the packaged parser scored **99.05% component F1** and **95.23% completely correct parses** on its 3,000-row US test shard through the public `parse_address` API. This measures parsing supplied addresses, not detecting them in prose.

## Artifact

The bundle is model version `0.3.0`, format `2`, and 3,489,616 bytes (3.49 MB uncompressed). It was packaged with runtime version `0.1.0`. The bounded 4,000-update run finished; the full configured training schedule did not, so metadata retains `detector_training_complete=false`.

Its SHA-256 is:

```text
abfa972301f82d23240cc15bc9f70c7c574bb047e33473254a737abacfe995ff
```

The Safetensors header has 39 metadata fields. `bundle.json` includes the same metadata in readable form; `tessera-v1.sha256` contains the `sha256-` integrity string used by Tessera.

## Architecture

The detector uses hashed character n-grams, token features, 96 hidden channels, seven residual convolution blocks with channel RMS normalization, and dilations `[1, 2, 4, 8, 16, 1, 64]`. It predicts BIO labels for PERSON, ORG, and ADDRESS. The separate parser labels address components. Bundle metadata identifies the exact graph, decoder, model version, source checkpoint hashes, and licensing.

## Usage

Use the [Tessera runtime](https://github.com/theiskaa/tessera) and follow the [Rust, JavaScript, and CLI guide](https://github.com/theiskaa/tessera/blob/main/docs/getting-started.md). This is a custom Tessera bundle, not a Transformers `AutoModel` checkpoint. There is no hosted inference endpoint included.

Download `tessera-v1.safetensors` and `tessera-v1.sha256` at the same repository revision. Supply the checksum to Tessera when loading the model. `bundle.json` records artifact size, checksum, and embedded metadata.

Rust offsets are UTF-8 bytes; JavaScript offsets are UTF-16 code units. Ends are exclusive.

## Training and attribution

Training combines synthetic contact documents and annotated real US source documents. The parser has its own training history. Raw corpora, labels, and local run evidence are excluded from this model repository.

The weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain the included `NOTICE` when redistributing. It preserves underlying source attribution, including OpenStreetMap and Open Addresses UK notices. The runtime code is separately licensed under MIT OR Apache-2.0.
