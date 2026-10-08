# tessera

A tiny model for extracting contact details from text, locally.

<p align="center">

[![CI](https://github.com/theiskaa/tessera/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/theiskaa/tessera/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/model-int8%20%C2%B7%203.49%20MB-blue)](#model)
[![Code license](https://img.shields.io/badge/code-MIT%20OR%20Apache--2.0-blue)](#license)
[![Weights license](https://img.shields.io/badge/weights-CC%20BY%204.0-blue)](NOTICE)

</p>

[Live demo](https://tessera.theiskaa.com) · [Hugging Face](https://huggingface.co/theiskaa/tessera)

Tessera finds people, organizations, postal addresses, emails, and phone numbers. It returns their exact positions in the original text, splits addresses into components, and groups related details into contacts. The same Rust implementation runs natively and through WebAssembly in browsers, Node, and Bun.

Email and phone extraction uses validating rules. Names, organizations, and addresses use an int8 neural network; a separate network parses address components. Inference stays in your process or browser worker. No text is sent to an inference service.

**This release is experimental.** The learned detector currently focuses on English United States material. Organization detection and unfamiliar prose need review; fresh unseen release accuracy has not been established. See [results and limitations](#results-and-limitations).

[Usage](#usage) · [Model](#model) · [Results](#results-and-limitations) · [License](#license)

## Usage

| Operation                              | Use it to                                                | Returns                                      |
| -------------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| `detect`                               | Find individual entities in a document                   | Kind, source offsets, confidence, and origin |
| `parse_address` / `parseAddress`       | Split text already known to be one address               | Labeled address components                   |
| `extract_contacts` / `extractContacts` | Associate extracted details with people or organizations | Contacts and unassigned entities             |

Rust and CLI offsets use UTF-8 bytes; the CLI's JSON mode can return UTF-16 instead. JavaScript offsets use UTF-16 code units, so `text.slice(start, end)` returns the entity. Every `end` is exclusive. Keep the original string when using offsets.

### Rust

For an application beside a Tessera checkout, add the runtime as a local dependency:

```toml
[dependencies]
tessera = { path = "../tessera/tessera" }
```

Load the bundle and its checksum once, then reuse the extractor:

```rust
use tessera::{Config, Kind, Query, Tessera};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("models/tessera-v1.safetensors")?;
    let checksum = std::fs::read_to_string("models/tessera-v1.sha256")?;
    let extractor = Tessera::load(&bytes, Config {
        kinds: Kind::all(),
        expected_checksum: Some(checksum.trim()),
    })?;
    let query = Query { country_hint: &["US"], ..Query::default() };
    let text = "Jordan Avery, Project Coordinator\nAcme Corporation\n500 Main St, Springfield, IL 62701\n+1 202-555-0199\njordan@acme.example";
    let contacts = extractor.extract_contacts(text, &query)?;
    println!("{} contacts, {} unassigned entities",
        contacts.contacts.len(), contacts.unassigned.len());
    let address = extractor.parse_address("400 Broad St, Seattle, WA 98109", &query)?;
    println!("{} address components", address.components.len());
    Ok(())
}
```

`unassigned` is part of a successful extraction: the grouper leaves details there when it cannot confidently choose a contact. Experimental detection may omit or mislabel entities.

### JavaScript

Serve the bundle and checksum with your application:

```js
import { createTessera } from "tessera";

const response = await fetch("/models/tessera-v1.sha256");
if (!response.ok) throw new Error("Could not load model checksum");
const integrity = (await response.text()).trim();
const extractor = await createTessera({
  modelUrl: "/models/tessera-v1.safetensors",
  integrity,
  worker: true,
});

try {
  const text = "Contact hello@example.org or +1 202-555-0123.";
  const entities = await extractor.detect(text, { countryHint: ["US"] });
  for (const entity of entities) {
    console.log(entity.kind, text.slice(entity.start, entity.end));
  }
  const contacts = await extractor.extractContacts(text, {
    countryHint: ["US"],
  });
  console.log(contacts.contacts, contacts.unassigned);
} finally {
  extractor.dispose();
}
```

For Node or Bun, read the bundle from disk and pass `modelBytes` instead of `modelUrl`. `worker: true` runs inference off the browser's main thread when workers are available; it is ignored outside that environment. Rules alone need no model fetch: `createTessera({ kinds: ["email", "phone"] })`.

The complete JavaScript API is in [index.d.ts](tessera/js/index.d.ts).

### Options

- `country_hint` / `countryHint` helps interpret national phone numbers; it does not change the detector's training coverage.
- `include_uncertain` / `includeUncertain` returns additional uncertain results. Internal detector thresholds still apply.
- `parse_address` accepts one address of at most 256 non-whitespace tokens and 8 KiB.
- Rust's optional `markdown` feature scans prose and contact link destinations while retaining source offsets. The default JavaScript package omits it and rejects `format: "markdown"`.
- Rust failures use `tessera::Error`; JavaScript failures expose a `TesseraError` code. An empty successful result means nothing was returned.

### Command-line tool

The crates.io and npm name `tessera` belongs to unrelated projects, so install the binary from this repository, or download one from a [GitHub release](https://github.com/theiskaa/tessera/releases):

```sh
cargo install --git https://github.com/theiskaa/tessera tessera --features cli
```

`tessera [--kinds email,phone] [--model PATH] [FILE]` prints the entities in a file or stdin. For programs that drive it as a subprocess, `tessera json --bundle DIR` answers one JSON request read from stdin with one JSON line on stdout. `DIR` holds a `bundle.json` like [models/bundle.json](models/bundle.json); the weights file it names is verified against its `sha256` before loading. The text is only ever read from stdin, never from arguments, which other users can see in `ps`.

| Request field       | Value                                                                        |
| ------------------- | ---------------------------------------------------------------------------- |
| `operation`         | `"detect"`, `"contacts"`, or `"address"`; required                           |
| `text`              | The document, or for `"address"` the one address; required                   |
| `kinds`             | Any of `"person"`, `"org"`, `"address"`, `"email"`, `"phone"`; default all   |
| `country_hint`      | Region codes such as `["US"]`; default inferred from the text                |
| `include_uncertain` | Return low-confidence results too; default `false`                           |
| `format`            | `"text"`, or `"markdown"` in a build with the `markdown` feature             |
| `offsets`           | `"utf8"` bytes (default) or `"utf16"` code units for every `start` and `end` |

```sh
echo '{"operation":"detect","text":"Write to jordan@acme.example","kinds":["email"]}' \
  | tessera json --bundle models
```

```json
{"model":"tessera","operation":"detect","entities":[{"kind":"email","text":"jordan@acme.example","start":9,"end":28,"confidence":0.99,"review_recommended":false,"source":"rules","normalized":"jordan@acme.example"}]}
```

A response always has `model` and `operation`, then `entities` for `detect`, `contacts` and `unassigned` for `contacts`, or `address` with its `components` for `address`. Field names follow [index.d.ts](tessera/js/index.d.ts) in snake case. The exit status is 0 on success, 2 when the request is at fault (malformed JSON, an unknown field or value, or input the library rejects as too large), and 1 when the run fails (I/O, a missing or mismatched bundle, or an inference failure). Errors are one line on stderr.

## Model

The distributed [Safetensors bundle](models/tessera-v1.safetensors) contains both the entity detector and address parser. It uses Tessera's own runtime: it is not a Transformers `AutoModel` checkpoint, and no hosted inference endpoint is included.

| Property                             | Value                                                 |
| ------------------------------------ | ----------------------------------------------------- |
| Model version / bundle format        | `0.3.0` / `2`                                         |
| Runtime version                      | `0.1.0`                                               |
| Bundle size                          | 3,489,616 bytes (3.49 MB; about 3.09 MB with gzip)    |
| Weight encoding                      | Per-channel symmetric int8; float32 scales and biases |
| Detector architecture                | `detector-context96-rms-v2`                           |
| Hidden channels / convolution kernel | 96 / 3                                                |
| Detector dilations                   | `[1, 2, 4, 8, 16, 1, 64]`                             |
| Normalization                        | Channel RMS after each residual block, epsilon `1e-5` |
| Context radius / window / overlap    | 96 / 2,048 / 448 retained tokens                      |

The detector combines hashed character n-grams of lengths 2, 3, and 4 with script, shape, and contextual features. It predicts BIO labels for PERSON, ORG, and ADDRESS. Email and phone spans found by rules are protected from competing model labels. The parser labels components of supplied or detected addresses, and the grouper associates related entities.

The Safetensors header contains 39 metadata fields, including graph and decoder identifiers, dimensions, labels, feature settings, licensing, experimental status, and source checkpoint hashes. [bundle.json](models/bundle.json) exposes the same metadata in readable form. Load [tessera-v1.sha256](models/tessera-v1.sha256) with the bundle to verify its integrity.

Format 2 binds the context graph and decoder explicitly. The runtime also reads legacy format 1 bundles using their original graph. The asset filename is independent of its embedded model version.

## Results and limitations

The detector completed 4,000 optimizer updates. It failed the automatic quality-promotion requirements, including the full 95% training-seen gate, and was accepted for an experimental release. The bounded run finished, but the full configured training schedule did not; metadata retains `detector_training_complete=false`.

These are **native checkpoint exact entity F1** scores, measured before int8 packaging. Both the kind and boundaries must match:

| Evaluation set                    | Documents | PERSON |    ORG | ADDRESS |
| --------------------------------- | --------: | -----: | -----: | ------: |
| Real training-seen documents      |     1,799 | 96.47% | 75.39% |  97.33% |
| Reused real development documents |       196 | 76.86% | 53.76% |  89.22% |

Training-seen results measure fitting. The development set was reused during model selection. **Neither establishes fresh unseen release accuracy.** These scores do not measure email/phone rules, address parsing, or contact grouping. Aggregate PERSON results also conceal weaker biography coverage; organization names, aliases, and boundaries remain a substantial source of error.

Separately, the **packaged address parser** scored 99.05% component F1 and 95.23% completely correct parses on its 3,000-row US test shard through `parse_address`. This tests parsing supplied addresses, not detecting them in prose.

Eight of thirteen small public detector fixture cases remain documented failures, including missed people in prose and tables. Their expected labels are retained. These examples expose limitations; they are not an independent accuracy benchmark.

Confidence and `review_recommended` are signals for inspection, not guarantees. Check names, organizations, and contact assignments before using them in important records. Other countries, languages, and unfamiliar document layouts have no demonstrated general accuracy from this evaluation.

## License

Code is available under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Trained weights are licensed under CC BY 4.0; retain [NOTICE](NOTICE) when redistributing them. It preserves the underlying source attributions, including OpenStreetMap and Open Addresses UK. Training combines synthetic contacts and annotated real US documents; the parser has its own training history. Raw corpora and local run evidence are not part of the model upload.
