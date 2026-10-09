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

**This release is experimental.** The learned detector currently focuses on English United States material and expects `country_hint: ["US"]`. Organization detection and unfamiliar prose need review, and accuracy on fresh documents rests on a small holdout. See [results and limitations](#results-and-limitations).

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

- `country_hint` / `countryHint` helps interpret national phone numbers; it does not change the detector's training coverage. The current model was evaluated with `["US"]`; without it, most US numbers written without `+1` are missed.
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
| `country_hint`      | Region codes such as `["US"]`; default the hint the bundle's detector expects |
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

With the `json` feature, a program answers the same request in process with `tessera::json::answer(dir, request)`, which returns the response object or a `Fault` saying whether the request or the run is to blame.

A response always has `model` and `operation`, then `entities` for `detect`, `contacts` and `unassigned` for `contacts`, or `address` with its `components` for `address`. Field names follow [index.d.ts](tessera/js/index.d.ts) in snake case. The exit status is 0 on success, 2 when the request is at fault (malformed JSON, an unknown field or value, or input the library rejects as too large), and 1 when the run fails (I/O, a missing or mismatched bundle, or an inference failure). Errors are one line on stderr.

## Model

The distributed [Safetensors bundle](models/tessera-v1.safetensors) contains both the entity detector and address parser. It uses Tessera's own runtime: it is not a Transformers `AutoModel` checkpoint, and no hosted inference endpoint is included.

| Property                             | Value                                                 |
| ------------------------------------ | ----------------------------------------------------- |
| Model version / bundle format        | `0.4.0` / `2`                                         |
| Runtime version                      | `0.3.0`                                               |
| Bundle size                          | 3,490,312 bytes (3.49 MB; about 3.08 MB with gzip)    |
| Weight encoding                      | Per-channel symmetric int8; float32 scales and biases |
| Detector architecture                | `detector-context96-rms-v2`                           |
| Hidden channels / convolution kernel | 96 / 3                                                |
| Detector dilations                   | `[1, 2, 4, 8, 16, 1, 64]`                             |
| Normalization                        | Channel RMS after each residual block, epsilon `1e-5` |
| Context radius / window / overlap    | 96 / 2,048 / 448 retained tokens                      |
| Detector input / postprocessing      | 25 token flags / `address_labeled_fields_v1`          |

The detector combines hashed character n-grams of lengths 2, 3, and 4 with script, shape, and contextual features, including whether a token starts or ends a tab-separated cell. It predicts BIO labels for PERSON, ORG, and ADDRESS. Email and phone spans found by rules are protected from competing model labels. The parser labels components of supplied or detected addresses, and the grouper associates related entities.

The Safetensors header contains 45 metadata fields, including graph and decoder identifiers, dimensions, labels, feature settings, licensing, experimental status, and source checkpoint hashes. [bundle.json](models/bundle.json) exposes the same metadata in readable form. Load [tessera-v1.sha256](models/tessera-v1.sha256) with the bundle to verify its integrity.

Format 2 binds the context graph and decoder explicitly. The runtime also reads legacy format 1 bundles using their original graph. The asset filename is independent of its embedded model version.

## Results and limitations

The detector completed 6,160 optimizer updates on 905 reviewed US source pages and passed its learning gate, a check that it learned its own training data. It is released as experimental: it does not reach 95% exact precision and recall for every kind on every evaluation set.

The scores below come from the shipped int8 bundle through the library with `country_hint: ["US"]`. They are **exact** precision / recall: a prediction counts only when its kind and both boundaries match.

| Evaluation set  | Documents |      PERSON |         ORG |     ADDRESS |       EMAIL |       PHONE |
| --------------- | --------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| Fresh holdout   |        38 | 97.1 / 94.4 |           — | 96.7 / 98.3 | 100 / 100   | 99.3 / 99.3 |
| Development     |       100 | 94.5 / 94.5 | 90.3 / 81.2 | 100 / 100   | 100 / 100   | 100 / 100   |
| Held-out pieces |        89 | 97.3 / 92.2 | 84.7 / 70.1 | 88.5 / 87.7 | 100 / 95.5  | 99.8 / 98.3 |
| Calibration     |        44 | 91.5 / 95.6 | 64.0 / 74.2 | 89.0 / 90.3 | 100 / 100   | 95.9 / 94.9 |

- **Fresh holdout:** documents from North Dakota and Minnesota sources, set aside before training and scored once, on this model. It is small: one missed person moves PERSON recall by about 1.4 points. Organizations are sparsely labeled there, so ORG is not reported. The previous model, 0.3.0, scored 69.6 / 71.1 for PERSON and 95.8 / 51.1 for ADDRESS on the North Dakota part. This score was taken after packaging, so the bundle's metadata still records the fresh evaluation as pending.
- **Development:** never used to pick checkpoints within a run, but its results informed the choice among runs and training data, so it is optimistic.
- **Held-out pieces and calibration:** whole source pages kept out of training, mostly dense directories and rosters. Their results were also read while comparing runs, so they are not fresh either.

Without a country hint, PERSON and ADDRESS barely change, but PHONE recall falls to 38.9% on development and 64.1% on held-out pieces, because national numbers are not recognized.

Separately, the **packaged address parser** scored 99.05% component F1 and 95.23% completely correct parses on its 3,000-row US test shard through `parse_address`. This tests parsing supplied addresses, not detecting them in prose.

Seven of thirteen small public detector fixture cases remain documented failures, including missed people in prose and tables. Their expected labels are retained. These examples expose limitations; they are not an independent accuracy benchmark.

Confidence and `review_recommended` are signals for inspection, not guarantees. Check names, organizations, and contact assignments before using them in important records. Other countries, languages, and unfamiliar document layouts have no demonstrated general accuracy from this evaluation.

## License

Code is available under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Trained weights are licensed under CC BY 4.0; retain [NOTICE](NOTICE) when redistributing them. It preserves the underlying source attributions, including OpenStreetMap and Open Addresses UK. Training combines synthetic contacts and annotated real US documents; the parser has its own training history. Raw corpora and local run evidence are not part of the model upload.
