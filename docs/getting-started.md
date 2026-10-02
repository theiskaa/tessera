# Getting started

[README](../README.md) · [Model](model.md) · [Development](development.md)

## Choose an operation

Use `detect` to find individual entities in a document. Use `extract_contacts` to associate those entities with people or organizations. Use `parse_address` when the entire input is already one address.

Email and phone extraction needs no model bundle. People, organizations, and addresses need the relevant networks in a bundle.

## Rust

For a project beside a Tessera checkout, add the local dependency:

```toml
[dependencies]
tessera = { path = "../tessera/tessera" }
```

Load the bundle once and reuse the extractor:

```rust
use tessera::{Config, Kind, Query, Tessera};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("models/tessera-v1.safetensors")?;
    let checksum = std::fs::read_to_string("models/tessera-v1.sha256")?;
    let extractor = Tessera::load(&bytes, Config {
        kinds: Kind::all(),
        expected_checksum: Some(checksum.trim()),
    })?;

    let text = "Ada Example\nExample Research\nada@example.org\n+1 202-555-0123";
    let query = Query {
        country_hint: &["US"],
        ..Query::default()
    };
    for entity in extractor.detect(text, &query)? {
        println!("{}: {}", entity.kind.as_str(), entity.text(text));
    }
    let contacts = extractor.extract_contacts(text, &query)?;
    println!("{} contacts, {} unassigned entities",
        contacts.contacts.len(), contacts.unassigned.len());

    let address = extractor.parse_address(
        "1600 Pennsylvania Avenue NW, Washington, DC 20500",
        &Query::default(),
    )?;
    println!("{} address components", address.components.len());
    Ok(())
}
```

The example names illustrate the API; experimental detection may omit them. `unassigned` is part of a successful extraction: the grouper leaves details there when it cannot choose a contact confidently.

Keep the original string when using `Entity::text` or component offsets. Rust offsets index UTF-8 bytes in that exact string.

## JavaScript

Build the package with `just wasm`, then install the generated directory in your application:

```sh
npm install /path/to/tessera/tessera/pkg
```

Serve the model bundle with your application and provide its URL:

```js
import { createTessera } from "tessera";

const extractor = await createTessera({
  modelUrl: "/models/tessera-v1.safetensors",
  worker: true,
});

try {
  const text = "Contact hello@example.org or +1 202-555-0123.";
  const entities = await extractor.detect(text, { countryHint: ["US"] });
  for (const entity of entities) {
    console.log(entity.kind, text.slice(entity.start, entity.end));
  }
  const contacts = await extractor.extractContacts(text, { countryHint: ["US"] });
  console.log(contacts.contacts, contacts.unassigned);
} finally {
  extractor.dispose();
}
```

For Node or Bun, read the bundle from disk and pass `modelBytes` instead of `modelUrl`. Pass the checksum file's trimmed `sha256-…` string as `integrity` to verify it. `worker: true` runs inference off the browser's main thread when workers are available; it is ignored outside that environment.

Rules alone need no model fetch:

```js
const extractor = await createTessera({ kinds: ["email", "phone"] });
const entities = await extractor.detect("hello@example.org");
extractor.dispose();
```

JavaScript offsets index UTF-16 code units, including when the text contains emoji. The complete API is in [index.d.ts](../tessera/js/index.d.ts).

## Command line

Run these commands from the repository root. The CLI reads a file or standard input and prints JSON.

```sh
cargo run -p tessera --features cli -- --pretty document.txt
```

Its default kinds are email and phone. To enable all five kinds:

```sh
cargo run -p tessera --features cli -- \
  --model models/tessera-v1.safetensors \
  --kinds person,org,address,email,phone \
  --country US --pretty document.txt
```

Use `-` as the filename for standard input. `--help` lists the supported options. CLI JSON uses UTF-8 byte offsets and snake_case fields such as `review_recommended`.

## Options and limits

- `country_hint` / `countryHint` helps interpret national phone numbers. It does not change the detector's training coverage.
- `include_uncertain` / `includeUncertain` returns additional uncertain results. Internal detector thresholds still apply.
- `parse_address` accepts one address of at most 256 non-whitespace tokens and 8 KiB.
- Rust's optional `markdown` feature scans prose and contact link destinations while retaining source offsets. The default JavaScript package omits this feature and rejects `format: "markdown"`.
- Rust failures use `tessera::Error`; JavaScript failures expose a `TesseraError` code. An empty successful result means nothing was returned.
