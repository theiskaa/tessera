# tessera

Find contact details in text, locally.

Tessera detects people, organizations, postal addresses, emails, and phone numbers. It returns exact source spans, splits addresses into components, and groups related details into contacts. The same Rust implementation runs natively and through WebAssembly in browsers, Node, and Bun.

**The US detector is experimental.** Organization detection and unfamiliar prose still need review. The latest native checkpoint scores 76.86% PERSON, 53.76% ORG, and 89.22% ADDRESS exact F1 on a reused development set. These are development results; fresh unseen release accuracy has not been established. See [the model guide](docs/model.md) for the evaluation context.

## Start here

The email and phone rules work without model weights:

```rust
use tessera::{Config, Kind, Query, Tessera};

fn main() -> Result<(), tessera::Error> {
    let extractor = Tessera::load(&[], Config {
        kinds: Kind::Email | Kind::Phone,
        expected_checksum: None,
    })?;
    let text = "Contact hello@example.org or +1 202-555-0123.";
    for entity in extractor.detect(text, &Query::default())? {
        println!("{}: {}", entity.kind.as_str(), entity.text(text));
    }
    Ok(())
}
```

Load `models/tessera-v1.safetensors` to also detect names and addresses. The [getting started guide](docs/getting-started.md) covers Rust, JavaScript, and the command line.

## What you get

| Operation | Result |
| --- | --- |
| `detect` | Entities in source order, with kind, offsets, confidence, and origin |
| `parse_address` / `parseAddress` | Components of text already known to be one address |
| `extract_contacts` / `extractContacts` | Contacts plus entities that could not be assigned confidently |

Rust and CLI offsets use UTF-8 bytes. JavaScript offsets use UTF-16 code units, so `text.slice(start, end)` returns the entity. Every `end` is exclusive.

Inference stays in the calling process or browser worker. A browser can fetch model assets once; text is processed locally.

## Guides

- [Getting started](docs/getting-started.md): load a model, call the APIs, and use the CLI.
- [Model](docs/model.md): the pipeline, architecture, evaluation, and limits.
- [Development](docs/development.md): repository layout, tests, WebAssembly, and the site.
- [Releasing](docs/releasing.md): final checks, Hugging Face, and publication steps.

## Build from source

The repository pins Rust in [rust-toolchain.toml](rust-toolchain.toml).

```sh
cargo test --workspace
```

With `just`, `wasm-pack`, and Binaryen installed, `just wasm` builds the JavaScript package. With Trunk installed, `just site` builds the demo into `site/dist`.

## License

Code is available under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Model weights are licensed under CC BY 4.0 as described in [NOTICE](NOTICE), which also contains the training-source attribution and underlying dataset notices.
