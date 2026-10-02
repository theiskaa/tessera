# Release checklist

[README](../README.md) · [Model](model.md) · [Development](development.md)

The first release is experimental. Release preparation does not authorize publishing. Git pushes, GitHub releases, package uploads, website deployment, and Hugging Face uploads each wait for the owner's permission.

## 1. Freeze the model

Use the finished checkpoint, not another training run. Keep the original checkpoint and its evidence unchanged. `trainer promote` creates a separate local staging directory and requires an explicit experimental-quality waiver. It checks the native final snapshot, completed updates, learning checks, and evaluation hashes. The waiver accepts the documented quality limitations; it does not turn a failed accuracy gate into a pass.

```sh
target/release/trainer promote --help
target/release/trainer quantize --help
target/release/trainer export --help
```

Quantize the staged detector and export it with the paired parser. Quantization must pass its normal accuracy-drop check. Update the bundle, checksum, and golden vectors together. The bundle must describe the graph, decoder, experimental status, licensing, source checkpoint hashes, and evaluation scope.

## 2. Check the finished artifact

Run native checks, JavaScript tests in Node and Bun, browser tests, and the size gate:

```sh
just
just js-test
just test-web-ci
just size-gate
just site
```

Check the four live examples, custom input, address parsing, worker loading, and checksum rejection in the built site. Confirm the site's model is byte-for-byte identical to `models/tessera-v1.safetensors`. Record packaged-model results separately from native checkpoint results. Keep training-seen and reused-development measurements clearly labeled.

When an intentional architecture change increases size, update the measured size baseline in its own commit. Keep the growth limit intact. Inspect the working tree and make scoped commits; exclude raw data, processed shards, runs, and local archives.

## 3. Prepare Hugging Face locally

Create a separate model repository, such as `<account>/tessera`. It holds the model assets, not the codebase or training corpus. Its README is the model card, with architecture, intended use, evaluation limits, runtime instructions, and attribution. The code remains on GitHub.

```sh
python3 scripts/prepare-model-release.py --out runs/huggingface-release
```

This command checks the bundle checksum and release metadata, then copies only the model, checksum, model card, and notices. It writes `bundle.json` describing the exact artifact. It does not connect to Hugging Face or upload anything. Inspect every staged file before requesting upload approval.

Tessera uses its own Rust/WebAssembly runtime. The Safetensors file is not a Transformers `AutoModel` checkpoint, and this release does not provide a hosted inference endpoint.

## 4. Publish only after approval

Create a Hugging Face account at https://huggingface.co/join if you do not already have one. Choose the account and repository name. If the `hf` command is missing, install it into an isolated environment:

```sh
python3 -m venv runs/hf-cli-venv
runs/hf-cli-venv/bin/python -m pip install --upgrade huggingface_hub
source runs/hf-cli-venv/bin/activate
hf --help
```

Keep this environment out of Git. Authenticate in your terminal without putting a token in source files or commands shared in chat:

```sh
hf auth login
hf auth whoami
```

After the owner approves the upload, replace the account placeholder and run:

```sh
hf upload <account>/tessera runs/huggingface-release . --repo-type model
```

Save the resulting commit revision. Download the published artifact at that immutable revision and compare it to the local release:

```sh
hf download <account>/tessera tessera-v1.safetensors tessera-v1.sha256 --revision <commit> --local-dir runs/huggingface-verification
cmp models/tessera-v1.safetensors runs/huggingface-verification/tessera-v1.safetensors
cmp models/tessera-v1.sha256 runs/huggingface-verification/tessera-v1.sha256
```

Use a revision-pinned download URL in documentation. The site can serve its own identical bundle, avoiding a runtime dependency on a moving Hub branch.

Hugging Face documents the [upload/download CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and [model-card metadata](https://huggingface.co/docs/hub/model-cards).

## 5. Release the code and site

After the owner approves the Git push, confirm GitHub CI passes on that exact commit. Then obtain approval for the release tag and GitHub release, and for deploying `site/dist` to the chosen host. Attach the model checksum and link the immutable Hugging Face revision. Rust/npm registry publication is a separate decision; repository and local-package instructions work without it.

Finish by testing the deployed site's four examples and one custom input. Confirm its distributed model checksum matches the approved release.
