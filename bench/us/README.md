# US data checks

These tools validate prepared US detector data and compare evaluation errors.
Collection scripts and review recipes are local experiment material and are excluded from Git.
The prepared dataset archive includes the files needed to verify its provenance.

Run from the repository root:

```sh
python3 bench/us/check_data.py --config configs/detector-shared-v5.toml
./target/debug/trainer check-detector-data --config configs/detector-shared-v5.toml > /tmp/tessera-preflight.json
python3 bench/us/check_sampling.py --config configs/detector-shared-v5.toml --silver-counts /tmp/tessera-preflight.json
```

`check_data.py` checks file hashes, source separation and entity overlap. It reports repeated prose; `--strict-prose` also rejects every repeated twelve-word passage, including standard notice wording.
`check_sampling.py` checks the configured mix, repeated examples and source concentration.
`diagnose_predictions.py` explains errors in saved detector predictions.

Tests that need the private training corpus run locally with `cargo test -p trainer -- --ignored`. The default test suite runs without that corpus.
