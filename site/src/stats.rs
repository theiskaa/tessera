//! Numbers the page reports: the evaluation of the shipped parser, its training sample,
//! the golden gate, and the measured download sizes. Copied from the evaluation run files and
//! reports; update together. The tests hold the training-sample and golden numbers to the
//! committed manifest and golden files.

/// Exact-parse rates on one country's test rows.
pub(crate) struct Country {
    pub(crate) code: &'static str,
    pub(crate) name: &'static str,
    pub(crate) rows: u32,
    pub(crate) tessera: f64,
    pub(crate) rules: f64,
}

/// Component F1 of one label.
pub(crate) struct Label {
    pub(crate) name: &'static str,
    pub(crate) f1: f64,
}

/// One confidence bin: components whose stated confidence falls in `low..high`.
pub(crate) struct Bin {
    pub(crate) low: f64,
    pub(crate) high: f64,
    pub(crate) confidence: f64,
    pub(crate) accuracy: f64,
    pub(crate) count: u32,
}

/// One stage of the parser, described with the example address's real values.
pub(crate) struct Step {
    pub(crate) step: &'static str,
    pub(crate) what: &'static str,
}

/// What a funnel row counts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Row {
    /// Source lines left out, for one reason.
    Dropped,
    /// Source lines kept as rows.
    Kept,
    /// Rows written by rewriting kept training rows, not read from the source.
    Generated,
}

/// One row of the training-data funnel.
pub(crate) struct Funnel {
    pub(crate) name: &'static str,
    pub(crate) rows: u64,
    pub(crate) kind: Row,
}

/// Exact-parse rate of the shipped int8 bundle through the library.
pub(crate) const EXACT_TESSERA: f64 = 0.9487;
/// Exact-parse rate of the rules baseline.
pub(crate) const EXACT_RULES: f64 = 0.2085;

/// Exact parses per country, the shipped bundle against the rules baseline.
#[rustfmt::skip]
pub(crate) const BY_COUNTRY: &[Country] = &[
    Country { code: "GB", name: "Great Britain", rows: 3000, tessera: 0.9553, rules: 0.278 },
    Country { code: "DE", name: "Germany", rows: 3000, tessera: 0.9417, rules: 0.4543 },
    Country { code: "GE", name: "Georgia", rows: 3000, tessera: 0.9833, rules: 0.144 },
    Country { code: "US", name: "United States", rows: 3000, tessera: 0.98, rules: 0.1603 },
    Country { code: "JP", name: "Japan", rows: 2918, tessera: 0.8814, rules: 0.0003 },
];

/// Per-label F1 of the trained checkpoint, best first.
#[rustfmt::skip]
pub(crate) const BY_LABEL: &[Label] = &[
    Label { name: "country", f1: 0.9995 },
    Label { name: "postcode", f1: 0.9987 },
    Label { name: "unit", f1: 0.9966 },
    Label { name: "region", f1: 0.9935 },
    Label { name: "po_box", f1: 0.9932 },
    Label { name: "house_number", f1: 0.99 },
    Label { name: "city", f1: 0.9848 },
    Label { name: "level", f1: 0.9842 },
    Label { name: "road", f1: 0.9755 },
    Label { name: "district", f1: 0.9534 },
    Label { name: "suburb", f1: 0.9413 },
];

/// Stated confidence against accuracy of the trained checkpoint's components.
#[rustfmt::skip]
pub(crate) const CALIBRATION: &[Bin] = &[
    Bin { low: 0.0, high: 0.1, confidence: 0.025, accuracy: 0.0571, count: 35 },
    Bin { low: 0.1, high: 0.2, confidence: 0.1612, accuracy: 0.0625, count: 16 },
    Bin { low: 0.2, high: 0.3, confidence: 0.2635, accuracy: 0.0435, count: 23 },
    Bin { low: 0.3, high: 0.4, confidence: 0.3507, accuracy: 0.1746, count: 63 },
    Bin { low: 0.4, high: 0.5, confidence: 0.4534, accuracy: 0.3168, count: 101 },
    Bin { low: 0.5, high: 0.6, confidence: 0.5543, accuracy: 0.4129, count: 201 },
    Bin { low: 0.6, high: 0.7, confidence: 0.651, accuracy: 0.4473, count: 237 },
    Bin { low: 0.7, high: 0.8, confidence: 0.7533, accuracy: 0.6125, count: 289 },
    Bin { low: 0.8, high: 0.9, confidence: 0.8551, accuracy: 0.7212, count: 513 },
    Bin { low: 0.9, high: 1.0, confidence: 0.9986, accuracy: 0.9943, count: 57329 },
];
/// Expected calibration error over the bins.
pub(crate) const ECE: f64 = 0.0079;

/// The address the pipeline walkthrough follows.
pub(crate) const EXAMPLE: &str = "Flat 4, 221B Baker Street, London NW1 6XE, UK";
/// Each stage of `parse_address` on [`EXAMPLE`], from a traced run of the shipped bundle.
#[rustfmt::skip]
pub(crate) const PIPELINE: &[Step] = &[
    Step { step: "tokenize", what: "12 tokens after dropping spaces, each with its byte range\nFlat 0..4 · 4 5..6 · , 6..7 · 221B 8..12 · Baker 13..18 · Street 19..25 · , · London · NW1 · 6XE · , · UK" },
    Step { step: "features", what: "Flat → 12 hashed character n-grams (^F Fl la at t$ ^Fl Fla lat at$ ^Fla Flat lat$) into 32,768 buckets\nscript latin · shape title case · flags title_case line_start unit_term" },
    Step { step: "embed", what: "48 n-gram + 8 script + 8 shape + 23 flags = 87 numbers per token → linear → 96" },
    Step { step: "context", what: "4 residual convolutions, kernel 3, dilation 1 2 4 8 → each token sees 15 tokens either side" },
    Step { step: "label", what: "23 scores per token: O and B-/I- for 11 labels\nFlat B-unit 1.0000 · 221B B-house_number 0.9999 (B-road 0.0001) · London B-city 1.0000" },
    Step { step: "decode", what: "I-x only after B-x or I-x; runs merge into spans; confidence = mean probability\nunit 0..6 · house_number 8..12 · road 13..25 · city 27..33 · postcode 34..41 · country 43..45" },
];

/// Golden cases holding the library's forward pass to the trainer's.
pub(crate) const GOLDEN_CASES: u32 = 24;
/// Largest absolute logit difference across the golden cases.
pub(crate) const GOLDEN_WORST: f64 = 2.6702880859375e-5;
/// The per-case limit on that difference.
pub(crate) const GOLDEN_TOLERANCE: f64 = 0.001;
/// Validation component F1 lost going from f32 to int8 weights.
pub(crate) const INT8_F1_DROP: f64 = 4.263108197499754e-5;

/// Lines of the source read to draw the sample.
pub(crate) const LINES_READ: u64 = 320115455;
/// Where the source lines went, adding up to [`LINES_READ`], then the generated copies.
#[rustfmt::skip]
pub(crate) const FUNNEL: &[Funnel] = &[
    Funnel { name: "other countries", rows: 205700840, kind: Row::Dropped },
    Funnel { name: "over the country quota", rows: 97152930, kind: Row::Dropped },
    Funnel { name: "search query, not an address", rows: 10428838, kind: Row::Dropped },
    Funnel { name: "GB or US road type cut off", rows: 2435761, kind: Row::Dropped },
    Funnel { name: "no address component", rows: 2145394, kind: Row::Dropped },
    Funnel { name: "place name only, over 15%", rows: 957294, kind: Row::Dropped },
    Funnel { name: "postcode of another country", rows: 589693, kind: Row::Dropped },
    Funnel { name: "DE PO box with a street", rows: 240602, kind: Row::Dropped },
    Funnel { name: "JP kana reading", rows: 209246, kind: Row::Dropped },
    Funnel { name: "6 other reasons", rows: 75184, kind: Row::Dropped },
    Funnel { name: "kept: train", rows: 149828, kind: Row::Kept },
    Funnel { name: "kept: valid + test", rows: 29845, kind: Row::Kept },
    Funnel { name: "augmented copies (generated)", rows: 203677, kind: Row::Generated },
];
/// Size of libpostal's tagged OpenStreetMap address file.
pub(crate) const SOURCE_BYTES: u64 = 8033490172;
/// Rewritten copies each training row may get.
pub(crate) const AUGMENT_COPIES: u32 = 2;

/// `gzip -9` bytes of `models/tessera-v1.safetensors`.
pub(crate) const BUNDLE_GZIP: u64 = 3050609;
/// `gzip -9` bytes of the library's baseline wasm: rules, detector, and parser.
pub(crate) const WASM_GZIP: u64 = 107172;
/// `gzip -9` bytes of the same library built with simd128.
pub(crate) const WASM_SIMD_GZIP: u64 = 106914;
/// `gzip -9` bytes the generated phone tables add to the wasm: the baseline with them, less the
/// baseline without them (92,332), as `just wasm-size` prints both.
pub(crate) const PHONE_TABLES_GZIP: u64 = 14840;
/// `gzip -9` bytes libphonenumber's metadata cost before the tables replaced it.
pub(crate) const PHONE_METADATA_GZIP: u64 = 508357;
/// Regions the generated phone tables cover.
pub(crate) const PHONE_REGIONS: u32 = 11;
/// The libphonenumber release the tables are generated from.
pub(crate) const PHONE_SOURCE: &str = "libphonenumber v9.0.33";

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::Value;

    fn manifest() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../data/manifests/parser-sample.json"
        );
        let json = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    fn funnel(name: &str) -> u64 {
        FUNNEL.iter().find(|f| f.name == name).unwrap().rows
    }

    /// The sum of `counts.<country>.<split>` over the countries, for each split given.
    fn split_rows(manifest: &Value, splits: &[&str]) -> u64 {
        let counts = manifest["counts"].as_object().unwrap();
        counts
            .values()
            .flat_map(|c| splits.iter().map(move |s| c[*s].as_u64().unwrap()))
            .sum()
    }

    #[test]
    fn the_funnel_matches_the_sample_manifest() {
        let manifest = manifest();
        let dropped = manifest["dropped_rows"].as_object().unwrap();
        let rows = |key: &str| dropped[key].as_u64().unwrap();
        assert_eq!(LINES_READ, manifest["lines_read"].as_u64().unwrap());
        assert_eq!(
            u64::from(AUGMENT_COPIES),
            manifest["augment_copies"].as_u64().unwrap()
        );
        let named = [
            ("over the country quota", "over_quota"),
            ("search query, not an address", "excluded_query"),
            ("GB or US road type cut off", "truncated_road"),
            ("no address component", "no_component"),
            ("place name only, over 15%", "locality_over_cap"),
            ("postcode of another country", "foreign_postcode"),
            ("DE PO box with a street", "po_box_with_road"),
            ("JP kana reading", "kana_reading"),
        ];
        for (name, key) in named {
            assert_eq!(funnel(name), rows(key), "{name}");
        }
        // Unencodable copies are generated rows, not source lines, so they are not in the funnel.
        let others: Vec<u64> = dropped
            .iter()
            .filter(|(key, _)| *key != "unencodable_copies" && named.iter().all(|(_, k)| k != key))
            .map(|(_, n)| n.as_u64().unwrap())
            .filter(|&n| n > 0)
            .collect();
        assert_eq!(funnel("6 other reasons"), others.iter().sum::<u64>());
        assert_eq!(others.len(), 6);
        assert_eq!(funnel("kept: train"), split_rows(&manifest, &["train"]));
        assert_eq!(
            funnel("kept: valid + test"),
            split_rows(&manifest, &["valid", "test"])
        );
        assert_eq!(
            funnel("augmented copies (generated)"),
            split_rows(&manifest, &["augmented"])
        );
        // "other countries" is not a manifest count; it is what the source rows leave of the
        // lines read, which the figure's note says they add up to.
        let source: u64 = FUNNEL
            .iter()
            .filter(|f| f.kind != Row::Generated)
            .map(|f| f.rows)
            .sum();
        assert_eq!(source, LINES_READ);
    }

    #[test]
    fn the_golden_count_matches_the_golden_files() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/golden/parser");
        let files = std::fs::read_dir(dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".json")
            })
            .count();
        assert_eq!(GOLDEN_CASES as usize, files);
    }
}
