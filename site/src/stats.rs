//! Numbers the page reports: the evaluation of the shipped parser, its training sample,
//! the golden gate, and the measured download sizes. Generated from the evaluation run
//! files, the data manifests, and the bundle; regenerate rather than edit.

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
pub(crate) const EXACT_TESSERA: f64 = 0.9504;
/// Exact-parse rate of the rules baseline.
pub(crate) const EXACT_RULES: f64 = 0.2085;

/// Exact parses per country, the shipped bundle against the rules baseline.
#[rustfmt::skip]
pub(crate) const BY_COUNTRY: &[Country] = &[
    Country { code: "GB", name: "Great Britain", rows: 3000, tessera: 0.955, rules: 0.278 },
    Country { code: "DE", name: "Germany", rows: 3000, tessera: 0.943, rules: 0.4543 },
    Country { code: "GE", name: "Georgia", rows: 3000, tessera: 0.9843, rules: 0.144 },
    Country { code: "US", name: "United States", rows: 3000, tessera: 0.981, rules: 0.1603 },
    Country { code: "JP", name: "Japan", rows: 2918, tessera: 0.8869, rules: 0.0003 },
];

/// Per-label F1 of the trained checkpoint, best first.
#[rustfmt::skip]
pub(crate) const BY_LABEL: &[Label] = &[
    Label { name: "country", f1: 0.9996 },
    Label { name: "postcode", f1: 0.9992 },
    Label { name: "unit", f1: 0.9961 },
    Label { name: "po_box", f1: 0.9945 },
    Label { name: "region", f1: 0.9942 },
    Label { name: "house_number", f1: 0.9897 },
    Label { name: "city", f1: 0.9866 },
    Label { name: "level", f1: 0.9862 },
    Label { name: "road", f1: 0.9775 },
    Label { name: "district", f1: 0.9524 },
    Label { name: "suburb", f1: 0.9414 },
];

/// Stated confidence against accuracy of the trained checkpoint's components.
#[rustfmt::skip]
pub(crate) const CALIBRATION: &[Bin] = &[
    Bin { low: 0.0, high: 0.1, confidence: 0.0455, accuracy: 0.1, count: 20 },
    Bin { low: 0.1, high: 0.2, confidence: 0.1481, accuracy: 0.0, count: 12 },
    Bin { low: 0.2, high: 0.3, confidence: 0.2446, accuracy: 0.1111, count: 36 },
    Bin { low: 0.3, high: 0.4, confidence: 0.3602, accuracy: 0.2982, count: 57 },
    Bin { low: 0.4, high: 0.5, confidence: 0.4557, accuracy: 0.2609, count: 92 },
    Bin { low: 0.5, high: 0.6, confidence: 0.5546, accuracy: 0.4676, count: 216 },
    Bin { low: 0.6, high: 0.7, confidence: 0.65, accuracy: 0.5459, count: 218 },
    Bin { low: 0.7, high: 0.8, confidence: 0.7527, accuracy: 0.6194, count: 310 },
    Bin { low: 0.8, high: 0.9, confidence: 0.8546, accuracy: 0.7196, count: 510 },
    Bin { low: 0.9, high: 1.0, confidence: 0.9987, accuracy: 0.9943, count: 57331 },
];
/// Expected calibration error over the bins.
pub(crate) const ECE: f64 = 0.0074;

/// The address the pipeline walkthrough follows.
pub(crate) const EXAMPLE: &str = "Flat 4, 221B Baker Street, London NW1 6XE, UK";
/// Each stage of `parse_address` on [`EXAMPLE`], from a traced run of the shipped bundle.
#[rustfmt::skip]
pub(crate) const PIPELINE: &[Step] = &[
    Step { step: "tokenize", what: "12 tokens after dropping spaces, each with its byte range\nFlat 0..4 · 4 5..6 · , 6..7 · 221B 8..12 · Baker 13..18 · Street 19..25 · , · London · NW1 · 6XE · , · UK" },
    Step { step: "features", what: "Flat → 12 hashed character n-grams (^F Fl la at t$ ^Fl Fla lat at$ ^Fla Flat lat$) into 32,768 buckets\nscript latin · shape title case · flags title_case line_start unit_term" },
    Step { step: "embed", what: "48 n-gram + 8 script + 8 shape + 22 flags = 86 numbers per token → linear → 96" },
    Step { step: "context", what: "4 residual convolutions, kernel 3, dilation 1 2 4 8 → each token sees 15 tokens either side" },
    Step { step: "label", what: "23 scores per token: O and B-/I- for 11 labels\nFlat B-unit 1.0000 · 221B B-house_number 0.9999 (B-road 0.0001) · London B-city 1.0000" },
    Step { step: "decode", what: "I-x only after B-x or I-x; runs merge into spans; confidence = mean probability\nunit 0..6 · house_number 8..12 · road 13..25 · city 27..33 · postcode 34..41 · country 43..45" },
];

/// Golden cases holding the library's forward pass to the trainer's.
pub(crate) const GOLDEN_CASES: u32 = 24;
/// Largest absolute logit difference across the golden cases.
pub(crate) const GOLDEN_WORST: f64 = 2.288818359375e-5;
/// The per-case limit on that difference.
pub(crate) const GOLDEN_TOLERANCE: f64 = 0.001;
/// Validation component F1 lost going from f32 to int8 weights.
pub(crate) const INT8_F1_DROP: f64 = 0.00017618346940684315;

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
pub(crate) const BUNDLE_GZIP: u64 = 1498542;
/// `gzip -9` bytes of the library's baseline wasm, rules and parser.
pub(crate) const WASM_GZIP: u64 = 95093;
/// `gzip -9` bytes of the same library built with simd128.
pub(crate) const WASM_SIMD_GZIP: u64 = 94905;
/// `gzip -9` bytes the generated phone tables add to the wasm.
pub(crate) const PHONE_TABLES_GZIP: u64 = 13640;
/// `gzip -9` bytes libphonenumber's metadata cost before the tables replaced it.
pub(crate) const PHONE_METADATA_GZIP: u64 = 508357;
/// Regions the generated phone tables cover.
pub(crate) const PHONE_REGIONS: u32 = 11;
/// The libphonenumber release the tables are generated from.
pub(crate) const PHONE_SOURCE: &str = "libphonenumber v9.0.33";
