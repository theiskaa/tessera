//! The entity detector's data and scores: synthetic documents encoded with the rules layer's
//! email and phone spans as features, BIO labels over person, org, and address, greedy
//! decoding that keeps rule spans out of model spans, and exact and lenient span scores.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail, ensure};
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use polars::prelude::{ParquetReader, SerReader};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tessera::Kind;
use tessera::internal::{
    DETECT_MIN_ADDRESS, DETECT_MIN_ORG, DETECT_MIN_PERSON, FeatureConfig, MAX_ENTITY_TOKENS,
    decode_detector, featurize, flag, is_content, normalized_us_address_end, paragraph_breaks,
    scan_rules, tokenize,
};

use crate::data::Split;
use crate::dataset::{Encoded, ParserBatch, ParserBatcher};
use crate::net::TaggerNet;

/// The model kinds, in label order: kind `k` has `B = 1 + 2k` and `I = 2 + 2k`.
pub use tessera::internal::DETECTOR_KINDS as KINDS;
pub use tessera::internal::DETECTOR_LABELS;
/// Lowest mean label probability kept per kind, in `KINDS` order, from the library's policy.
const DETECT_MIN: [f32; 3] = [DETECT_MIN_PERSON, DETECT_MIN_ORG, DETECT_MIN_ADDRESS];

fn kind_index(kind: Kind) -> Option<usize> {
    KINDS.iter().position(|k| *k == kind)
}

fn label(kind: usize, begin: bool) -> u8 {
    let k = kind as u8;
    if begin { 1 + 2 * k } else { 2 + 2 * k }
}

/// One gold or predicted entity: kind index and byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindSpan {
    pub kind: usize,
    pub start: u32,
    pub end: u32,
}

/// A detector document as the network sees it, with its gold spans and, per retained token,
/// whether a paragraph break comes before it (where the decoder closes every span).
#[derive(Debug, Clone)]
pub struct DetectorDoc {
    /// Original source text for the runtime's address boundary adjustment.
    pub text: String,
    pub enc: Encoded,
    pub gold: Vec<KindSpan>,
    pub breaks: Vec<bool>,
}

/// Why a document could not be encoded.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum DetectorEncodeError {
    #[error("span {0:?} does not start or end on a token boundary")]
    PartialToken(KindSpan),
    #[error("span {0:?} overlaps an email or phone found by the rules")]
    RuleOverlap(KindSpan),
    #[error("span {0:?} contains {1} content tokens, above the decoder limit")]
    EntityTooLong(KindSpan, usize),
}

/// Encodes one document. Rule spans come from the rules layer, not from gold, so training
/// sees exactly what inference computes; a gold span that overlaps one is an error, because
/// the decoder can never produce it.
pub fn encode_document(
    text: &str,
    gold: &[KindSpan],
    fc: &FeatureConfig,
) -> Result<Encoded, DetectorEncodeError> {
    let tokens = tokenize(text);
    let rule_spans: Vec<(usize, usize)> = scan_rules(text, &[])
        .iter()
        .map(|e| (e.start, e.end))
        .collect();
    for g in gold {
        let (s, e) = (g.start as usize, g.end as usize);
        if rule_spans.iter().any(|&(rs, re)| s < re && rs < e) {
            return Err(DetectorEncodeError::RuleOverlap(*g));
        }
    }
    let feats = featurize(text, &tokens, &rule_spans, None, fc, None);
    let mut enc = Encoded {
        token_spans: Vec::new(),
        ngram_ids: Vec::new(),
        script: Vec::new(),
        shape: Vec::new(),
        flags: Vec::new(),
        labels: Vec::new(),
        country: String::new(),
    };
    for (t, f) in tokens.iter().zip(&feats) {
        if !is_content(t) {
            continue;
        }
        enc.token_spans.push((t.start as u32, t.end as u32));
        enc.ngram_ids
            .push(f.ngram_ids.iter().map(|id| id + 1).collect());
        enc.script.push(f.script);
        enc.shape.push(f.shape);
        enc.flags.push(f.flags);
        enc.labels.push(0);
    }
    for g in gold {
        let first = enc.token_spans.iter().position(|t| t.0 == g.start);
        let last = enc.token_spans.iter().rposition(|t| t.1 == g.end);
        let (Some(first), Some(last)) = (first, last) else {
            return Err(DetectorEncodeError::PartialToken(*g));
        };
        if last < first {
            return Err(DetectorEncodeError::PartialToken(*g));
        }
        let span_tokens = last + 1 - first;
        if span_tokens > MAX_ENTITY_TOKENS {
            return Err(DetectorEncodeError::EntityTooLong(*g, span_tokens));
        }
        enc.labels[first] = label(g.kind, true);
        for l in &mut enc.labels[first + 1..=last] {
            *l = label(g.kind, false);
        }
    }
    Ok(enc)
}

#[derive(Deserialize)]
struct GoldJson {
    kind: String,
    start: u32,
    end: u32,
}

#[derive(Deserialize)]
struct DevelopmentGoldLine {
    name: String,
    input: String,
    expected: Vec<DevelopmentGoldSpan>,
    country: String,
}

#[derive(Deserialize)]
struct DevelopmentGoldSpan {
    kind: String,
    start: u32,
    end: u32,
    text: String,
}

/// Encodes reviewed real development cases for detector checkpoint selection. Gold remains
/// separate from the input features, so an unreachable span counts as a miss during scoring.
pub fn load_development_gold(path: &Path, fc: &FeatureConfig) -> anyhow::Result<Vec<DetectorDoc>> {
    let data =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut docs = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for (index, line) in data
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
    {
        let row: DevelopmentGoldLine = serde_json::from_str(line).with_context(|| {
            format!("{}:{} invalid development gold", path.display(), index + 1)
        })?;
        ensure!(
            row.country == "US",
            "{}:{} is not a US development case",
            path.display(),
            index + 1
        );
        ensure!(
            names.insert(row.name.clone()),
            "{}:{} duplicates development case {}",
            path.display(),
            index + 1,
            row.name
        );
        let mut declared = Vec::with_capacity(row.expected.len());
        for span in row.expected {
            ensure!(
                row.input.get(span.start as usize..span.end as usize) == Some(span.text.as_str())
                    && span.start < span.end,
                "{}:{} has a gold span that differs from the source text",
                path.display(),
                index + 1
            );
            declared.push(GoldJson {
                kind: span.kind,
                start: span.start,
                end: span.end,
            });
        }
        let gold = model_spans(&row.input, &declared)
            .with_context(|| format!("{}:{}", path.display(), index + 1))?;
        let mut enc = encode_document(&row.input, &[], fc)
            .with_context(|| format!("{}:{}", path.display(), index + 1))?;
        ensure!(
            !enc.token_spans.is_empty(),
            "{}:{} has no retained tokens",
            path.display(),
            index + 1
        );
        enc.country = row.country;
        docs.push(DetectorDoc {
            text: row.input.clone(),
            enc,
            gold,
            breaks: breaks_of(&row.input),
        });
    }
    ensure!(
        !docs.is_empty(),
        "{} has no development cases",
        path.display()
    );
    Ok(docs)
}

/// Validates every declared span before dropping the rule-layer kinds from BIO targets.
fn model_spans(text: &str, declared: &[GoldJson]) -> anyhow::Result<Vec<KindSpan>> {
    let mut targets = Vec::new();
    for (index, span) in declared.iter().enumerate() {
        let start = span.start as usize;
        let end = span.end as usize;
        ensure!(
            start < end && text.get(start..end).is_some(),
            "entity {index} has an empty, out-of-range, or non-UTF-8 span {}..{}",
            span.start,
            span.end
        );
        let Some(kind) = Kind::from_str_label(&span.kind) else {
            bail!("entity {index} has unknown kind {:?}", span.kind);
        };
        if let Some(kind) = kind_index(kind) {
            targets.push((
                index,
                KindSpan {
                    kind,
                    start: span.start,
                    end: span.end,
                },
            ));
        }
    }
    let mut by_start = targets.clone();
    by_start.sort_by_key(|(_, span)| (span.start, span.end));
    for pair in by_start.windows(2) {
        ensure!(
            pair[0].1.end <= pair[1].1.start,
            "entities {} and {} overlap as detector targets",
            pair[0].0,
            pair[1].0
        );
    }
    Ok(targets.into_iter().map(|(_, span)| span).collect())
}

/// Documents loaded from a synthetic split. Invalid rows fail the load.
#[derive(Debug, Default, Clone, Serialize)]
pub struct LoadCounts {
    pub encoded: usize,
    pub partial_token: usize,
    pub rule_overlap: usize,
}

/// Reads one split of the synthetic corpus and encodes it. Email and phone spans are the
/// rules layer's, so only person, org, and address are model targets.
pub fn load_split(
    dir: &Path,
    split: Split,
    fc: &FeatureConfig,
) -> anyhow::Result<(Vec<DetectorDoc>, LoadCounts)> {
    let path = dir.join(format!("{}.parquet", split.name()));
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;
    let col = |name: &str| -> anyhow::Result<_> {
        Ok(df
            .column(name)
            .with_context(|| format!("{} has no {name} column", path.display()))?
            .str()?
            .clone())
    };
    let (text, entities, country) = (col("text")?, col("entities_json")?, col("country")?);
    let mut docs = Vec::with_capacity(df.height());
    let mut counts = LoadCounts::default();
    for i in 0..df.height() {
        let row_text = text
            .get(i)
            .with_context(|| format!("{} row {} has null text", path.display(), i + 1))?;
        let row_country = country
            .get(i)
            .with_context(|| format!("{} row {} has null country", path.display(), i + 1))?;
        ensure!(
            row_country.len() == 2 && row_country.bytes().all(|b| b.is_ascii_uppercase()),
            "{} row {} has invalid country {row_country:?}",
            path.display(),
            i + 1
        );
        ensure!(
            row_country == "US",
            "{} row {} has country {row_country:?}; detector data must be US-only",
            path.display(),
            i + 1
        );
        let declared: Vec<GoldJson> = serde_json::from_str(
            entities
                .get(i)
                .with_context(|| format!("{} row {} has null entities", path.display(), i + 1))?,
        )
        .with_context(|| format!("{} row {} has invalid entities JSON", path.display(), i + 1))?;
        let gold = model_spans(row_text, &declared)
            .with_context(|| format!("{} row {}", path.display(), i + 1))?;
        let mut enc = encode_document(row_text, &gold, fc)
            .with_context(|| format!("{} row {}", path.display(), i + 1))?;
        enc.country = row_country.to_string();
        docs.push(DetectorDoc {
            text: row_text.to_owned(),
            enc,
            gold,
            breaks: breaks_of(row_text),
        });
        counts.encoded += 1;
    }
    Ok((docs, counts))
}

#[derive(Deserialize)]
struct SilverJson {
    text: String,
    country: String,
    entities: Vec<GoldJson>,
}

/// What loading the silver documents kept and dropped.
#[derive(Debug, Default, Clone, Serialize)]
pub struct SilverCounts {
    pub documents: usize,
    /// Exact-text copies with identical detector labels that were not encoded again.
    pub duplicate_documents: usize,
    pub pieces: usize,
    pub spans: usize,
    /// Spans no BIO tagging can produce. A nonzero count rejects the whole input before training.
    pub unreachable_spans: usize,
    /// Actual distinct training supervision after deduplication, by country and kind.
    pub by_country: BTreeMap<String, SilverCountryCounts>,
    /// Encoded pieces from each input path, in the same order as `paths`.
    pub source_pieces: Vec<usize>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SilverCountryCounts {
    pub documents: usize,
    pub pieces: usize,
    pub person: usize,
    pub org: usize,
    pub address: usize,
}

/// Reads silver-labelled real documents and encodes them. A document longer than
/// `max_tokens` content tokens is cut at paragraph breaks that no span crosses, as the
/// synthetic documents are never longer.
pub fn load_silver(
    paths: &[String],
    fc: &FeatureConfig,
    max_tokens: usize,
) -> anyhow::Result<(Vec<DetectorDoc>, SilverCounts)> {
    let mut docs = Vec::new();
    let mut counts = SilverCounts::default();
    let mut seen: HashMap<[u8; 32], (Vec<KindSpan>, String)> = HashMap::new();
    let mut unreachable_details = Vec::new();
    for path in paths {
        let pieces_before = docs.len();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        for (line_no, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let doc: SilverJson = serde_json::from_str(line)
                .with_context(|| format!("{path}:{} invalid silver JSON", line_no + 1))?;
            ensure!(
                doc.country == "US",
                "{path}:{} has country {:?}; detector silver must be US-only",
                line_no + 1,
                doc.country
            );
            let gold = model_spans(&doc.text, &doc.entities)
                .with_context(|| format!("{path}:{}", line_no + 1))?;
            let mut sorted_gold = gold.clone();
            sorted_gold.sort_by_key(|span| (span.start, span.end, span.kind));
            let text_hash: [u8; 32] = Sha256::digest(doc.text.as_bytes()).into();
            if let Some((previous, location)) = seen.get(&text_hash) {
                ensure!(
                    previous == &sorted_gold,
                    "{path}:{} conflicts with {location}: identical silver text has different detector labels",
                    line_no + 1
                );
                counts.duplicate_documents += 1;
                continue;
            }
            seen.insert(text_hash, (sorted_gold, format!("{path}:{}", line_no + 1)));
            counts.documents += 1;
            let per_country = counts.by_country.entry(doc.country.clone()).or_default();
            per_country.documents += 1;
            for (start, end) in pieces(&doc.text, &gold, max_tokens) {
                let piece = &doc.text[start..end];
                let content_tokens = tokenize(piece)
                    .iter()
                    .filter(|token| is_content(token))
                    .count();
                ensure!(
                    content_tokens <= max_tokens,
                    "{path}:{} silver piece has {content_tokens} content tokens above limit {max_tokens}; split the source text at a safe paragraph boundary before training",
                    line_no + 1
                );
                let rebased: Vec<KindSpan> = gold
                    .iter()
                    .filter(|g| g.start as usize >= start && g.end as usize <= end)
                    .map(|g| KindSpan {
                        kind: g.kind,
                        start: g.start - start as u32,
                        end: g.end - start as u32,
                    })
                    .collect();
                let reachable = reachable_spans(piece, &rebased);
                for unreachable in rebased.iter().filter(|span| !reachable.contains(span)) {
                    unreachable_details.push(format!(
                        "{path}:{} {} {}..{} ({})",
                        line_no + 1,
                        KINDS[unreachable.kind].as_str(),
                        unreachable.start + start as u32,
                        unreachable.end + start as u32,
                        unreachable_reason(piece, unreachable),
                    ));
                }
                counts.spans += reachable.len();
                counts.unreachable_spans += rebased.len() - reachable.len();
                per_country.pieces += 1;
                for span in &reachable {
                    match span.kind {
                        0 => per_country.person += 1,
                        1 => per_country.org += 1,
                        2 => per_country.address += 1,
                        _ => unreachable!(),
                    }
                }
                let mut enc = encode_document(piece, &reachable, fc)
                    .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
                enc.country = doc.country.clone();
                docs.push(DetectorDoc {
                    text: piece.to_owned(),
                    enc,
                    gold: reachable,
                    breaks: breaks_of(piece),
                });
                counts.pieces += 1;
            }
        }
        counts.source_pieces.push(docs.len() - pieces_before);
    }
    ensure!(
        unreachable_details.is_empty(),
        "{} silver spans are unreachable by the detector encoder; review labels or tokenization before training:\n{}",
        unreachable_details.len(),
        unreachable_details.join("\n")
    );
    Ok((docs, counts))
}

fn unreachable_reason(text: &str, span: &KindSpan) -> String {
    let (start, end) = (span.start as usize, span.end as usize);
    let tokens = tokenize(text);
    let mut reasons = Vec::new();
    if !tokens
        .iter()
        .filter(|token| is_content(token))
        .any(|token| token.start == start)
        || !tokens
            .iter()
            .filter(|token| is_content(token))
            .any(|token| token.end == end)
    {
        reasons.push("token_boundary");
    }
    let retained: Vec<_> = tokens.iter().filter(|token| is_content(token)).collect();
    if retained
        .iter()
        .zip(breaks_of(text))
        .any(|(token, is_break)| is_break && start < token.start && token.start < end)
    {
        reasons.push("blank_line");
    }
    if scan_rules(text, &[])
        .iter()
        .any(|rule| start < rule.end && rule.start < end)
    {
        reasons.push("rule_overlap");
    }
    if retained
        .iter()
        .filter(|token| start <= token.start && token.end <= end)
        .count()
        > MAX_ENTITY_TOKENS
    {
        reasons.push("entity_too_long");
    }
    if reasons.is_empty() {
        reasons.push("unknown");
    }
    reasons.join(",")
}

/// The byte ranges `text` is cut into: whole paragraphs, at most `max_tokens` content tokens
/// each unless one paragraph alone is longer, never cutting through a span.
fn pieces(text: &str, gold: &[KindSpan], max_tokens: usize) -> Vec<(usize, usize)> {
    let tokens = tokenize(text);
    let content = |s: usize, e: usize| {
        tokens
            .iter()
            .filter(|t| t.start >= s && t.end <= e && is_content(t))
            .count()
    };
    let mut cuts: Vec<usize> = text
        .match_indices("\n\n")
        .map(|(i, _)| i + 2)
        .filter(|&c| {
            !gold
                .iter()
                .any(|g| (g.start as usize) < c && c < g.end as usize)
        })
        .collect();
    cuts.push(text.len());
    let mut out = Vec::new();
    let (mut start, mut last) = (0, 0);
    for c in cuts {
        if last > start && content(start, c) > max_tokens {
            out.push((start, last));
            start = last;
        }
        last = c;
    }
    if last > start {
        out.push((start, last));
    }
    out
}

/// The spans of `gold` a tagger can produce on `text`: on token boundaries, within one
/// paragraph (the decoder closes every span at a blank line), at most the decoder's span
/// length, and clear of the rules layer's emails and phones.
fn reachable_spans(text: &str, gold: &[KindSpan]) -> Vec<KindSpan> {
    let tokens = tokenize(text);
    let retained_indices: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| is_content(token))
        .map(|(index, _)| index)
        .collect();
    let break_positions: Vec<_> = paragraph_breaks(&tokens, &retained_indices)
        .into_iter()
        .zip(&retained_indices)
        .filter_map(|(is_break, &index)| is_break.then_some(tokens[index].start))
        .collect();
    let rules: Vec<(usize, usize)> = scan_rules(text, &[])
        .iter()
        .map(|e| (e.start, e.end))
        .collect();
    gold.iter()
        .filter(|g| {
            let (s, e) = (g.start as usize, g.end as usize);
            retained_indices
                .iter()
                .any(|&index| tokens[index].start == s)
                && retained_indices.iter().any(|&index| tokens[index].end == e)
                && retained_indices
                    .iter()
                    .filter(|&&index| s <= tokens[index].start && tokens[index].end <= e)
                    .count()
                    <= MAX_ENTITY_TOKENS
                && !break_positions.iter().any(|&at| s < at && at < e)
                && !rules.iter().any(|&(rs, re)| s < re && rs < e)
        })
        .copied()
        .collect()
}

/// Whether a paragraph break comes before each retained token of `text`, as the library's
/// decoder reads them.
pub fn breaks_of(text: &str) -> Vec<bool> {
    let tokens = tokenize(text);
    let retained: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| is_content(t))
        .map(|(i, _)| i)
        .collect();
    paragraph_breaks(&tokens, &retained)
}

fn predicted_span(text: &str, kind: usize, start: u32, raw_end: u32) -> KindSpan {
    let end = if KINDS[kind] == Kind::Address {
        normalized_us_address_end(text, start as usize, raw_end as usize) as u32
    } else {
        raw_end
    };
    KindSpan { kind, start, end }
}

/// Predicted spans per document in byte offsets, thresholded by `DETECT_MIN`.
pub fn predict<B: Backend>(
    model: &TaggerNet<B>,
    docs: &[DetectorDoc],
    batch_size: usize,
    device: &B::Device,
) -> Vec<Vec<KindSpan>> {
    let mut out = Vec::with_capacity(docs.len());
    for chunk in docs.chunks(batch_size.max(1)) {
        let items: Vec<Encoded> = chunk.iter().map(|d| d.enc.clone()).collect();
        let batch: ParserBatch<B> = ParserBatcher.batch(items, device);
        let logits = model.forward(
            batch.ngram_ids,
            batch.script,
            batch.shape,
            batch.flags,
            batch.mask,
        );
        let [_, l, c] = logits.dims();
        let probs: Vec<f32> = softmax(logits, 2).into_data().to_vec().unwrap_or_default();
        for (i, d) in chunk.iter().enumerate() {
            let n = d.enc.token_spans.len();
            let rows = &probs[i * l * c..(i * l + n) * c];
            let masked: Vec<bool> = d
                .enc
                .flags
                .iter()
                .map(|f| f & flag::IN_RULE_SPAN != 0)
                .collect();
            out.push(
                decode_detector(rows, &masked, &d.breaks)
                    .into_iter()
                    .filter_map(|s| {
                        let kind = kind_index(s.kind)?;
                        (s.confidence >= DETECT_MIN[kind]).then(|| {
                            predicted_span(
                                &d.text,
                                kind,
                                d.enc.token_spans[s.first].0,
                                d.enc.token_spans[s.last].1,
                            )
                        })
                    })
                    .collect(),
            );
        }
    }
    out
}

/// Precision, recall, and F1.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Prf {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl Prf {
    fn from_counts(tp: usize, pred: usize, gold: usize) -> Prf {
        let (precision, recall, f1) = crate::eval::prf(tp, pred, gold);
        Prf {
            precision,
            recall,
            f1,
        }
    }
}

/// Scores of one kind: exact spans, lenient spans (same kind, overlap over union at least
/// one half), and the share of lenient matches whose boundaries are exact.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct KindScores {
    pub exact: Prf,
    pub lenient: Prf,
    pub boundary_accuracy: f64,
    pub gold: usize,
}

/// Micro-averaged scores per kind and the macro exact F1 over the three kinds.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SpanScores {
    pub per_kind: BTreeMap<&'static str, KindScores>,
    pub macro_exact_f1: f64,
}

fn iou(a: KindSpan, b: KindSpan) -> f64 {
    let inter = a.end.min(b.end).saturating_sub(a.start.max(b.start));
    let union = a.end.max(b.end) - a.start.min(b.start);
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Scores predictions against gold, document by document. Each prediction matches at most
/// one gold span and the reverse, so repeated overlapping predictions are not all credited.
fn matched_spans(
    gold: &[KindSpan],
    pred: &[KindSpan],
    matches: impl Fn(KindSpan, KindSpan) -> bool,
) -> usize {
    fn assign(
        p: usize,
        edges: &[Vec<usize>],
        seen: &mut [bool],
        owner: &mut [Option<usize>],
    ) -> bool {
        for &g in &edges[p] {
            if seen[g] {
                continue;
            }
            seen[g] = true;
            if owner[g].is_none_or(|previous| assign(previous, edges, seen, owner)) {
                owner[g] = Some(p);
                return true;
            }
        }
        false
    }

    let edges: Vec<Vec<usize>> = pred
        .iter()
        .map(|p| {
            gold.iter()
                .enumerate()
                .filter_map(|(g, span)| matches(*p, *span).then_some(g))
                .collect()
        })
        .collect();
    let mut owner = vec![None; gold.len()];
    for p in 0..pred.len() {
        let mut seen = vec![false; gold.len()];
        assign(p, &edges, &mut seen, &mut owner);
    }
    owner.iter().filter(|owner| owner.is_some()).count()
}

pub fn score(gold: &[Vec<KindSpan>], pred: &[Vec<KindSpan>]) -> SpanScores {
    assert_eq!(
        gold.len(),
        pred.len(),
        "detector evaluation needs one prediction list per document"
    );
    #[derive(Default, Clone, Copy)]
    struct Counts {
        exact: usize,
        lenient: usize,
        pred: usize,
        gold: usize,
    }
    let mut counts = [Counts::default(); 3];
    for (g, p) in gold.iter().zip(pred) {
        for (k, c) in counts.iter_mut().enumerate() {
            let gk: Vec<KindSpan> = g.iter().filter(|s| s.kind == k).copied().collect();
            let pk: Vec<KindSpan> = p.iter().filter(|s| s.kind == k).copied().collect();
            c.pred += pk.len();
            c.gold += gk.len();
            c.exact += matched_spans(&gk, &pk, |p, g| p == g);
            c.lenient += matched_spans(&gk, &pk, |p, g| iou(p, g) >= 0.5);
        }
    }
    let mut out = SpanScores::default();
    for (k, c) in counts.iter().enumerate() {
        let scores = KindScores {
            exact: Prf::from_counts(c.exact, c.pred, c.gold),
            lenient: Prf::from_counts(c.lenient, c.pred, c.gold),
            boundary_accuracy: if c.lenient == 0 {
                0.0
            } else {
                c.exact as f64 / c.lenient as f64
            },
            gold: c.gold,
        };
        out.per_kind.insert(KINDS[k].as_str(), scores);
    }
    out.macro_exact_f1 =
        out.per_kind.values().map(|s| s.exact.f1).sum::<f64>() / KINDS.len() as f64;
    out
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use polars::prelude::{Column, DataFrame, ParquetWriter};

    use super::*;
    use crate::train::masked_loss;

    fn span(kind: usize, text: &str, part: &str) -> KindSpan {
        let start = text.find(part).unwrap() as u32;
        KindSpan {
            kind,
            start,
            end: start + part.len() as u32,
        }
    }

    fn declared(kind: &str, start: u32, end: u32) -> GoldJson {
        GoldJson {
            kind: kind.into(),
            start,
            end,
        }
    }

    #[test]
    fn declared_detector_spans_fail_closed_before_encoding() {
        let text = "Nino Straße";
        for (name, spans) in [
            ("unknown", vec![declared("unknown", 0, 4)]),
            ("empty", vec![declared("person", 0, 0)]),
            ("past end", vec![declared("person", 0, 80)]),
            ("inside UTF-8", vec![declared("address", 5, 10)]),
            (
                "overlap",
                vec![declared("person", 0, 4), declared("org", 2, 4)],
            ),
        ] {
            assert!(model_spans(text, &spans).is_err(), "{name}");
        }
        let good = model_spans(
            "Nino nino@example.org Berlin",
            &[
                declared("person", 0, 4),
                declared("email", 5, 21),
                declared("address", 22, 28),
            ],
        )
        .unwrap();
        assert_eq!(good.len(), 2);
        assert_eq!(good[0].kind, 0);
        assert_eq!(good[1].kind, 2);
    }

    #[test]
    fn silver_loader_reports_path_line_and_label() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        std::fs::write(
            &path,
            "{\"text\":\"Nino\",\"country\":\"US\",\"entities\":[{\"kind\":\"person\",\"start\":0,\"end\":4}]}\n{\"text\":\"Nino\",\"country\":\"US\",\"entities\":[{\"kind\":\"bogus\",\"start\":0,\"end\":4}]}\n",
        )
        .unwrap();
        let error = format!(
            "{:#}",
            load_silver(
                &[path.display().to_string()],
                &FeatureConfig::default(),
                900
            )
            .unwrap_err()
        );
        assert!(error.contains("silver.jsonl:2"), "{error}");
        assert!(error.contains("entity 0 has unknown kind"), "{error}");
    }

    #[test]
    fn silver_loader_rejects_non_us_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        std::fs::write(
            &path,
            "{\"text\":\"London\",\"country\":\"GB\",\"entities\":[]}\n",
        )
        .unwrap();
        let error = load_silver(
            &[path.display().to_string()],
            &FeatureConfig::default(),
            900,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("silver.jsonl:1 has country \"GB\""),
            "{error}"
        );
    }

    #[test]
    fn silver_loader_skips_identical_copies_and_rejects_conflicting_labels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        let person = serde_json::json!({
            "text": "Nino", "country": "US",
            "entities": [{"kind": "person", "start": 0, "end": 4}]
        });
        std::fs::write(&path, format!("{person}\n{person}\n")).unwrap();
        let (docs, counts) = load_silver(
            &[path.display().to_string()],
            &FeatureConfig::default(),
            900,
        )
        .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(counts.documents, 1);
        assert_eq!(counts.duplicate_documents, 1);
        assert_eq!(counts.by_country["US"].documents, 1);
        assert_eq!(counts.by_country["US"].person, 1);

        let unlabeled = serde_json::json!({
            "text": "Nino", "country": "US", "entities": []
        });
        let other = dir.path().join("other.jsonl");
        std::fs::write(&other, format!("{unlabeled}\n")).unwrap();
        let error = format!(
            "{:#}",
            load_silver(
                &[path.display().to_string(), other.display().to_string()],
                &FeatureConfig::default(),
                900,
            )
            .unwrap_err()
        );
        assert!(error.contains("other.jsonl:1 conflicts with"), "{error}");
        assert!(error.contains("silver.jsonl:1"), "{error}");
    }

    #[test]
    fn silver_loader_rejects_unreachable_positive_instead_of_training_it_as_negative() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        let row = serde_json::json!({
            "text": "the office opened", "country": "US",
            "entities": [{"kind": "org", "start": 5, "end": 8}]
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();
        let error = format!(
            "{:#}",
            load_silver(
                &[path.display().to_string()],
                &FeatureConfig::default(),
                900,
            )
            .unwrap_err()
        );
        assert!(error.contains("silver.jsonl:1"), "{error}");
        assert!(error.contains("org 5..8 (token_boundary)"), "{error}");
    }

    #[test]
    fn silver_loader_and_encoder_reject_spans_longer_than_decoder_limit() {
        let at_limit = vec!["office"; MAX_ENTITY_TOKENS].join(" ");
        assert!(
            encode_document(
                &at_limit,
                &[KindSpan {
                    kind: 1,
                    start: 0,
                    end: at_limit.len() as u32,
                }],
                &FeatureConfig::default()
            )
            .is_ok()
        );
        let text = vec!["office"; MAX_ENTITY_TOKENS + 1].join(" ");
        let target = KindSpan {
            kind: 1,
            start: 0,
            end: text.len() as u32,
        };
        assert!(matches!(
            encode_document(&text, &[target], &FeatureConfig::default()),
            Err(DetectorEncodeError::EntityTooLong(span, count))
                if span == target && count == MAX_ENTITY_TOKENS + 1
        ));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        let row = serde_json::json!({
            "text": text, "country": "US",
            "entities": [{"kind": "org", "start": 0, "end": target.end}]
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();
        let error = format!(
            "{:#}",
            load_silver(
                &[path.display().to_string()],
                &FeatureConfig::default(),
                900
            )
            .unwrap_err()
        );
        assert!(error.contains("silver.jsonl:1"), "{error}");
        assert!(error.contains("entity_too_long"), "{error}");
    }

    #[test]
    fn silver_loader_rejects_one_long_paragraph_above_piece_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("silver.jsonl");
        let row = serde_json::json!({
            "text": "one two three", "country": "US", "entities": []
        });
        std::fs::write(&path, format!("{row}\n")).unwrap();
        let error = format!(
            "{:#}",
            load_silver(&[path.display().to_string()], &FeatureConfig::default(), 2,).unwrap_err()
        );
        assert!(error.contains("silver.jsonl:1"), "{error}");
        assert!(error.contains("3 content tokens above limit 2"), "{error}");
    }

    #[test]
    fn synthetic_loader_reports_path_row_and_label() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("text".into(), vec!["Nino"]),
            Column::new(
                "entities_json".into(),
                vec!["[{\"kind\":\"bogus\",\"start\":0,\"end\":4}]"],
            ),
            Column::new("country".into(), vec!["US"]),
        ])
        .unwrap();
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        let error = format!(
            "{:#}",
            load_split(dir.path(), Split::Train, &FeatureConfig::default()).unwrap_err()
        );
        assert!(error.contains("train.parquet row 1"), "{error}");
        assert!(error.contains("entity 0 has unknown kind"), "{error}");
    }

    #[test]
    fn synthetic_loader_rejects_unreachable_gold_instead_of_skipping_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("text".into(), vec!["Nino"]),
            Column::new(
                "entities_json".into(),
                vec!["[{\"kind\":\"person\",\"start\":1,\"end\":4}]"],
            ),
            Column::new("country".into(), vec!["US"]),
        ])
        .unwrap();
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        let error = format!(
            "{:#}",
            load_split(dir.path(), Split::Train, &FeatureConfig::default()).unwrap_err()
        );
        assert!(error.contains("train.parquet row 1"), "{error}");
        assert!(error.contains("token boundary"), "{error}");
    }

    #[test]
    fn synthetic_loader_rejects_invalid_country() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("text".into(), vec!["Nino"]),
            Column::new("entities_json".into(), vec!["[]"]),
            Column::new("country".into(), vec!["??"]),
        ])
        .unwrap();
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        let error = format!(
            "{:#}",
            load_split(dir.path(), Split::Train, &FeatureConfig::default()).unwrap_err()
        );
        assert!(error.contains("train.parquet row 1"), "{error}");
        assert!(error.contains("invalid country"), "{error}");
    }

    #[test]
    fn synthetic_loader_rejects_non_us_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("text".into(), vec!["London"]),
            Column::new("entities_json".into(), vec!["[]"]),
            Column::new("country".into(), vec!["GB"]),
        ])
        .unwrap();
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        let error = load_split(dir.path(), Split::Train, &FeatureConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("detector data must be US-only"), "{error}");
    }

    #[test]
    fn bio_labels_cover_retained_tokens() {
        let text = "Nino Beridze\nKavkaz Freight LLC";
        let gold = [
            span(0, text, "Nino Beridze"),
            span(1, text, "Kavkaz Freight LLC"),
        ];
        let enc = encode_document(text, &gold, &FeatureConfig::default()).unwrap();
        assert_eq!(enc.labels, vec![1, 2, 3, 4, 4]);
    }

    #[test]
    fn a_span_starting_mid_token_is_rejected() {
        let text = "Nino Beridze";
        let gold = [KindSpan {
            kind: 0,
            start: 1,
            end: 12,
        }];
        assert!(matches!(
            encode_document(text, &gold, &FeatureConfig::default()),
            Err(DetectorEncodeError::PartialToken(_))
        ));
    }

    #[test]
    fn silver_pieces_cut_at_paragraphs_outside_spans() {
        let text = "one two three\n\nfour five\nsix\n\nseven eight";
        let whole = [span(0, text, "five\nsix")];
        assert_eq!(pieces(text, &whole, 100), vec![(0, text.len())]);
        let p = pieces(text, &whole, 3);
        assert_eq!(p, vec![(0, 15), (15, 30), (30, text.len())]);
        let across = [KindSpan {
            kind: 0,
            start: 4,
            end: 21,
        }];
        assert_eq!(pieces(text, &across, 3), vec![(0, 30), (30, text.len())]);
    }

    #[test]
    fn silver_spans_inside_tokens_or_over_rules_are_unreachable() {
        let text = "the EPA's office, EPA staff, mail epa@epa.example";
        let gold = [
            span(1, text, "EPA's"),
            KindSpan {
                kind: 1,
                start: 4,
                end: 7,
            },
            KindSpan {
                kind: 1,
                start: 10,
                end: 13,
            },
            span(1, text, "EPA staff"),
            span(1, text, "epa@epa.example"),
        ];
        assert_eq!(
            reachable_spans(text, &gold),
            vec![gold[0], gold[1], gold[3]],
            "a possessive is a token of its own, the middle of `office` is not"
        );
        let broken = "7290 Investment Drive, South\n\n[[Page 7]]\n\nCarolina 29418";
        assert!(reachable_spans(broken, &[span(2, broken, broken)]).is_empty());
        for separated in ["12 Main St\r\rTbilisi", "12 Main St\u{2028}\u{2028}Tbilisi"] {
            assert!(
                reachable_spans(separated, &[span(2, separated, separated)]).is_empty(),
                "runtime closes spans at this paragraph break: {separated:?}"
            );
        }
        let leading_space = " Nino";
        assert!(
            reachable_spans(leading_space, &[span(0, leading_space, leading_space)]).is_empty(),
            "a span starting on skipped whitespace cannot be encoded"
        );
    }

    #[test]
    fn a_span_over_a_rule_email_is_rejected() {
        let text = "write to nino@kavkaz.example today";
        let gold = [span(0, text, "nino@kavkaz.example")];
        assert!(matches!(
            encode_document(text, &gold, &FeatureConfig::default()),
            Err(DetectorEncodeError::RuleOverlap(_))
        ));
    }

    #[test]
    fn padding_does_not_change_the_loss() {
        let device = Default::default();
        let weights =
            Tensor::<NdArray, 1>::from_floats([1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5], &device);
        let logits = Tensor::<NdArray, 3>::random(
            [1, 2, DETECTOR_LABELS],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let labels = Tensor::<NdArray, 2, Int>::from_ints([[3, 0]], &device);
        let mask = Tensor::<NdArray, 2>::from_floats([[1.0, 0.0]], &device);
        let both: f32 = masked_loss(logits.clone(), labels, mask, weights.clone()).into_scalar();
        let first: f32 = masked_loss(
            logits.slice([0..1, 0..1]),
            Tensor::from_ints([[3]], &device),
            Tensor::from_floats([[1.0]], &device),
            weights,
        )
        .into_scalar();
        assert!((both - first).abs() < 1e-6, "{both} vs {first}");
    }

    #[test]
    fn detector_network_shapes_and_size() {
        let device = Default::default();
        let net = crate::net::TaggerNetConfig::new(32768, vec![1, 2, 4, 8, 16, 1], DETECTOR_LABELS)
            .init::<NdArray>(&device);
        let out = net.forward(
            Tensor::zeros([2, 10, 24], &device),
            Tensor::zeros([2, 10], &device),
            Tensor::zeros([2, 10], &device),
            Tensor::zeros([2, 10, crate::dataset::FLAG_BITS], &device),
            Tensor::ones([2, 10], &device),
        );
        assert_eq!(out.dims(), [2, 10, DETECTOR_LABELS]);
        // proj 87*96 + 96, six blocks of 96*96*3 + 96, head 96*7 + 7.
        assert_eq!(net.non_embedding_params(), 8_448 + 6 * 27_744 + 679);
    }

    #[test]
    fn scores_count_exact_and_lenient_matches() {
        let g = |k, s, e| KindSpan {
            kind: k,
            start: s,
            end: e,
        };
        let gold = vec![vec![g(0, 0, 10), g(1, 20, 30)]];
        let pred = vec![vec![g(0, 0, 10), g(1, 20, 28), g(2, 40, 50)]];
        let s = score(&gold, &pred);
        let person = &s.per_kind["person"];
        assert_eq!(person.exact.f1, 1.0);
        let org = &s.per_kind["org"];
        assert_eq!(org.exact.f1, 0.0);
        assert_eq!(org.lenient.f1, 1.0);
        assert_eq!(org.boundary_accuracy, 0.0);
        assert_eq!(s.per_kind["address"].exact.precision, 0.0);
    }

    #[test]
    fn checkpoint_prediction_uses_runtime_zip4_boundary() {
        let text = "550 17th Street NW, Washington, DC 20429-0146, next";
        let zip_start = text.find("-0146").unwrap() as u32;
        let gold = KindSpan {
            kind: 2,
            start: 0,
            end: zip_start + 5,
        };
        for raw_end in [zip_start, zip_start + 1, zip_start + 6] {
            let pred = predicted_span(text, 2, 0, raw_end);
            assert_eq!(pred, gold);
            assert_eq!(
                score(&[vec![gold]], &[vec![pred]]).per_kind["address"]
                    .exact
                    .f1,
                1.0
            );
        }
        assert_eq!(predicted_span(text, 0, 0, zip_start).end, zip_start);
    }

    #[test]
    fn duplicate_predictions_match_one_gold_only_once() {
        let span = KindSpan {
            kind: 0,
            start: 0,
            end: 10,
        };
        let scores = score(&[vec![span]], &[vec![span, span]]);
        let person = &scores.per_kind["person"];
        assert_eq!(person.exact.precision, 0.5);
        assert_eq!(person.exact.recall, 1.0);
        assert_eq!(person.lenient.precision, 0.5);
        assert_eq!(person.lenient.recall, 1.0);
        assert!(person.exact.f1 <= 1.0);
    }

    #[test]
    fn lenient_matching_finds_maximum_one_to_one_assignment() {
        let s = |start, end| KindSpan {
            kind: 0,
            start,
            end,
        };
        let scores = score(&[vec![s(0, 10), s(10, 20)]], &[vec![s(0, 20), s(0, 10)]]);
        assert_eq!(scores.per_kind["person"].lenient.f1, 1.0);
    }

    #[test]
    #[should_panic(expected = "one prediction list per document")]
    fn score_rejects_missing_prediction_documents() {
        score(&[vec![]], &[]);
    }
}
