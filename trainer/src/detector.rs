//! The entity detector's data and scores: synthetic documents encoded with the rules layer's
//! email and phone spans as features, BIO labels over person, org, and address, greedy
//! decoding that keeps rule spans out of model spans, and exact and lenient span scores.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use polars::prelude::{ParquetReader, SerReader};
use serde::{Deserialize, Serialize};
use tessera::Kind;
use tessera::internal::{
    DETECT_MIN_ADDRESS, DETECT_MIN_ORG, DETECT_MIN_PERSON, FeatureConfig, decode_detector,
    featurize, flag, is_content, paragraph_breaks, scan_rules, tokenize,
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

/// Documents skipped while loading a split, by reason.
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
        let row_text = text.get(i).context("null text")?;
        let gold: Vec<KindSpan> =
            serde_json::from_str::<Vec<GoldJson>>(entities.get(i).context("null entities")?)?
                .into_iter()
                .filter_map(|g| {
                    let kind = kind_index(Kind::from_str_label(&g.kind)?)?;
                    Some(KindSpan {
                        kind,
                        start: g.start,
                        end: g.end,
                    })
                })
                .collect();
        match encode_document(row_text, &gold, fc) {
            Ok(mut enc) => {
                enc.country = country.get(i).unwrap_or_default().to_string();
                docs.push(DetectorDoc {
                    enc,
                    gold,
                    breaks: breaks_of(row_text),
                });
                counts.encoded += 1;
            }
            Err(DetectorEncodeError::PartialToken(_)) => counts.partial_token += 1,
            Err(DetectorEncodeError::RuleOverlap(_)) => counts.rule_overlap += 1,
        }
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
    pub pieces: usize,
    pub spans: usize,
    /// Spans no BIO tagging can produce: inside a token (`EPA` in `EPA's`) or over a rule
    /// email or phone. Their tokens stay `O`, which is the cost of keeping the document.
    pub unreachable_spans: usize,
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
    for path in paths {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let doc: SilverJson =
                serde_json::from_str(line).with_context(|| format!("a line of {path}"))?;
            counts.documents += 1;
            let gold: Vec<KindSpan> = doc
                .entities
                .iter()
                .filter_map(|g| {
                    Some(KindSpan {
                        kind: kind_index(Kind::from_str_label(&g.kind)?)?,
                        start: g.start,
                        end: g.end,
                    })
                })
                .collect();
            for (start, end) in pieces(&doc.text, &gold, max_tokens) {
                let piece = &doc.text[start..end];
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
                counts.spans += reachable.len();
                counts.unreachable_spans += rebased.len() - reachable.len();
                let mut enc = encode_document(piece, &reachable, fc)
                    .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
                enc.country = doc.country.clone();
                docs.push(DetectorDoc {
                    enc,
                    gold: reachable,
                    breaks: breaks_of(piece),
                });
                counts.pieces += 1;
            }
        }
    }
    Ok((docs, counts))
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
/// paragraph (the decoder closes every span at a blank line), and clear of the rules layer's
/// emails and phones.
fn reachable_spans(text: &str, gold: &[KindSpan]) -> Vec<KindSpan> {
    let tokens = tokenize(text);
    let rules: Vec<(usize, usize)> = scan_rules(text, &[])
        .iter()
        .map(|e| (e.start, e.end))
        .collect();
    gold.iter()
        .filter(|g| {
            let (s, e) = (g.start as usize, g.end as usize);
            tokens.iter().any(|t| t.start == s)
                && tokens.iter().any(|t| t.end == e)
                && !has_blank_line(&text[s..e])
                && !rules.iter().any(|&(rs, re)| s < re && rs < e)
        })
        .copied()
        .collect()
}

fn has_blank_line(s: &str) -> bool {
    let lines: Vec<&str> = s.split('\n').collect();
    lines.len() > 2
        && lines[1..lines.len() - 1]
            .iter()
            .any(|l| l.trim().is_empty())
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
                        (s.confidence >= DETECT_MIN[kind]).then(|| KindSpan {
                            kind,
                            start: d.enc.token_spans[s.first].0,
                            end: d.enc.token_spans[s.last].1,
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
pub fn score(gold: &[Vec<KindSpan>], pred: &[Vec<KindSpan>]) -> SpanScores {
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
            c.exact += pk.iter().filter(|s| gk.contains(s)).count();
            let mut used = vec![false; gk.len()];
            for s in &pk {
                if let Some(j) = (0..gk.len()).find(|&j| !used[j] && iou(*s, gk[j]) >= 0.5) {
                    used[j] = true;
                    c.lenient += 1;
                }
            }
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
            span(1, text, "EPA staff"),
            span(1, text, "epa@epa.example"),
        ];
        assert_eq!(reachable_spans(text, &gold), vec![gold[0], gold[2]]);
        let broken = "7290 Investment Drive, South\n\n[[Page 7]]\n\nCarolina 29418";
        assert!(reachable_spans(broken, &[span(2, broken, broken)]).is_empty());
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
}
