//! The model half of `detect`: features over the whole document, the detector per window,
//! spans kept where their window trusts them, merged with the rule spans, and each detected
//! address split into components by the parser.

use crate::{
    AddressLabel, Entity, Error, Kind, Model, Query, Source, Tessera, chunk, features, internal,
    model, policy, rules, token,
};

impl Tessera {
    /// The loaded model and its detector, or the `detect` stage's inference error when the
    /// bundle has no detector.
    pub(crate) fn detector(&self) -> Result<(&Model, &model::Tagger), Error> {
        let stage = Error::Inference {
            stage: policy::STAGE_DETECT,
        };
        let model = self.model.as_ref().ok_or(stage.clone())?;
        let detector = model.detector.as_ref().ok_or(stage)?;
        Ok((model, detector))
    }

    /// The detector's entities of the wanted kinds: features over the whole document once,
    /// the network per window, spans kept only where their window trusts them, merged across
    /// windows and against the rule spans, and address components attached by the parser.
    pub(crate) fn detect_model(
        &self,
        text: &str,
        rule_entities: &[Entity],
        model: &Model,
        detector: &model::Tagger,
        query: &Query<'_>,
        mask: Option<&chunk::Mask>,
    ) -> Result<Vec<Entity>, Error> {
        let DetectorInputs {
            tokens,
            retained,
            feats,
            masked,
            outside_mask,
        } = detector_inputs(text, rule_entities, model, mask);
        let breaks = chunk::paragraph_breaks(&tokens, &retained);
        let mut candidates = Vec::new();
        for w in chunk::windows(
            &tokens,
            &retained,
            chunk::WINDOW_TOKENS,
            detector.window_overlap(),
            mask.map(|_| outside_mask.as_slice()),
        )? {
            let range = w.tok_start..w.tok_end;
            let mut probs = detector.forward(&feats[range.clone()])?;
            model::kernels::softmax_rows(&mut probs, detector.labels());
            let bounds: Vec<_> = if detector.uses_address_continuation() {
                retained[range.clone()]
                    .iter()
                    .map(|&i| (tokens[i].start, tokens[i].end))
                    .collect()
            } else {
                Vec::new()
            };
            let decoded = crate::detector_postprocess::decode(
                detector.uses_address_continuation(),
                model.detector_postprocess_contract,
                text,
                &bounds,
                &probs,
                &masked[range.clone()],
                &breaks[range.clone()],
            )?;
            for s in decoded {
                let (first, last) = (w.tok_start + s.first, w.tok_start + s.last);
                if chunk::trusted_with_margin(&w, first, last, detector.context_margin())
                    && s.confidence >= policy::detect_min(s.kind)
                {
                    let mut entity =
                        span_entity(&tokens, &retained, s.kind, (first, last), s.confidence);
                    if s.kind == Kind::Address {
                        entity.end = normalized_us_address_end(text, entity.start, entity.end);
                    }
                    candidates.push(entity);
                }
            }
        }
        let merged = chunk::merge(
            candidates
                .into_iter()
                .chain(rule_entities.iter().cloned())
                .collect(),
            mask,
        );
        let mut out = Vec::new();
        for mut e in merged {
            if e.source != Source::Model || !self.kinds.contains(e.kind) {
                continue;
            }
            if e.kind == Kind::Address {
                self.attach_components(text, &mut e, query)?;
            }
            out.push(e);
        }
        Ok(out)
    }

    /// The detector over all of `text` as one window, with rule spans found without a country
    /// hint, exactly as the trainer encodes a legacy bundle's golden cases.
    pub(crate) fn detect_trace(&self, text: &str) -> Result<internal::DetectTrace, Error> {
        self.trace_with_rules(text, &rules::scan(text, &[]))
    }

    /// The detector over all of `text` as one window, with rule spans found as public
    /// plain-text detection finds them for `country_hint`, exactly as the trainer encodes the
    /// golden cases of a bundle whose detector declares that input policy.
    #[doc(hidden)]
    pub fn detect_trace_with_country_hint(
        &self,
        text: &str,
        country_hint: &[&str],
    ) -> Result<internal::DetectTrace, Error> {
        self.trace_with_rules(text, &rules::scan_text_rules(text, country_hint))
    }

    fn trace_with_rules(
        &self,
        text: &str,
        rule_entities: &[Entity],
    ) -> Result<internal::DetectTrace, Error> {
        let (model, detector) = self.detector()?;
        let inputs = detector_inputs(text, rule_entities, model, None);
        let logits = detector.forward(&inputs.feats)?;
        let mut probs = logits.clone();
        model::kernels::softmax_rows(&mut probs, detector.labels());
        let decoded = probs
            .chunks(detector.labels())
            .zip(&inputs.masked)
            .map(|(row, &m)| if m { 0 } else { model::bio::argmax(row) as u8 })
            .collect();
        Ok(internal::DetectTrace {
            token_spans: inputs
                .retained
                .iter()
                .map(|&i| (inputs.tokens[i].start, inputs.tokens[i].end))
                .collect(),
            features: inputs.feats,
            masked: inputs.masked,
            logits,
            decoded,
        })
    }

    /// Parses a detected address on its own text, as the parser was trained, and attaches its
    /// components, with the address's confidence bounded by their mean. An address too long to
    /// parse keeps no components.
    fn attach_components(
        &self,
        text: &str,
        e: &mut Entity,
        query: &Query<'_>,
    ) -> Result<(), Error> {
        let parsed = match self.parse_address(&text[e.start..e.end], query) {
            Ok(p) => p,
            Err(Error::InputTooLarge) => return Ok(()),
            Err(other) => return Err(other),
        };
        let confidences: Vec<f32> = parsed
            .components
            .iter()
            .filter(|c| c.label != AddressLabel::Unknown)
            .map(|c| c.confidence)
            .collect();
        if !confidences.is_empty() {
            let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
            e.confidence = e.confidence.min(mean);
        }
        e.components = parsed
            .components
            .into_iter()
            .map(|mut c| {
                c.start += e.start;
                c.end += e.start;
                c
            })
            .collect();
        Ok(())
    }
}

/// What the detector reads for one document: its tokens, the retained (non-whitespace)
/// positions, their features computed over the whole document, which of them the decoder must
/// read as `O` (inside a rule span or outside the mask), and which lie outside the mask.
struct DetectorInputs {
    tokens: Vec<token::Token>,
    retained: Vec<usize>,
    feats: Vec<features::TokenFeatures>,
    masked: Vec<bool>,
    outside_mask: Vec<bool>,
}

fn detector_inputs(
    text: &str,
    rule_entities: &[Entity],
    model: &Model,
    mask: Option<&chunk::Mask>,
) -> DetectorInputs {
    let tokens = stage!(Tokenize, token::tokenize(text));
    let rule_spans: Vec<(usize, usize)> = rule_entities.iter().map(|e| (e.start, e.end)).collect();
    let all_feats = stage!(
        Featurize,
        crate::detector_features::featurize_detector(
            text,
            &tokens,
            &rule_spans,
            None,
            &model.feature_config,
            mask,
            model.detector_feature_contract
        )
    );
    let retained: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| features::is_content(t))
        .map(|(i, _)| i)
        .collect();
    let feats: Vec<features::TokenFeatures> =
        retained.iter().map(|&i| all_feats[i].clone()).collect();
    let flagged = |bits: u32| -> Vec<bool> { feats.iter().map(|f| f.flags & bits != 0).collect() };
    let masked = flagged(features::flag::IN_RULE_SPAN | features::flag::MASKED);
    let outside_mask = flagged(features::flag::MASKED);
    DetectorInputs {
        tokens,
        retained,
        feats,
        masked,
        outside_mask,
    }
}

/// A model entity over retained positions `first..=last`, exactly as the detector decoded it.
fn span_entity(
    tokens: &[token::Token],
    retained: &[usize],
    kind: Kind,
    (first, last): (usize, usize),
    confidence: f32,
) -> Entity {
    Entity {
        kind,
        start: tokens[retained[first]].start,
        end: tokens[retained[last]].end,
        confidence,
        review_recommended: false,
        source: Source::Model,
        components: Vec::new(),
        normalized: None,
        region: None,
    }
}

pub(crate) fn normalized_us_address_end(text: &str, start: usize, end: usize) -> usize {
    let bytes = text.as_bytes();
    let zip5 = |at: usize| {
        at >= start + 5
            && bytes[at - 5..at].iter().all(u8::is_ascii_digit)
            && (at == start + 5 || !bytes[at - 6].is_ascii_digit())
    };
    let zip4 = |at: usize| {
        at >= start + 10
            && bytes[at - 5] == b'-'
            && zip5(at - 5)
            && bytes[at - 4..at].iter().all(u8::is_ascii_digit)
    };
    let mut end = end;
    if end > start && matches!(bytes[end - 1], b',' | b'.' | b';') {
        let before_punctuation = end - 1;
        if zip5(before_punctuation) || zip4(before_punctuation) {
            end = before_punctuation;
        }
    }
    let suffix_len = if zip5(end)
        && bytes.get(end) == Some(&b'-')
        && bytes
            .get(end + 1..end + 5)
            .is_some_and(|part| part.iter().all(u8::is_ascii_digit))
    {
        5
    } else if end > start
        && bytes[end - 1] == b'-'
        && zip5(end - 1)
        && bytes
            .get(end..end + 4)
            .is_some_and(|part| part.iter().all(u8::is_ascii_digit))
    {
        4
    } else {
        0
    };
    if suffix_len > 0
        && bytes
            .get(end + suffix_len)
            .is_none_or(|next| !next.is_ascii_alphanumeric() && *next != b'-')
    {
        end += suffix_len;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    const SIGNATURE: &str = "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example";

    #[test]
    fn address_end_completes_zip4_and_trims_postcode_punctuation() {
        let text = "550 17th Street NW, Washington, DC 20429-0146, next";
        let zip_end = text.find("-0146").unwrap();
        assert_eq!(normalized_us_address_end(text, 0, zip_end), zip_end + 5);
        assert_eq!(normalized_us_address_end(text, 0, zip_end + 1), zip_end + 5);
        assert_eq!(normalized_us_address_end(text, 0, zip_end + 6), zip_end + 5);
    }

    #[test]
    fn address_end_does_not_extend_a_partial_number_or_word() {
        let text = "Road 123456-7890A";
        assert_eq!(normalized_us_address_end(text, 0, 11), 11);
        let text = "Road 12345-7890A";
        assert_eq!(normalized_us_address_end(text, 0, 10), 10);
    }

    fn load_all(bundle: &[u8]) -> Result<Tessera, Error> {
        Tessera::load(
            bundle,
            Config {
                kinds: Kind::all(),
                expected_checksum: None,
            },
        )
    }

    #[test]
    fn detect_keeps_its_invariants_with_any_weights() {
        for seed in [3, 5, 8] {
            let t = load_all(&model::testing::with_random_detector(seed, 6)).unwrap();
            for include_uncertain in [false, true] {
                let query = Query {
                    country_hint: &["GE"],
                    include_uncertain,
                    ..Query::default()
                };
                let found = t.detect(SIGNATURE, &query).unwrap();
                for pair in found.windows(2) {
                    assert!(pair[0].end <= pair[1].start, "{pair:?}");
                }
                let rules: Vec<&Entity> =
                    found.iter().filter(|e| e.source == Source::Rules).collect();
                assert!(rules.iter().any(|e| e.kind == Kind::Email));
                // Without the phone tables a number with no country code is not scanned.
                #[cfg(feature = "phone-metadata")]
                assert!(rules.iter().any(|e| e.kind == Kind::Phone));
                for m in found.iter().filter(|e| e.source == Source::Model) {
                    assert!(rules.iter().all(|r| m.end <= r.start || r.end <= m.start));
                    assert!(
                        SIGNATURE.is_char_boundary(m.start) && SIGNATURE.is_char_boundary(m.end)
                    );
                }
            }
        }
    }

    #[test]
    fn rules_only_detect_needs_no_bundle() {
        let kinds = Kind::Email | Kind::Phone;
        let t = Tessera::load(
            &[],
            Config {
                kinds,
                expected_checksum: None,
            },
        )
        .unwrap();
        let query = Query {
            country_hint: &["GE"],
            ..Query::default()
        };
        let found = t.detect(SIGNATURE, &query).unwrap();
        let rules = policy::apply(rules::scan(SIGNATURE, &["GE"]), false);
        assert_eq!(found, rules);
    }

    #[test]
    fn people_need_a_detector_in_the_bundle() {
        let t = Tessera::load(
            &model::testing::parser_bundle(),
            Config {
                kinds: Kind::Person.into(),
                expected_checksum: None,
            },
        );
        assert_eq!(t.unwrap_err(), Error::BundleInvalid);
    }

    #[test]
    fn an_unbroken_document_is_too_large() {
        let t = load_all(&model::testing::with_random_detector(1, 6)).unwrap();
        let text = ".".repeat(6000);
        assert_eq!(
            t.detect(&text, &Query::default()).unwrap_err(),
            Error::InputTooLarge
        );
    }
}
