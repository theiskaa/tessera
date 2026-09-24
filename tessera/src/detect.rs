//! The model half of `detect`: features over the whole document, the detector per window,
//! spans trimmed and kept where their window trusts them, merged with the rule spans, and each
//! detected address checked by the parser.

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
    /// windows and against the rule spans, and addresses checked by the parser.
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
            chunk::OVERLAP_TOKENS,
            mask.map(|_| outside_mask.as_slice()),
        )? {
            let range = w.tok_start..w.tok_end;
            let mut probs = detector.forward(&feats[range.clone()]);
            model::kernels::softmax_rows(&mut probs, detector.labels());
            for s in model::bio::decode_detector(&probs, &masked[range.clone()], &breaks[range]) {
                let (first, last) = (w.tok_start + s.first, w.tok_start + s.last);
                if chunk::trusted(&w, first, last) && s.confidence >= policy::detect_min(s.kind) {
                    candidates.extend(span_entity(
                        text,
                        &tokens,
                        &retained,
                        s.kind,
                        (first, last),
                        s.confidence,
                    ));
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
        let taken: Vec<(usize, usize)> = merged.iter().map(|e| (e.start, e.end)).collect();
        let mut out = Vec::new();
        for mut e in merged {
            if e.source != Source::Model || !self.kinds.contains(e.kind) {
                continue;
            }
            if e.kind == Kind::Address && !self.check_address(text, &mut e, query)? {
                continue;
            }
            out.push(e);
        }
        if mask.is_none() {
            let repeats = repeated_names(text, &tokens, &out, &taken);
            out.extend(repeats);
            out.sort_by_key(|e| e.start);
        }
        Ok(out)
    }

    /// The detector over all of `text` as one window, with rule spans found without a country
    /// hint, exactly as the trainer encodes its golden cases.
    pub(crate) fn detect_trace(&self, text: &str) -> Result<internal::DetectTrace, Error> {
        let (model, detector) = self.detector()?;
        let inputs = detector_inputs(text, &rules::scan(text, &[]), model, None);
        let logits = detector.forward(&inputs.feats);
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

    /// Parses a detected address on its own text, as the parser was trained, and decides
    /// whether to keep it: at least two components with distinct labels (each at least
    /// `MEDIUM`, since weaker ones are `Unknown`) accept it, with confidence bounded by their
    /// mean; otherwise it is kept as uncertain only when the caller asks for uncertain results.
    fn check_address(&self, text: &str, e: &mut Entity, query: &Query<'_>) -> Result<bool, Error> {
        let parsed = match self.parse_address(&text[e.start..e.end], query) {
            Ok(p) => p,
            Err(Error::InputTooLarge) => return Ok(false),
            Err(other) => return Err(other),
        };
        let mut labels: Vec<AddressLabel> = parsed
            .components
            .iter()
            .map(|c| c.label)
            .filter(|l| *l != AddressLabel::Unknown)
            .collect();
        let labelled = labels.len();
        labels.sort_by_key(|l| *l as u8);
        labels.dedup();
        let components = parsed
            .components
            .into_iter()
            .map(|mut c| {
                c.start += e.start;
                c.end += e.start;
                c
            })
            .collect::<Vec<_>>();
        if labels.len() >= 2 {
            let mean = components
                .iter()
                .filter(|c| c.label != AddressLabel::Unknown)
                .map(|c| c.confidence)
                .sum::<f32>()
                / labelled as f32;
            e.confidence = e.confidence.min(mean);
            e.components = components;
            Ok(true)
        } else if query.include_uncertain {
            e.confidence = e.confidence.min(policy::MEDIUM - 0.01);
            e.components = components;
            Ok(true)
        } else {
            Ok(false)
        }
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
        features::featurize(
            text,
            &tokens,
            &rule_spans,
            None,
            &model.feature_config,
            mask
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

/// A model entity over retained positions `first..=last`, trimmed of what the model tends to
/// sweep in at the edges: quotes at either end, a bracket whose partner lies outside the span or
/// a pair wrapping all of it, separators (`,` `;` `:` and their full-width forms) at either
/// end, a trailing `。`, and a trailing `.` except after a short abbreviation on an org or an
/// address (`Acme Ltd.`, `Main St.`). `None` when nothing is left.
fn span_entity(
    text: &str,
    tokens: &[token::Token],
    retained: &[usize],
    kind: Kind,
    (mut first, mut last): (usize, usize),
    confidence: f32,
) -> Option<Entity> {
    const QUOTES: &[&str] = &[
        "\"", "'", "\u{201c}", "\u{201d}", "\u{2018}", "\u{2019}", "\u{ab}", "\u{bb}", "\u{201e}",
    ];
    const BRACKETS: &[(&str, &str)] = &[
        ("(", ")"),
        ("[", "]"),
        ("<", ">"),
        ("\u{ff08}", "\u{ff09}"),
        ("\u{300c}", "\u{300d}"),
        ("\u{300e}", "\u{300f}"),
        ("\u{3010}", "\u{3011}"),
    ];
    const SEPARATORS: &[&str] = &[
        ",", ";", ":", "\u{3001}", "\u{ff0c}", "\u{ff1b}", "\u{ff1a}",
    ];
    let at = |i: usize| {
        let t = &tokens[retained[i]];
        &text[t.start..t.end]
    };
    let holds =
        |range: std::ops::RangeInclusive<usize>, s: &str| range.into_iter().any(|i| at(i) == s);
    let abbreviation = |i: usize| {
        let word = at(i);
        word.chars().count() <= 4 && word.chars().all(char::is_alphabetic)
    };
    let mut article_trimmed = false;
    while first < last {
        if kind == Kind::Org && !article_trimmed && leading_article(at(first), at(first + 1)) {
            first += 1;
            article_trimmed = true;
            continue;
        }
        let opener = BRACKETS.iter().find(|(open, _)| at(first) == *open);
        let closer = BRACKETS.iter().find(|(_, close)| at(last) == *close);
        let period = at(last) == "\u{3002}"
            || (at(last) == "." && (kind == Kind::Person || !abbreviation(last - 1)));
        // A bracket opening at the end or closing at the start encloses nothing of the span.
        let dangling_open = BRACKETS.iter().any(|(open, _)| at(last) == *open);
        let dangling_close = BRACKETS.iter().any(|(_, close)| at(first) == *close);
        if QUOTES.contains(&at(first)) || SEPARATORS.contains(&at(first)) || dangling_close {
            first += 1;
        } else if QUOTES.contains(&at(last))
            || SEPARATORS.contains(&at(last))
            || period
            || dangling_open
        {
            last -= 1;
        } else if let Some((_, close)) = opener
            && ((at(last) == *close && !holds(first + 1..=last - 1, close))
                || !holds(first + 1..=last, close))
        {
            first += 1;
            if at(last) == *close {
                last -= 1;
            }
        } else if let Some((open, _)) = closer
            && !holds(first..=last - 1, open)
        {
            last -= 1;
        } else {
            break;
        }
    }
    // A bracket pair with nothing inside trims past itself.
    if first > last {
        return None;
    }
    let lone = at(first);
    if first == last
        && (QUOTES.contains(&lone)
            || (kind == Kind::Org && ARTICLES.contains(&lone.to_lowercase().as_str()))
            || SEPARATORS.contains(&lone)
            || lone == "."
            || lone == "\u{3002}"
            || BRACKETS.iter().any(|(o, c)| lone == *o || lone == *c))
    {
        return None;
    }
    let start = tokens[retained[first]].start;
    Some(Entity {
        kind,
        start,
        end: inside_last_token(text, start, tokens[retained[last]].end, kind),
        confidence,
        review_recommended: false,
        source: Source::Model,
        components: Vec::new(),
        normalized: None,
        region: None,
    })
}

/// Other mentions of the organizations and people in `found`: the same text on whole tokens,
/// or with a possessive after it (`NRC's`), clear of everything the detector and the rules
/// found (`taken`, of every kind). Documents name a body or a person in full once and then
/// repeat it (`HMRC` eighty times on one contact page, `Kobakhidze said`) where the context
/// around a repeat is too thin for the network alone. Only confident, distinctive names are
/// repeated, longest first, so a stray prediction does not spread: two or more words, an
/// acronym, or a name in a script without letter case, and a person's family name alone (see
/// `family_name`). A Han name is not repeated inside a longer run of Han characters
/// (`総務省令`).
fn repeated_names<'t>(
    text: &'t str,
    tokens: &[token::Token],
    found: &[Entity],
    taken: &[(usize, usize)],
) -> Vec<Entity> {
    // Capitalized words the document also writes in lower case outside every found span are
    // common words, not names: a role taken for a name (`Non-Executive Director`, `Service
    // Board`) must not spread. Particles (`von der`), initials, and scripts without capitals
    // are never common.
    let inside = |t: &token::Token| taken.iter().any(|&(s, e)| s < t.end && t.start < e);
    let lower: std::collections::HashSet<&str> = tokens
        .iter()
        .filter(|t| !inside(t))
        .map(|t| t.text(text))
        .filter(|w| w.chars().all(char::is_lowercase))
        .collect();
    let common = |word: &str| {
        let word = word.trim_end_matches([',', '.']);
        let capitalized = word.chars().next().is_some_and(char::is_uppercase)
            && word.chars().filter(|c| c.is_alphabetic()).count() >= 2;
        capitalized
            && (lower.contains(word.to_lowercase().as_str())
                || GLUED_ROLES.contains(&word)
                || ROLE_WORDS.contains(&word)
                || word.split('-').any(|part| GLUED_ROLES.contains(&part)))
    };
    // A person is repeated only when no word of the name is a common word.
    let person = |name: &'t str| -> Vec<&'t str> {
        if name.split_whitespace().any(common) {
            return Vec::new();
        }
        std::iter::once(name).chain(family_name(name)).collect()
    };
    let mut names: Vec<(&str, Kind, f32)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut family_names = std::collections::HashSet::new();
    for e in found.iter().filter(|e| e.confidence >= policy::HIGH) {
        let name = &text[e.start..e.end];
        let keys = match e.kind {
            Kind::Org => vec![name],
            Kind::Person => person(name),
            _ => continue,
        };
        for key in keys {
            if key != name {
                family_names.insert(key);
            }
            if (key != name || distinctive(key)) && seen.insert(key) {
                names.push((key, e.kind, e.confidence));
            }
        }
    }
    names.sort_by_key(|(name, _, _)| std::cmp::Reverse(name.len()));
    // A family name alone must stand alone: one followed by another capitalized word or a
    // number is part of a longer name or a date (`Paul Smith`, `3 May 2024`).
    let followed = |end: usize| {
        let rest = text[end..].trim_start_matches([' ', '\u{a0}']);
        rest.len() < text.len() - end
            && rest
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())
    };
    let mut by_first: std::collections::HashMap<&str, Vec<usize>> =
        std::collections::HashMap::new();
    for t in tokens {
        by_first.entry(t.text(text)).or_default().push(t.start);
    }
    // A name's first token may be the whole of a token that also carries a possessive.
    for t in tokens {
        let word = t.text(text);
        for possessive in ["'s", "\u{2019}s"] {
            if let Some(bare) = word.strip_suffix(possessive)
                && !bare.is_empty()
            {
                by_first.entry(bare).or_default().push(t.start);
            }
        }
    }
    let ends: std::collections::HashSet<usize> = tokens.iter().map(|t| t.end).collect();
    let mut blocked: std::collections::BTreeMap<usize, usize> = taken.iter().copied().collect();
    let overlaps = |blocked: &std::collections::BTreeMap<usize, usize>, at: usize, end: usize| {
        blocked
            .range(..end)
            .next_back()
            .is_some_and(|(_, &e)| e > at)
    };
    let han = |c: Option<char>| {
        c.is_some_and(|c| matches!(token::script_of(c as u32), Some(token::Script::Han)))
    };
    let mut out = Vec::new();
    for (name, kind, confidence) in names {
        let Some(first) = token::tokenize(name).first().map(|t| t.text(name)) else {
            continue;
        };
        for &at in by_first.get(first).map(Vec::as_slice).unwrap_or_default() {
            let end = at + name.len();
            if text.get(at..end) != Some(name) {
                continue;
            }
            let possessive = ["'s", "\u{2019}s"]
                .iter()
                .any(|p| text[end..].starts_with(p) && ends.contains(&(end + p.len())));
            if !ends.contains(&end) && !possessive
                || family_names.contains(name) && !possessive && followed(end)
            {
                continue;
            }
            let glued_han = han(name.chars().next()) && han(text[..at].chars().next_back())
                || han(name.chars().last()) && han(text[end..].chars().next());
            if !glued_han && !overlaps(&blocked, at, end) {
                blocked.insert(at, end);
                out.push(Entity {
                    kind,
                    start: at,
                    end,
                    confidence,
                    review_recommended: false,
                    source: Source::Model,
                    components: Vec::new(),
                    normalized: None,
                    region: None,
                });
            }
        }
    }
    out
}

/// Titles, roles, and office words that a role line taken for a name would end in, beyond
/// `GLUED_ROLES`.
const ROLE_WORDS: [&str; 18] = [
    "General",
    "Minister",
    "Owner",
    "Manager",
    "Executive",
    "Team",
    "Care",
    "Media",
    "Embassy",
    "Office",
    "Service",
    "Services",
    "Council",
    "Ministry",
    "Chief",
    "Ambassador",
    "Consul",
    "Commissioner",
];

/// The family name a person is mentioned by after the first time: the last word of a name of
/// two or more words in a cased script, skipping generational suffixes and initials
/// (`Kobakhidze` from `Irakli Kobakhidze`, `Parham` from `William N. Parham, III`). It must
/// start with a capital, have three or more letters, and not be all capitals, which would more
/// often be an acronym. Scripts without letter case glue case endings or titles to the family
/// name, so their names are repeated only in full.
fn family_name(name: &str) -> Option<&str> {
    let words: Vec<(usize, &str)> = name
        .split_whitespace()
        .map(|w| (w.as_ptr() as usize - name.as_ptr() as usize, w))
        .collect();
    // `Smith, John`: the family name comes first and the last word is the given name.
    if words.len() < 2 || words[0].1.ends_with(',') {
        return None;
    }
    let k = words.iter().rposition(|(_, w)| {
        let bare = w.trim_end_matches([',', '.']);
        !matches!(bare, "Jr" | "Sr" | "II" | "III" | "IV") && bare.chars().count() > 1
    })?;
    let (at, word) = words[k];
    let word = word.trim_end_matches(',');
    let letters = word.chars().filter(|c| c.is_alphabetic()).count();
    let shaped = word.chars().next().is_some_and(char::is_uppercase)
        && word.chars().skip(1).any(char::is_lowercase)
        && word
            .chars()
            .all(|c| c.is_alphabetic() || matches!(c, '-' | '\'' | '\u{2019}'));
    if letters < 3 || !shaped || k == 0 {
        return None;
    }
    // `von der Leyen`, `van Dijk`, `bin Salman`: particles belong to the family name.
    let mut first = k;
    while first > 1 && PARTICLES.contains(&words[first - 1].1) {
        first -= 1;
    }
    Some(&name[words[first].0..at + word.len()])
}

/// Lower-case particles that open a family name.
const PARTICLES: [&str; 12] = [
    "von", "van", "der", "den", "de", "da", "di", "du", "le", "la", "bin", "al",
];

/// Whether an organization's name is distinctive enough to repeat: two or more words, an
/// acronym of two or more capitals, or a name in a script without letter case. A single
/// capitalized word (`Item`, `The`, `Projekte`) is too often a stray prediction.
fn distinctive(name: &str) -> bool {
    let words = name.split_whitespace().count();
    let letters: Vec<char> = name.chars().filter(|c| c.is_alphabetic()).collect();
    let cased: Vec<&char> = letters
        .iter()
        .filter(|c| c.is_uppercase() || c.is_lowercase())
        .collect();
    let latin_like = cased
        .iter()
        .any(|c| c.is_ascii() || ('\u{C0}'..='\u{24F}').contains(*c));
    if letters.len() < 2 {
        return false;
    }
    if !latin_like {
        return true;
    }
    let acronym = letters.iter().filter(|c| c.is_uppercase()).count() >= 2
        && !letters.iter().any(|c| c.is_lowercase())
        && words == 1;
    let proper = words >= 2 && letters.iter().any(|c| c.is_uppercase());
    acronym || proper
}

/// Articles and contractions that open a noun phrase before an organization's name and are not
/// part of it: `the Planning Inspectorate`, `Die DB Fernverkehr AG`, `des Deutschen Bundestages`.
const ARTICLES: [&str; 15] = [
    "the", "der", "die", "das", "dem", "den", "des", "im", "am", "vom", "zum", "zur", "beim",
    "ins", "ans",
];

/// Names whose leading article is part of them: `Die Linke`, `Die Zeit`, `Der Spiegel`.
const ARTICLE_NAMES: [&str; 9] = [
    "linke",
    "grünen",
    "zeit",
    "welt",
    "tageszeitung",
    "partei",
    "spiegel",
    "erste",
    "paritätische",
];

/// Whether `word`, the first token of an organization span, is an article to trim, given the
/// token after it. An all-capital word other than `THE` is an acronym (`AM Best`, `DAS
/// Rechtsschutz`), not an article.
fn leading_article(word: &str, next: &str) -> bool {
    let lower = word.to_lowercase();
    let acronym = word.chars().all(|c| !c.is_lowercase()) && word != "THE";
    ARTICLES.contains(&lower.as_str())
        && !acronym
        && !ARTICLE_NAMES.contains(&next.to_lowercase().as_str())
}

/// Roles that directories glue to a name with a hyphen (`Alice KUHNKE-Member`), which the
/// tokenizer keeps in the name's token.
const GLUED_ROLES: [&str; 23] = [
    "Member",
    "Delegate",
    "Substitute",
    "Alternate",
    "Observer",
    "Chair",
    "Chairman",
    "Chairwoman",
    "Chairperson",
    "Vice-Chair",
    "President",
    "Vice-President",
    "Governor",
    "Director",
    "Head",
    "Deputy",
    "Secretary",
    "Adviser",
    "Advisor",
    "Officer",
    "Coordinator",
    "Assistant",
    "Rapporteur",
];

/// The end of a span that the token grid cannot cut: a person's name stops before a role glued
/// on with a hyphen, and a person or organization stops before a possessive `'s` (`the SEC's
/// Office`). Only the last token can hold either.
fn inside_last_token(text: &str, start: usize, end: usize, kind: Kind) -> usize {
    let mut end = end;
    if kind == Kind::Person {
        let span = &text[start..end];
        let glued = span.match_indices('-').find_map(|(i, _)| {
            let rest = &span[i + 1..];
            GLUED_ROLES
                .iter()
                .any(|role| {
                    rest.strip_prefix(role)
                        .is_some_and(|after| !after.starts_with(char::is_alphabetic))
                })
                .then_some(i)
        });
        if let Some(cut) = glued.filter(|&cut| cut > 0) {
            end = start + cut;
        }
    }
    if matches!(kind, Kind::Person | Kind::Org) {
        for possessive in ["'s", "\u{2019}s"] {
            if let Some(kept) = text[start..end].strip_suffix(possessive)
                && !kept.is_empty()
            {
                end = start + kept.len();
                break;
            }
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    const SIGNATURE: &str = "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example";

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

    fn org(text: &str, name: &str, from: usize) -> Entity {
        let start = from + text[from..].find(name).unwrap();
        Entity {
            kind: Kind::Org,
            start,
            end: start + name.len(),
            confidence: 0.9,
            review_recommended: false,
            source: Source::Model,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    fn repeats(text: &str, found: &[Entity]) -> Vec<String> {
        let tokens = crate::token::tokenize(text);
        let taken: Vec<(usize, usize)> = found.iter().map(|e| (e.start, e.end)).collect();
        repeated_names(text, &tokens, found, &taken)
            .iter()
            .map(|e| text[e.start..e.end].to_string())
            .collect()
    }

    #[test]
    fn an_organization_found_once_is_found_wherever_it_repeats() {
        let text = "Contact HMRC today. HMRC replies within HMRCs days; ask HMRC-Online or hmrc.";
        assert_eq!(repeats(text, &[org(text, "HMRC", 0)]), ["HMRC"]);

        let text = "The Bank said so. The Bank of England agreed. Write to the Bank of England.";
        let found = [org(text, "Bank", 0), org(text, "Bank of England", 20)];
        assert_eq!(
            repeats(text, &found),
            ["Bank of England"],
            "longest names first"
        );

        let text = "Item one. Item two. The end. The report.";
        assert!(repeats(text, &[org(text, "Item", 0), org(text, "The", 0)]).is_empty());

        let text = "Acme Ltd, then Acme Ltd again";
        let mut person = org(text, "Acme Ltd", 10);
        person.kind = Kind::Person;
        assert!(repeats(text, &[org(text, "Acme Ltd", 0), person]).is_empty());

        let mut unsure = org(text, "Acme Ltd", 0);
        unsure.confidence = 0.6;
        assert!(
            repeats(text, &[unsure]).is_empty(),
            "only confident names repeat"
        );

        let text = "総務か。総務省は発表した。総務省令により、総務省が対応する。";
        assert_eq!(repeats(text, &[org(text, "総務省", 0)]), ["総務省"]);

        let text = "შემოსავლების სამსახური და შემოსავლების სამსახური";
        assert_eq!(
            repeats(text, &[org(text, "შემოსავლების სამსახური", 0)]).len(),
            1
        );
    }

    #[test]
    fn a_person_named_in_full_is_found_by_family_name() {
        let person = |text: &str, name: &str, from: usize| {
            let mut e = org(text, name, from);
            e.kind = Kind::Person;
            e
        };
        let text = "Irakli Kobakhidze spoke. Kobakhidze said that Kobakhidze’s plan and \
                    Kobakhidzes and KOBAKHIDZE stand; Irakli Kobakhidze left.";
        assert_eq!(
            repeats(text, &[person(text, "Irakli Kobakhidze", 0)]),
            ["Irakli Kobakhidze", "Kobakhidze", "Kobakhidze"]
        );

        let text = "William N. Parham, III wrote. Parham agreed.";
        assert_eq!(
            repeats(text, &[person(text, "William N. Parham, III", 0)]),
            ["Parham"]
        );

        let text = "Director General Anna Wu. Non-Executive Director. The director and Wu.";
        let found = [
            person(text, "Director General", 0),
            person(text, "Non-Executive Director", 26),
        ];
        assert!(repeats(text, &found).is_empty(), "roles do not spread");

        let text = "Media Team said the media would wait. Team and Media.";
        assert!(repeats(text, &[person(text, "Media Team", 0)]).is_empty());

        let text = "Ursula von der Leyen spoke. Later von der Leyen left.";
        assert_eq!(
            repeats(text, &[person(text, "Ursula von der Leyen", 0)]),
            ["von der Leyen"]
        );

        let text = "John A. Smith spoke; write to smith@x.org. Smith said.";
        let mut email = org(text, "smith@x.org", 0);
        email.kind = Kind::Email;
        let found = [person(text, "John A. Smith", 0), email];
        assert_eq!(repeats(text, &found), ["Smith"]);

        let text = "ირაკლი კობახიძე თქვა. ირაკლი კობახიძე წავიდა.";
        assert_eq!(
            repeats(text, &[person(text, "ირაკლი კობახიძე", 0)]),
            ["ირაკლი კობახიძე"]
        );

        let text = "Theresa May spoke on 3 May 2024. May said so. Paul May Smith came.";
        assert_eq!(repeats(text, &[person(text, "Theresa May", 0)]), ["May"]);

        let text = "Smith, John spoke. John left.";
        assert!(repeats(text, &[person(text, "Smith, John", 0)]).is_empty());

        let text = "Jo Li met Li.";
        assert!(repeats(text, &[person(text, "Jo Li", 0)]).is_empty());

        let text = "Ms Smith of Smith & Co. Smith said.";
        let found = [person(text, "Ms Smith", 0), org(text, "Smith & Co", 12)];
        assert_eq!(repeats(text, &found), ["Smith"], "clear of other spans");

        let text = "The NRC's staff; the NRC said.";
        assert_eq!(repeats(text, &[org(text, "NRC", 16)]), ["NRC"]);
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

    /// `span_entity` over every content token of `text`, as the text it keeps.
    fn trimmed(kind: Kind, text: &str) -> Option<&str> {
        let tokens = token::tokenize(text);
        let retained: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| features::is_content(t))
            .map(|(i, _)| i)
            .collect();
        span_entity(text, &tokens, &retained, kind, (0, retained.len() - 1), 0.9)
            .map(|e| &text[e.start..e.end])
    }

    #[test]
    fn span_edges_are_trimmed() {
        let cases = [
            (
                Kind::Person,
                "\u{201c}Anna Schmidt\u{201d},",
                Some("Anna Schmidt"),
            ),
            (
                Kind::Org,
                "Nippon Freight Co., Ltd.",
                Some("Nippon Freight Co., Ltd."),
            ),
            (Kind::Org, "Acme Holdings.", Some("Acme Holdings")),
            (Kind::Person, "Anna Lee.", Some("Anna Lee")),
            (Kind::Address, ", Tbilisi 0108", Some("Tbilisi 0108")),
            (Kind::Org, "株式会社ハルカ。", Some("株式会社ハルカ")),
            (Kind::Org, "、株式会社ハルカ", Some("株式会社ハルカ")),
            (Kind::Org, "( )", None),
            (Kind::Org, "[ ]", None),
            (Kind::Org, "()", None),
            (Kind::Org, "\"( )\",", None),
            (
                Kind::Org,
                "\u{ff08}株式会社ハルカ\u{ff09}",
                Some("株式会社ハルカ"),
            ),
            (Kind::Org, "株式会社ハルカ\u{ff09}", Some("株式会社ハルカ")),
            (Kind::Org, "(Acme Ltd", Some("Acme Ltd")),
            (Kind::Org, "Acme (UK)", Some("Acme (UK)")),
            (Kind::Org, "(UK) Acme (Europe)", Some("(UK) Acme (Europe)")),
            (
                Kind::Address,
                "14 Rustaveli Avenue, Tbilisi 0108.",
                Some("14 Rustaveli Avenue, Tbilisi 0108"),
            ),
            (
                Kind::Address,
                "221 Canal Street, Leeds,",
                Some("221 Canal Street, Leeds"),
            ),
            (Kind::Address, "1 Main St.", Some("1 Main St.")),
            (Kind::Person, "\"", None),
            (Kind::Org, "(", None),
            (Kind::Person, "Oliver Grant <", Some("Oliver Grant")),
            (Kind::Person, "<Oliver Grant>", Some("Oliver Grant")),
            (Kind::Org, "Acme Ltd (", Some("Acme Ltd")),
            (Kind::Org, ") Acme Ltd", Some("Acme Ltd")),
            (Kind::Person, "Alice KUHNKE-Member", Some("Alice KUHNKE")),
            (Kind::Person, "Anna BERG-Vice-Chair", Some("Anna BERG")),
            (
                Kind::Person,
                "Jean-Pierre Dubois",
                Some("Jean-Pierre Dubois"),
            ),
            (Kind::Person, "Ana Headley-Smith", Some("Ana Headley-Smith")),
            (Kind::Org, "the SEC's", Some("SEC")),
            (
                Kind::Org,
                "Die DB Fernverkehr AG",
                Some("DB Fernverkehr AG"),
            ),
            (
                Kind::Org,
                "des Deutschen Bundestages",
                Some("Deutschen Bundestages"),
            ),
            (Kind::Org, "Die Linke", Some("Die Linke")),
            (Kind::Org, "the", None),
            (Kind::Org, "AM Best Europe", Some("AM Best Europe")),
            (Kind::Org, "DAS Rechtsschutz", Some("DAS Rechtsschutz")),
            (Kind::Org, "DIE LINKE", Some("DIE LINKE")),
            (Kind::Org, "Der Spiegel", Some("Der Spiegel")),
            (Kind::Org, "the THE Group", Some("THE Group")),
            (Kind::Person, "Die Anna Berg", Some("Die Anna Berg")),
            (Kind::Person, "Gunta ANČA-Delegate", Some("Gunta ANČA")),
            (Kind::Org, "DVLA\u{2019}s", Some("DVLA")),
            (Kind::Address, "Kings's Road", Some("Kings's Road")),
        ];
        for (kind, text, want) in cases {
            assert_eq!(trimmed(kind, text), want, "{text:?}");
        }
    }
}
