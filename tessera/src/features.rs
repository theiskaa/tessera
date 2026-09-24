//! Per-token features shared by training and inference: hashed character
//! n-grams, script and shape codes, and deterministic signal flags backed by
//! small dictionaries. Nothing here allocates a vocabulary; names and streets
//! are hashed, not stored.
//!
//! Dictionary sources: road, unit, and region terms from libpostal's
//! dictionaries for en, de, nl, ka, ja (MIT); legal forms from GLEIF's entity
//! legal form list; honorifics, salutations, and closings hand-written.

use crate::chunk::Mask;
use crate::token::{INVISIBLE, Token, TokenClass};

/// How token text is hashed into n-gram ids. A bundle records the values it was trained
/// with, and inference must use the same ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureConfig {
    /// Character n-gram lengths taken over `^text$`.
    pub ngram_sizes: Vec<u8>,
    /// Number of embedding rows the n-gram hashes are reduced into.
    pub hash_buckets: u32,
    /// Seed mixed into the hash, so a bundle can change its buckets without new code.
    pub hash_seed: u64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig {
            ngram_sizes: vec![2, 3, 4],
            hash_buckets: 65_536,
            hash_seed: 0,
        }
    }
}

/// Features of one token: the model's inputs and the baselines' signals.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenFeatures {
    /// Hashed ids of the token's character n-grams; empty for whitespace.
    pub ngram_ids: Vec<u32>,
    /// The token's `Script`, as its discriminant.
    pub script: u8,
    /// 0 lower, 1 upper, 2 title, 3 digit, 4 alnum, 5 punct, 6 space, 7 newline, 8 other, 9 mixed case.
    pub shape: u8,
    /// Bits from [`flag`].
    pub flags: u32,
}

/// Bit positions of [`TokenFeatures::flags`]. Fixed for the life of a bundle version.
pub mod flag {
    /// The token contains an ASCII digit.
    pub const HAS_DIGIT: u32 = 1 << 0;
    /// The token is a digit run.
    pub const ALL_DIGIT: u32 = 1 << 1;
    /// An alphabetic token with no lowercase letters.
    pub const ALL_UPPER: u32 = 1 << 2;
    /// An alphabetic token with only its first letter uppercase.
    pub const TITLE_CASE: u32 = 1 << 3;
    /// A punctuation token.
    pub const HAS_PUNCT: u32 = 1 << 4;
    /// The first non-space token of its line.
    pub const LINE_START: u32 = 1 << 5;
    /// The last non-space token of its line.
    pub const LINE_END: u32 = 1 << 6;
    /// On a line of at most eight tokens next to another such line, the shape of signatures.
    pub const IN_SHORT_LINE_BLOCK: u32 = 1 << 7;
    /// Inside an email or phone found by the rules layer.
    pub const IN_RULE_SPAN: u32 = 1 << 8;
    /// Part of a postcode shape for the country, or any known country.
    pub const POSTCODE_LIKE: u32 = 1 << 9;
    /// One to five digits, optionally followed by one letter.
    pub const HOUSE_NUMBER_LIKE: u32 = 1 << 10;
    /// Part of a road term such as `street`, `straße`, or `丁目`.
    pub const ROAD_TERM: u32 = 1 << 11;
    /// Part of a unit term such as `flat`, `suite`, or `号室`.
    pub const UNIT_TERM: u32 = 1 << 12;
    /// Part of a region term such as `county` or `県`.
    pub const REGION_TERM: u32 = 1 << 13;
    /// Part of a country name or code.
    pub const COUNTRY_TERM: u32 = 1 << 14;
    /// Part of a company legal form such as `Ltd`, `GmbH`, or `株式会社`.
    pub const LEGAL_FORM: u32 = 1 << 15;
    /// Part of an honorific such as `Dr` or `様`.
    pub const HONORIFIC: u32 = 1 << 16;
    /// Part of a salutation such as `Dear` or `Sehr geehrte`.
    pub const SALUTATION: u32 = 1 << 17;
    /// Part of a closing such as `Kind regards` or `敬具`.
    pub const CLOSING: u32 = 1 << 18;
    /// The first token after a blank line.
    pub const AFTER_NEWLINE_BLANK: u32 = 1 << 19;
    /// A whitespace token.
    pub const IS_SPACE: u32 = 1 << 20;
    /// A line break token.
    pub const IS_NEWLINE: u32 = 1 << 21;
    /// A token outside the mask of structured input: Markdown syntax, code, a link destination.
    pub const MASKED: u32 = 1 << 22;
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit hash of `bytes`, with `seed` mixed into the offset basis.
pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = FNV_OFFSET ^ seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// The network reads at most this many n-gram ids per token; generation stops here. Every
/// n-gram of sizes 2 to 4 fits for tokens of up to 21 characters.
pub const MAX_NGRAMS_PER_TOKEN: usize = 64;

/// Hashed ids of the character n-grams of `^text$`, for each size in `config.ngram_sizes`
/// in order, at most [`MAX_NGRAMS_PER_TOKEN`] of them. Invisible characters are skipped, so
/// `Te\u{AD}le\u{AD}kom` reads as `Telekom`.
pub(crate) fn ngram_ids(text: &str, config: &FeatureConfig) -> Vec<u32> {
    let wrapped: Vec<char> = core::iter::once('^')
        .chain(text.chars().filter(|c| !INVISIBLE.contains(c)))
        .chain(core::iter::once('$'))
        .collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    for &n in &config.ngram_sizes {
        let n = n as usize;
        if wrapped.len() < n {
            continue;
        }
        for w in wrapped.windows(n) {
            if out.len() == MAX_NGRAMS_PER_TOKEN {
                return out;
            }
            buf.clear();
            buf.extend(w);
            out.push(
                (fnv1a(buf.as_bytes(), config.hash_seed) % u64::from(config.hash_buckets)) as u32,
            );
        }
    }
    out
}

/// Road and street terms, lowercase, matched against lowercased token runs.
pub(crate) const ROAD_TERMS: &[&str] = &[
    // en
    "street",
    "st",
    "road",
    "rd",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "way",
    "close",
    "crescent",
    "place",
    "pl",
    "square",
    "sq",
    "boulevard",
    "blvd",
    "court",
    "ct",
    "terrace",
    "highway",
    "hwy",
    "row",
    "gardens",
    "grove",
    "hill",
    "park",
    "walk",
    "mews",
    "parade",
    "circle",
    "cir",
    "trail",
    "parkway",
    "pkwy",
    // de
    "straße",
    "strasse",
    "str",
    "weg",
    "platz",
    "allee",
    "gasse",
    "ring",
    "damm",
    "ufer",
    "chaussee",
    "steig",
    // nl
    "straat",
    "laan",
    "plein",
    "gracht",
    "kade",
    "dijk",
    "singel",
    "steeg",
    "dreef",
    // ka
    "ქუჩა",
    "ქ",
    "გამზირი",
    "გამზ",
    "ჩიხი",
    "მოედანი",
    "შესახვევი",
    "გზატკეცილი",
    // ja
    "通り",
    "通",
    "丁目",
    "番地",
    "番",
    "号",
    "町",
    "筋",
];

/// Unit, floor, and room terms, lowercase, matched against lowercased token runs.
pub(crate) const UNIT_TERMS: &[&str] = &[
    "flat",
    "apt",
    "apartment",
    "suite",
    "ste",
    "unit",
    "floor",
    "fl",
    "room",
    "rm",
    "level",
    "lvl",
    "office",
    "wohnung",
    "whg",
    "stock",
    "og",
    "eg",
    "etage",
    "zimmer",
    "verdieping",
    "appartement",
    "bus",
    "ბინა",
    "სართული",
    "კორპუსი",
    "კორპ",
    "号室",
    "階",
    "室",
    "棟",
];

/// Administrative region terms, lowercase, matched against lowercased token runs.
pub(crate) const REGION_TERMS: &[&str] = &[
    "county",
    "state",
    "province",
    "region",
    "prefecture",
    "territory",
    "district",
    "borough",
    "bundesland",
    "kreis",
    "landkreis",
    "kanton",
    "bezirk",
    "provincie",
    "gemeente",
    "მხარე",
    "რაიონი",
    "მუნიციპალიტეტი",
    "県",
    "府",
    "道",
    "都",
    "市",
    "区",
    "郡",
    "村",
];

/// Country names and codes, lowercase, matched against lowercased token runs.
pub(crate) const COUNTRY_TERMS: &[&str] = &[
    "united states",
    "usa",
    "us",
    "u.s.",
    "u.s.a.",
    "america",
    "canada",
    "united kingdom",
    "uk",
    "u.k.",
    "great britain",
    "britain",
    "england",
    "scotland",
    "wales",
    "northern ireland",
    "germany",
    "deutschland",
    "de",
    "austria",
    "österreich",
    "switzerland",
    "schweiz",
    "netherlands",
    "nederland",
    "the netherlands",
    "holland",
    "nl",
    "belgium",
    "belgië",
    "belgique",
    "georgia",
    "საქართველო",
    "sakartvelo",
    "ge",
    "japan",
    "日本",
    "日本国",
    "nippon",
    "nihon",
    "jp",
    "ireland",
    "éire",
    "france",
    "italy",
    "spain",
    "poland",
    "turkey",
    "türkiye",
    "armenia",
    "azerbaijan",
    "russia",
];

/// Company legal forms, lowercase, matched against lowercased token runs.
pub(crate) const LEGAL_FORMS: &[&str] = &[
    "ltd",
    "ltd.",
    "limited",
    "llc",
    "l.l.c.",
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corp.",
    "corporation",
    "plc",
    "llp",
    "lp",
    "co",
    "co.",
    "company",
    "group",
    "holdings",
    "partners",
    "trust",
    "foundation",
    "gmbh",
    "mbh",
    "ag",
    "kg",
    "ohg",
    "ug",
    "e.v.",
    "ev",
    "gbr",
    "se",
    "kgaa",
    "eg",
    "bv",
    "b.v.",
    "nv",
    "n.v.",
    "vof",
    "v.o.f.",
    "cv",
    "stichting",
    "შპს",
    "სს",
    "ააიპ",
    "სპს",
    "კს",
    "იმ",
    "株式会社",
    "有限会社",
    "合同会社",
    "合資会社",
    "合名会社",
    "一般社団法人",
    "(株)",
    "(有)",
    "㈱",
    "㈲",
];

/// Honorifics and titles, lowercase, matched against lowercased token runs.
pub(crate) const HONORIFICS: &[&str] = &[
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "miss",
    "mx",
    "dr",
    "dr.",
    "prof",
    "prof.",
    "sir",
    "dame",
    "rev",
    "rev.",
    "herr",
    "frau",
    "hr.",
    "fr.",
    "dipl.-ing.",
    "dhr",
    "dhr.",
    "mevr",
    "mevr.",
    "mw",
    "mw.",
    "de heer",
    "mevrouw",
    "ბატონი",
    "ქალბატონი",
    "ბატონო",
    "ქალბატონო",
    "ბ-ნი",
    "ქ-ნი",
    "様",
    "さん",
    "氏",
    "殿",
    "先生",
];

/// Letter and email salutations, lowercase, matched against lowercased token runs.
pub(crate) const SALUTATIONS: &[&str] = &[
    "dear",
    "hi",
    "hello",
    "hey",
    "greetings",
    "hallo",
    "sehr geehrte",
    "sehr geehrter",
    "liebe",
    "lieber",
    "guten tag",
    "beste",
    "geachte",
    "hoi",
    "goedendag",
    "გამარჯობა",
    "მოგესალმებით",
    "ძვირფასო",
    "პატივცემულო",
    "拝啓",
    "前略",
    "お世話になっております",
];

/// Letter and email closings, lowercase, matched against lowercased token runs.
pub(crate) const CLOSINGS: &[&str] = &[
    "regards",
    "kind regards",
    "best regards",
    "warm regards",
    "best",
    "sincerely",
    "yours sincerely",
    "yours faithfully",
    "cheers",
    "thanks",
    "thank you",
    "many thanks",
    "best wishes",
    "cordially",
    "mit freundlichen grüßen",
    "freundliche grüße",
    "viele grüße",
    "beste grüße",
    "mfg",
    "liebe grüße",
    "herzliche grüße",
    "met vriendelijke groet",
    "met vriendelijke groeten",
    "groeten",
    "hartelijke groeten",
    "mvg",
    "პატივისცემით",
    "საუკეთესო სურვილებით",
    "მადლობა",
    "敬具",
    "よろしくお願いします",
    "よろしくお願いいたします",
    "以上",
];

/// Shape string: `A` for a letter, `9` for a digit, other chars kept.
pub(crate) fn shape_string(text: &str) -> String {
    text.chars()
        .filter(|c| !INVISIBLE.contains(c))
        .map(|c| {
            if c.is_ascii_digit() {
                '9'
            } else if c.is_alphabetic() {
                'A'
            } else {
                c
            }
        })
        .collect()
}

/// Number of tokens a postcode occupies starting at `i`, for `country` (ISO alpha-2, uppercase)
/// or for any known country when `None`.
pub(crate) fn postcode_at(
    text: &str,
    tokens: &[Token],
    i: usize,
    country: Option<&str>,
) -> Option<usize> {
    let tok = |k: usize| tokens.get(i + k).map(|t| (t, t.text(text)));
    let (_, s0) = tok(0)?;
    let shape0 = shape_string(s0);
    let two_tokens = |second: &dyn Fn(&str) -> bool| -> Option<usize> {
        let (t1, _) = tok(1)?;
        let (_, s2) = tok(2)?;
        (t1.class == TokenClass::Space && second(&shape_string(s2))).then_some(3)
    };
    let gb = || {
        matches!(
            shape0.as_str(),
            "A9" | "A99" | "A9A" | "AA9" | "AA99" | "AA9A"
        )
        .then(|| two_tokens(&|s| s == "9AA"))
        .flatten()
    };
    let us = || match shape0.as_str() {
        "99999" => Some(
            if matches!((tok(1), tok(2)), (Some((t1, "-")), Some((_, s2))) if t1.class == TokenClass::Punct && shape_string(s2) == "9999")
            {
                3
            } else {
                1
            },
        ),
        _ => None,
    };
    let ca = || {
        (shape0 == "A9A")
            .then(|| two_tokens(&|s| s == "9A9"))
            .flatten()
    };
    let de = || (shape0 == "99999").then_some(1);
    let nl = || {
        (shape0 == "9999")
            .then(|| two_tokens(&|s| s == "AA"))
            .flatten()
    };
    let four_digit = || (shape0 == "9999").then_some(1);
    let jp = || {
        (shape0 == "999")
            .then(|| match (tok(1), tok(2)) {
                (Some((t1, "-")), Some((_, s2)))
                    if t1.class == TokenClass::Punct && shape_string(s2) == "9999" =>
                {
                    Some(3)
                }
                _ => None,
            })
            .flatten()
    };
    match country {
        Some("GB") => gb(),
        Some("US") => us(),
        Some("CA") => ca(),
        Some("DE") => de(),
        Some("AT") | Some("CH") | Some("GE") => four_digit(),
        Some("NL") => nl(),
        Some("JP") => jp(),
        Some(_) => None,
        None => gb()
            .or_else(us)
            .or_else(ca)
            .or_else(nl)
            .or_else(jp)
            .or_else(de)
            .or_else(four_digit),
    }
}

/// `14`, `221B`, `12a`, `1-1`, `No. 5` style tokens: 1 to 5 digits optionally followed by one letter.
pub(crate) fn house_number_like(text: &str) -> bool {
    let s = shape_string(text);
    let digits = s.chars().take_while(|c| *c == '9').count();
    (1..=5).contains(&digits) && (s.len() == digits || (s.len() == digits + 1 && s.ends_with('A')))
}

/// Whether the lowercased `lower` is an entry of `dict`.
pub(crate) fn is_term(dict: &[&str], lower: &str) -> bool {
    dict.contains(&lower)
}

/// Shape code: 0 lower, 1 upper, 2 title, 3 digit, 4 alnum, 5 punct, 6 space, 7 newline,
/// 8 other, 9 mixed case.
fn shape_code(t: &Token, s: &str) -> u8 {
    match t.class {
        TokenClass::Digit => 3,
        TokenClass::Alnum => 4,
        TokenClass::Punct => 5,
        TokenClass::Space => 6,
        TokenClass::Newline => 7,
        TokenClass::Other => 8,
        TokenClass::Alpha => {
            let upper = s.chars().any(char::is_uppercase);
            let lower = s.chars().any(char::is_lowercase);
            let first_upper = s.chars().next().is_some_and(char::is_uppercase);
            match (upper, lower) {
                (false, true) => 0,
                (true, false) => 1,
                (true, true) if first_upper && !s.chars().skip(1).any(char::is_uppercase) => 2,
                _ => 9,
            }
        }
    }
}

/// Token index ranges of each line, excluding the newline tokens between them.
pub fn line_ranges(tokens: &[Token]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, t) in tokens.iter().enumerate() {
        if t.class == TokenClass::Newline {
            out.push((start, i));
            start = i + 1;
        }
    }
    out.push((start, tokens.len()));
    out
}

/// Whether a token is neither whitespace nor a line break.
pub fn is_content(t: &Token) -> bool {
    !matches!(t.class, TokenClass::Space | TokenClass::Newline)
}

/// Per token: its line index. Per line: the first and last content token, if any.
/// Computed once so that line features stay linear in document length.
fn line_layout(
    tokens: &[Token],
    lines: &[(usize, usize)],
) -> (Vec<usize>, Vec<Option<(usize, usize)>>) {
    let mut line_of = vec![0usize; tokens.len()];
    let mut edges = Vec::with_capacity(lines.len());
    for (line, &(a, b)) in lines.iter().enumerate() {
        // The newline token that ends a line, at index `b`, belongs to that line.
        for slot in line_of.iter_mut().take(b + 1).skip(a) {
            *slot = line;
        }
        let first = (a..b).find(|&k| is_content(&tokens[k]));
        let last = (a..b).rfind(|&k| is_content(&tokens[k]));
        edges.push(first.zip(last));
    }
    (line_of, edges)
}

/// Dictionary terms span several tokens once the tokenizer splits punctuation and Han
/// ideographs (`ltd.`, `株式会社`, `mit freundlichen grüßen`); this is the longest run matched.
const MAX_TERM_TOKENS: usize = 6;

/// Flag bits for every token covered by a dictionary term, matching runs of up to
/// `MAX_TERM_TOKENS` consecutive tokens with any whitespace inside a line read as one space.
fn term_flags(text: &str, tokens: &[Token]) -> Vec<u32> {
    const DICTS: [(&[&str], u32); 8] = [
        (ROAD_TERMS, flag::ROAD_TERM),
        (UNIT_TERMS, flag::UNIT_TERM),
        (REGION_TERMS, flag::REGION_TERM),
        (COUNTRY_TERMS, flag::COUNTRY_TERM),
        (LEGAL_FORMS, flag::LEGAL_FORM),
        (HONORIFICS, flag::HONORIFIC),
        (SALUTATIONS, flag::SALUTATION),
        (CLOSINGS, flag::CLOSING),
    ];
    let mut out = vec![0u32; tokens.len()];
    let mut phrase = String::new();
    for i in 0..tokens.len() {
        if !is_content(&tokens[i]) {
            continue;
        }
        phrase.clear();
        let mut used = 0;
        for (k, t) in tokens.iter().enumerate().skip(i) {
            match t.class {
                TokenClass::Newline => break,
                TokenClass::Space => {
                    phrase.push(' ');
                    continue;
                }
                _ => phrase.extend(
                    t.text(text)
                        .chars()
                        .filter(|c| !INVISIBLE.contains(c))
                        .flat_map(char::to_lowercase),
                ),
            }
            used += 1;
            let bits = DICTS
                .iter()
                .filter(|(dict, _)| is_term(dict, &phrase))
                .fold(0, |acc, (_, bit)| acc | bit);
            if bits != 0 {
                for (slot, tok) in out.iter_mut().zip(tokens).take(k + 1).skip(i) {
                    if is_content(tok) {
                        *slot |= bits;
                    }
                }
            }
            if used == MAX_TERM_TOKENS {
                break;
            }
        }
    }
    out
}

/// Tokens of lines with one to eight content tokens that neighbour another such line:
/// the shape of signature blocks and letterheads.
fn short_line_block_mask(tokens: &[Token], lines: &[(usize, usize)]) -> Vec<bool> {
    const MAX_SHORT_LINE_TOKENS: usize = 8;
    let short: Vec<bool> = lines
        .iter()
        .map(|&(a, b)| {
            let n = (a..b).filter(|&k| is_content(&tokens[k])).count();
            (1..=MAX_SHORT_LINE_TOKENS).contains(&n)
        })
        .collect();
    let mut out = vec![false; tokens.len()];
    for (line, &(a, b)) in lines.iter().enumerate() {
        let neighbour =
            (line > 0 && short[line - 1]) || short.get(line + 1).copied().unwrap_or(false);
        if short[line] && neighbour {
            for flag in out.iter_mut().take(b).skip(a) {
                *flag = true;
            }
        }
    }
    out
}

/// One `TokenFeatures` per token. `rule_spans` are byte ranges of rule-layer
/// entities; `country` narrows postcode shapes when known.
pub fn featurize(
    text: &str,
    tokens: &[Token],
    rule_spans: &[(usize, usize)],
    country: Option<&str>,
    config: &FeatureConfig,
    mask: Option<&Mask>,
) -> Vec<TokenFeatures> {
    let n = tokens.len();
    let mut out: Vec<TokenFeatures> = Vec::with_capacity(n);
    let lines = line_ranges(tokens);
    let short_block = short_line_block_mask(tokens, &lines);
    let (line_of, edges) = line_layout(tokens, &lines);
    let terms = term_flags(text, tokens);
    let mut spans = rule_spans.to_vec();
    spans.sort_unstable();
    // Tokens and spans are both in order, so one pointer finds each token's overlapping span.
    let mut span_at = 0;
    let mut postcode_until = 0usize;

    for (i, t) in tokens.iter().enumerate() {
        let s = t.text(text);
        let mut f = TokenFeatures {
            script: t.script as u8,
            shape: shape_code(t, s),
            ..Default::default()
        };
        if !matches!(t.class, TokenClass::Space | TokenClass::Newline) {
            f.ngram_ids = ngram_ids(s, config);
        }
        let mut flags = 0u32;
        if mask.is_some_and(|m| !m.covers(t)) {
            flags |= flag::MASKED;
        }
        if s.chars().any(|c| c.is_ascii_digit()) {
            flags |= flag::HAS_DIGIT;
        }
        if t.class == TokenClass::Digit {
            flags |= flag::ALL_DIGIT;
        }
        if t.class == TokenClass::Alpha
            && s.chars().all(|c| !c.is_lowercase())
            && s.chars().any(char::is_uppercase)
        {
            flags |= flag::ALL_UPPER;
        }
        if f.shape == 2 {
            flags |= flag::TITLE_CASE;
        }
        if t.class == TokenClass::Punct {
            flags |= flag::HAS_PUNCT;
        }
        if t.class == TokenClass::Space {
            flags |= flag::IS_SPACE;
        }
        if t.class == TokenClass::Newline {
            flags |= flag::IS_NEWLINE;
        }
        while spans.get(span_at).is_some_and(|&(_, b)| b <= t.start) {
            span_at += 1;
        }
        if spans
            .get(span_at)
            .is_some_and(|&(a, b)| a < t.end && t.start < b)
        {
            flags |= flag::IN_RULE_SPAN;
        }
        if short_block[i] {
            flags |= flag::IN_SHORT_LINE_BLOCK;
        }

        let line = line_of[i];
        let line_start = edges[line].is_some_and(|(first, _)| first == i);
        if line_start {
            flags |= flag::LINE_START;
        }
        if edges[line].is_some_and(|(_, last)| last == i) {
            flags |= flag::LINE_END;
        }
        // A blank line may hold spaces, and the next line may be indented.
        if line_start && line > 0 && edges[line - 1].is_none() {
            flags |= flag::AFTER_NEWLINE_BLANK;
        }

        if i >= postcode_until
            && let Some(len) = postcode_at(text, tokens, i, country)
        {
            postcode_until = i + len;
        }
        if i < postcode_until {
            flags |= flag::POSTCODE_LIKE;
        }
        if matches!(t.class, TokenClass::Digit | TokenClass::Alnum) && house_number_like(s) {
            flags |= flag::HOUSE_NUMBER_LIKE;
        }

        flags |= terms[i];
        f.flags = flags;
        out.push(f);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::tokenize;

    fn feats(text: &str, country: Option<&str>) -> Vec<(String, u32)> {
        let toks = tokenize(text);
        featurize(text, &toks, &[], country, &FeatureConfig::default(), None)
            .into_iter()
            .zip(&toks)
            .map(|(f, t)| (t.text(text).to_string(), f.flags))
            .collect()
    }

    fn has(v: &[(String, u32)], text: &str, bit: u32) -> bool {
        v.iter().any(|(s, f)| s == text && f & bit != 0)
    }

    #[test]
    fn invisible_characters_do_not_change_ngrams() {
        let config = FeatureConfig::default();
        assert_eq!(
            ngram_ids("Te\u{AD}le\u{AD}kom", &config),
            ngram_ids("Telekom", &config)
        );
    }

    #[test]
    fn invisible_characters_do_not_hide_terms_or_shapes() {
        let plain = feats("Berliner Straße 5, Deutschland", Some("DE"));
        let soft = feats("Berliner Stra\u{AD}ße 5, Deutsch\u{AD}land", Some("DE"));
        let flags = |f: &[(String, u32)]| f.iter().map(|(_, b)| *b).collect::<Vec<_>>();
        assert_eq!(flags(&soft), flags(&plain));
        assert_eq!(shape_string("10\u{AD}115"), "99999");
    }

    #[test]
    fn ngrams_are_deterministic_and_bounded() {
        let cfg = FeatureConfig::default();
        let a = ngram_ids("Baker", &cfg);
        assert_eq!(a, ngram_ids("Baker", &cfg));
        assert_eq!(a.len(), 6 + 5 + 4); // ^Baker$ has 7 chars: 6 bigrams, 5 trigrams, 4 four-grams
        assert!(a.iter().all(|&id| id < cfg.hash_buckets));
        assert_ne!(a, ngram_ids("baker", &cfg));
    }

    #[test]
    fn postcodes() {
        let v = feats("221B Baker Street, London NW1 6XE", Some("GB"));
        assert!(has(&v, "NW1", flag::POSTCODE_LIKE));
        assert!(has(&v, "6XE", flag::POSTCODE_LIKE));
        assert!(!has(&v, "221B", flag::POSTCODE_LIKE));
        let v = feats("Springfield, IL 62701-1234", Some("US"));
        assert!(has(&v, "62701", flag::POSTCODE_LIKE));
        assert!(has(&v, "1234", flag::POSTCODE_LIKE));
        let v = feats("〒100-0001 東京都", Some("JP"));
        assert!(has(&v, "100", flag::POSTCODE_LIKE));
        assert!(has(&v, "0001", flag::POSTCODE_LIKE));
        let v = feats("10117 Berlin", None);
        assert!(has(&v, "10117", flag::POSTCODE_LIKE));
    }

    #[test]
    fn terms_and_shapes() {
        let v = feats(
            "Dear Dr Smith,\nAcme Ltd\n14 Rustaveli Avenue\nKind regards",
            None,
        );
        assert!(has(&v, "Dear", flag::SALUTATION));
        assert!(has(&v, "Dr", flag::HONORIFIC));
        assert!(has(&v, "Ltd", flag::LEGAL_FORM));
        assert!(has(&v, "14", flag::HOUSE_NUMBER_LIKE));
        assert!(has(&v, "Avenue", flag::ROAD_TERM));
        assert!(has(&v, "Kind", flag::CLOSING));
        assert!(has(&v, "Smith", flag::TITLE_CASE));
        assert!(has(&v, "Dear", flag::LINE_START));
        assert!(has(&v, "Ltd", flag::LINE_END));
        assert!(has(&v, "Acme", flag::IN_SHORT_LINE_BLOCK));
    }

    #[test]
    fn four_digit_postcodes_in_austria_and_switzerland() {
        assert!(has(
            &feats("Wien 1010, Österreich", Some("AT")),
            "1010",
            flag::POSTCODE_LIKE
        ));
        assert!(has(
            &feats("8001 Zürich", Some("CH")),
            "8001",
            flag::POSTCODE_LIKE
        ));
        assert!(!has(
            &feats("8001 Zürich", Some("DE")),
            "8001",
            flag::POSTCODE_LIKE
        ));
    }

    #[test]
    fn terms_split_by_the_tokenizer_still_match() {
        let v = feats("株式会社トヨタ ㈱", None);
        assert!(has(&v, "株", flag::LEGAL_FORM));
        assert!(has(&v, "社", flag::LEGAL_FORM));
        assert!(has(&feats("Acme Ltd.", None), "Ltd", flag::LEGAL_FORM));
        assert!(has(&feats("Acme e.V.", None), "e", flag::LEGAL_FORM));
        assert!(has(
            &feats("東京都千代田区1丁目", None),
            "丁",
            flag::ROAD_TERM
        ));
        let v = feats("Mit freundlichen Grüßen\nNino", None);
        assert!(has(&v, "Mit", flag::CLOSING));
        assert!(has(&v, "Grüßen", flag::CLOSING));
        assert!(!has(&v, "Nino", flag::CLOSING));
        assert!(has(&feats("日本", None), "日", flag::COUNTRY_TERM));
    }

    #[test]
    fn blank_lines_with_spaces_and_indented_lines() {
        for text in [
            "Thanks\n  \nNino Beridze",
            "Thanks\n\n  Nino Beridze",
            "Thanks\r\n\r\nNino",
        ] {
            let toks = tokenize(text);
            let f = featurize(text, &toks, &[], None, &FeatureConfig::default(), None);
            let nino = toks.iter().position(|t| t.text(text) == "Nino").unwrap();
            assert!(f[nino].flags & flag::AFTER_NEWLINE_BLANK != 0, "{text:?}");
            assert!(f[nino].flags & flag::LINE_START != 0, "{text:?}");
        }
        let text = "Thanks\nNino";
        let toks = tokenize(text);
        let f = featurize(text, &toks, &[], None, &FeatureConfig::default(), None);
        assert!(f[2].flags & flag::AFTER_NEWLINE_BLANK == 0);
    }

    /// A single very long line used to cost time quadratic in its length.
    #[test]
    fn many_rule_spans_flag_their_tokens() {
        let text = "a@b.example ".repeat(20_000);
        let toks = tokenize(&text);
        let spans: Vec<(usize, usize)> = (0..20_000).map(|k| (k * 12, k * 12 + 11)).collect();
        let started = std::time::Instant::now();
        let f = featurize(&text, &toks, &spans, None, &FeatureConfig::default(), None);
        assert!(
            started.elapsed().as_secs_f64() < 2.0,
            "{:?}",
            started.elapsed()
        );
        for (t, feat) in toks.iter().zip(&f) {
            let inside = t.class != TokenClass::Space;
            assert_eq!(
                feat.flags & flag::IN_RULE_SPAN != 0,
                inside,
                "{:?}",
                t.text(&text)
            );
        }
    }

    #[test]
    fn long_single_line_is_linear() {
        let text = "word ".repeat(40_000);
        let toks = tokenize(&text);
        let f = featurize(&text, &toks, &[], None, &FeatureConfig::default(), None);
        assert_eq!(f.len(), toks.len());
        assert!(f[0].flags & flag::LINE_START != 0);
    }

    #[test]
    fn rule_spans_and_blank_lines() {
        let text = "Hi\n\nnino@x.example";
        let toks = tokenize(text);
        let f = featurize(
            text,
            &toks,
            &[(4, 18)],
            None,
            &FeatureConfig::default(),
            None,
        );
        let nino = toks.iter().position(|t| t.text(text) == "nino").unwrap();
        assert!(f[nino].flags & flag::IN_RULE_SPAN != 0);
        assert!(f[nino].flags & flag::AFTER_NEWLINE_BLANK != 0);
        assert!(f[0].flags & flag::AFTER_NEWLINE_BLANK == 0);
    }

    #[test]
    fn tokens_outside_the_mask_carry_the_masked_flag() {
        let text = "a b c";
        let toks = tokenize(text);
        let mask = Mask::new(&[(0, 1)], &[]);
        let f = featurize(
            text,
            &toks,
            &[],
            None,
            &FeatureConfig::default(),
            Some(&mask),
        );
        let masked: Vec<bool> = f.iter().map(|f| f.flags & flag::MASKED != 0).collect();
        assert_eq!(masked, vec![false, true, true, true, true]);
    }
}
