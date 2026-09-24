//! Address corpus preparation: streaming the tagged libpostal TSV, mapping tags to the
//! public taxonomy, rendering realistic text, deduplication, grouped splits, augmentation,
//! Parquet shards, and the sample manifest.
//!
//! libpostal's tagged format loses the original whitespace: every token is joined with a
//! space, a line break is the token `|/FSEP`, and punctuation is a separate `,/SEP` token.
//! Rendering rebuilds text a person would write, so the model never learns padded commas.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tessera::AddressLabel;
use tessera::internal::{TokenClass, fnv1a, tokenize};

use crate::config::Config;

/// One labelled address: text plus non-overlapping component spans in byte offsets.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelledExample {
    /// Row id; augmented copies count down from `u64::MAX` (see `augment::original_id`).
    pub id: u64,
    /// Hash of the split key; every row of one record shares it.
    pub group_id: u64,
    /// ISO 3166-1 alpha-2, uppercase.
    pub country: String,
    /// The source's language tag, such as `en`, `ja`, `ja_rm`.
    pub language: String,
    /// The address as a person would write it.
    pub text: String,
    /// Labelled components, in order, as byte ranges of `text`.
    pub spans: Vec<Span>,
    /// The split the row's group key hashes to.
    pub split: Split,
    /// Whether this row is a rewritten copy of a training row.
    pub augmented: bool,
}

/// One component of an address, as a byte range of the example's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The component's label.
    pub label: AddressLabel,
    /// UTF-8 byte offset where the component starts.
    pub start: u32,
    /// UTF-8 byte offset where it ends, exclusive.
    pub end: u32,
}

/// Which split a row belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    Train,
    Valid,
    Test,
}

impl Split {
    /// Every split, in file order.
    pub const ALL: [Split; 3] = [Split::Train, Split::Valid, Split::Test];

    /// The split's name as used in shard file names.
    pub fn name(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Valid => "valid",
            Split::Test => "test",
        }
    }
}

/// What the label map says to do with one libpostal tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    Map(AddressLabel),
    /// Keep the text, labelled `O`: venue names, world regions, separator punctuation.
    Outside,
    /// Drop the whole example: search-query generator output.
    Exclude,
    /// Not text: a line break in the rendered address.
    Separator,
}

/// `data/manifests/label-map.json`, as written in Milestone 0.
pub struct LabelMap(HashMap<String, Fate>);

#[derive(Deserialize)]
struct LabelMapFile {
    map: HashMap<String, LabelMapEntry>,
}

#[derive(Deserialize)]
struct LabelMapEntry {
    fate: String,
    to: Option<String>,
}

impl LabelMap {
    /// Reads `label-map.json`; a tag without a known fate, or mapped to `unknown`, is an error.
    pub fn load(path: &Path) -> anyhow::Result<LabelMap> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: LabelMapFile =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let mut out = HashMap::new();
        for (tag, entry) in file.map {
            let fate = match (entry.fate.as_str(), entry.to.as_deref()) {
                ("map", Some(to)) => match AddressLabel::from_str_label(to) {
                    Some(AddressLabel::Unknown) | None => {
                        bail!("tag `{tag}` maps to `{to}`, which is not a training label")
                    }
                    Some(l) => Fate::Map(l),
                },
                ("outside", None) => Fate::Outside,
                ("exclude_example", None) => Fate::Exclude,
                ("separator", None) => Fate::Separator,
                (fate, to) => bail!("tag `{tag}` has fate `{fate}` with target {to:?}"),
            };
            out.insert(tag, fate);
        }
        Ok(LabelMap(out))
    }

    fn fate(&self, tag: &str) -> anyhow::Result<Fate> {
        self.0
            .get(tag)
            .copied()
            .with_context(|| format!("tag `{tag}` is not in the label map"))
    }
}

/// One libpostal line, split into its columns and `(token, tag)` pairs.
struct RawRecord<'a> {
    language: &'a str,
    /// ISO 3166-1 alpha-2, lowercase, as the source writes it.
    country: &'a str,
    pairs: Vec<(&'a str, &'a str)>,
}

fn parse_line(line: &str) -> Option<RawRecord<'_>> {
    // The files use CRLF line endings; the last label would otherwise read "country\r".
    let line = line.trim_end_matches(['\r', '\n']);
    let mut cols = line.splitn(3, '\t');
    let language = cols.next()?;
    let country = cols.next()?;
    let tagged = cols.next()?;
    let mut pairs = Vec::new();
    for piece in tagged.split(' ').filter(|p| !p.is_empty()) {
        // A token may itself contain `/`, as in `//house_number`, so split on the last one.
        let cut = piece.rfind('/')?;
        pairs.push((&piece[..cut], &piece[cut + 1..]));
    }
    (!pairs.is_empty()).then_some(RawRecord {
        language,
        country,
        pairs,
    })
}

/// An address as components and the separators between them. Augmentation edits these, and
/// rendering rebuilds text and spans from them, so labels never drift from their text.
#[derive(Debug, Clone, PartialEq)]
pub struct Pieces {
    /// The components and unlabelled stretches, in order.
    pub items: Vec<Piece>,
    /// `seps[i]` sits between `items[i]` and `items[i + 1]`.
    pub seps: Vec<String>,
}

/// One component or stretch of unlabelled text, with the label it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// `None` for text outside the address, such as a venue name.
    pub label: Option<AddressLabel>,
    /// The piece's text, without its separators.
    pub text: String,
}

/// Why a raw record did not become an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Rejection {
    Malformed,
    Excluded,
    NoComponent,
    ForeignPostcode,
    PoBoxWithRoad,
    /// A Japanese address written out as its kana reading (`ひろしまけんひろしまし`), which
    /// nobody writes; glued together it is one token spanning several labels.
    KanaReading,
    /// A label boundary falls inside a token, so the model could never predict it.
    CutsToken,
    /// A road whose type word was cut short by the source (`Kew Foot Roa`, `7th Stre`).
    TruncatedRoad,
    /// A Georgian row whose country is written `AB`, the code the source uses for Abkhazia.
    ForeignCountry,
    /// A component holding text of another kind: a whole address in a Japanese house number,
    /// or a map-export log as a road.
    Mislabelled,
    /// A span out of order, off a char boundary, or padded with whitespace.
    InvalidSpan,
}

/// Tokens that attach to the previous word without a space.
fn closes(tok: &str) -> bool {
    matches!(tok, "," | "." | ";" | ":" | ")" | "]" | "!" | "?")
}

/// Tokens that attach to the next word without a space.
fn opens(tok: &str) -> bool {
    matches!(tok, "(" | "[")
}

/// Han, kana, CJK punctuation such as `〒`, and fullwidth forms: written without spaces.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3000}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}' | '\u{FF01}'..='\u{FF60}')
}

/// Whether two tokens of one component are written with no space between them, as Japanese
/// is: `静岡県`, `１丁目`, `北3条西28`. libpostal tags CJK text one character at a time with
/// spaces between, which real text never has.
fn glued_within(left: &str, right: &str) -> bool {
    let (Some(a), Some(b)) = (left.chars().next_back(), right.chars().next()) else {
        return false;
    };
    let joins = |c: char| c.is_ascii_digit() || c == '-';
    (is_cjk(a) && (is_cjk(b) || joins(b))) || (is_cjk(b) && joins(a))
}

/// `c`'s kana script: 1 hiragana, 2 katakana, 0 neither.
fn kana(c: char) -> u8 {
    match c {
        '\u{3040}'..='\u{309F}' => 1,
        '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}' => 2,
        _ => 0,
    }
}

/// Whether two components are written with no separator: as within a component, except where
/// both sides are the same kana script. The tokenizer keeps a kana run as one token, so
/// `ロンドン` glued to `イギリス` would put a label boundary inside a token.
fn glued(left: &str, right: &str) -> bool {
    let (Some(a), Some(b)) = (left.chars().next_back(), right.chars().next()) else {
        return false;
    };
    glued_within(left, right) && !(kana(a) != 0 && kana(a) == kana(b))
}

/// Joins tokens the way they would be typed: no space before closing punctuation, after an
/// opening bracket, or between CJK characters.
fn join_tokens(tokens: &[&str]) -> String {
    let mut out = String::new();
    for (i, tok) in tokens.iter().enumerate() {
        if i > 0 && !closes(tok) && !opens(tokens[i - 1]) && !glued_within(tokens[i - 1], tok) {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

/// Builds pieces from a record: consecutive tokens with one tag form a component, `FSEP`
/// becomes a line break, `SEP` punctuation becomes part of the separator.
fn pieces_from_record(
    record: &RawRecord<'_>,
    map: &LabelMap,
    outside: &mut BTreeMap<String, u64>,
) -> anyhow::Result<Result<Pieces, Rejection>> {
    if record.country.eq_ignore_ascii_case("jp") && record.language.starts_with("ja_kana") {
        return Ok(Err(Rejection::KanaReading));
    }
    let mut items: Vec<Piece> = Vec::new();
    let mut seps: Vec<String> = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    let mut run_tag: Option<&str> = None;
    let mut run_label: Option<AddressLabel> = None;
    let mut pending_sep = String::new();

    let flush = |items: &mut Vec<Piece>,
                 seps: &mut Vec<String>,
                 run: &mut Vec<&str>,
                 label: Option<AddressLabel>,
                 pending: &mut String| {
        if run.is_empty() {
            return;
        }
        let text = join_tokens(run);
        if let Some(prev) = items.last() {
            let sep = std::mem::take(pending);
            let sep = match sep.as_str() {
                "" if glued(&prev.text, &text) => String::new(),
                "" => " ".to_string(),
                _ => sep,
            };
            seps.push(sep);
        } else {
            pending.clear();
        }
        items.push(Piece { label, text });
        run.clear();
    };

    for &(tok, tag) in &record.pairs {
        if tok.is_empty() {
            return Ok(Err(Rejection::Malformed));
        }
        // OpenStreetMap places Japanese addresses on their island (`本州`), which no Japanese
        // address writes; keeping it would teach a component real input never has.
        if tag == "island" && record.country.eq_ignore_ascii_case("jp") {
            continue;
        }
        match map.fate(tag)? {
            Fate::Exclude => return Ok(Err(Rejection::Excluded)),
            Fate::Separator => {
                flush(&mut items, &mut seps, &mut run, run_label, &mut pending_sep);
                run_tag = None;
                pending_sep = "\n".to_string();
            }
            Fate::Outside if tag == "SEP" => {
                flush(&mut items, &mut seps, &mut run, run_label, &mut pending_sep);
                run_tag = None;
                if pending_sep != "\n" {
                    // A hyphen between Japanese words joins them: `セブン-イレブン`.
                    let cjk_before = items
                        .last()
                        .is_some_and(|i| i.text.chars().next_back().is_some_and(is_cjk));
                    pending_sep = if tok == "-" && cjk_before {
                        "-".to_string()
                    } else if tok == "-" {
                        " - ".to_string()
                    } else {
                        format!("{tok} ")
                    };
                }
            }
            fate => {
                let label = match fate {
                    Fate::Map(l) => Some(l),
                    _ => {
                        *outside.entry(tag.to_string()).or_default() += 1;
                        None
                    }
                };
                if run_tag != Some(tag) {
                    flush(&mut items, &mut seps, &mut run, run_label, &mut pending_sep);
                    run_tag = Some(tag);
                    run_label = label;
                }
                run.push(tok);
            }
        }
    }
    flush(&mut items, &mut seps, &mut run, run_label, &mut pending_sep);
    if items.is_empty() {
        return Ok(Err(Rejection::Malformed));
    }
    if items.iter().all(|p| p.label.is_none()) {
        return Ok(Err(Rejection::NoComponent));
    }
    Ok(Ok(Pieces { items, seps }))
}

/// Filters from the Milestone 0 review: rows that would teach a format that does not occur.
fn check_country_rules(country: &str, p: &Pieces) -> Result<(), Rejection> {
    let has = |l: AddressLabel| p.items.iter().any(|i| i.label == Some(l));
    let postcodes = p
        .items
        .iter()
        .filter(|i| i.label == Some(AddressLabel::Postcode));
    let digits_only = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
    match country {
        // Georgian postcodes are four digits; six-digit ones are Russian, mostly from Abkhazia
        // and South Ossetia, and would define the wrong Georgian address format.
        "GE" if postcodes.clone().any(|pc| !digits_only(&pc.text, 4)) => {
            Err(Rejection::ForeignPostcode)
        }
        // `D-10117` and `DE-10117` are real German forms; `D10117` is a generator artifact.
        "DE" if postcodes.clone().any(|pc| {
            let digits = pc
                .text
                .strip_prefix("DE-")
                .or_else(|| pc.text.strip_prefix("D-"))
                .unwrap_or(&pc.text);
            !digits_only(digits, 5)
        }) =>
        {
            Err(Rejection::ForeignPostcode)
        }
        // Five-digit ZIP, optionally ZIP+4.
        "US" if postcodes.clone().any(|pc| {
            let (zip, plus4) = pc.text.split_once('-').unwrap_or((&pc.text, "0000"));
            !digits_only(zip, 5) || !digits_only(plus4, 4)
        }) =>
        {
            Err(Rejection::ForeignPostcode)
        }
        // `123-4567`, optionally after `〒`, in ASCII or fullwidth digits.
        "JP" if postcodes.clone().any(|pc| !is_jp_postcode(&pc.text)) => {
            Err(Rejection::ForeignPostcode)
        }
        // A German address has a street or a Postfach, never both.
        "DE" if has(AddressLabel::PoBox) && has(AddressLabel::Road) => {
            Err(Rejection::PoBoxWithRoad)
        }
        _ => Ok(()),
    }
}

/// Share of each country's quota that may be rows with no street, house number, or PO box.
const MAX_PLACE_ONLY_SHARE: f64 = 0.15;

/// Road types the GB and US sources write in full or in their standard short form; a last word
/// that is a strict prefix of one of these and not a standard short form was cut off.
const ROAD_TYPES: &[&str] = &[
    "road",
    "street",
    "avenue",
    "drive",
    "lane",
    "place",
    "court",
    "close",
    "crescent",
    "terrace",
    "gardens",
    "boulevard",
    "parkway",
    "highway",
    "square",
    "circle",
];
/// Standard short forms, and whole words that happen to start a road type (`Hyde Park`,
/// `Covent Garden`, `High`).
const ROAD_TYPE_SHORT: &[&str] = &[
    "rd", "st", "str", "la", "cr", "ave", "av", "dr", "ln", "pl", "ct", "cl", "cres", "ter",
    "terr", "gdns", "blvd", "pkwy", "hwy", "sq", "cir", "park", "garden", "high",
];

/// England's statistical regions, which OpenStreetMap tags as `state_district` and no one
/// writes in an address.
const GB_STATISTICAL_REGIONS: &[&str] = &[
    "north east england",
    "north west england",
    "yorkshire and the humber",
    "east midlands",
    "west midlands",
    "east of england",
    "south east",
    "south east england",
    "south west england",
    "south west",
    "north west",
    "north east",
    "w midlands",
    "wst midlands",
    "e midlands",
    "w mids",
    "e mids",
    "west mids",
    "east mids",
];

/// Japan's 47 prefectures, as a city piece that repeats one begins.
const JP_PREFECTURES: [&str; 47] = [
    "北海道",
    "青森県",
    "岩手県",
    "宮城県",
    "秋田県",
    "山形県",
    "福島県",
    "茨城県",
    "栃木県",
    "群馬県",
    "埼玉県",
    "千葉県",
    "東京都",
    "神奈川県",
    "新潟県",
    "富山県",
    "石川県",
    "福井県",
    "山梨県",
    "長野県",
    "岐阜県",
    "静岡県",
    "愛知県",
    "三重県",
    "滋賀県",
    "京都府",
    "大阪府",
    "兵庫県",
    "奈良県",
    "和歌山県",
    "鳥取県",
    "島根県",
    "岡山県",
    "広島県",
    "山口県",
    "徳島県",
    "香川県",
    "愛媛県",
    "高知県",
    "福岡県",
    "佐賀県",
    "長崎県",
    "熊本県",
    "大分県",
    "宮崎県",
    "鹿児島県",
    "沖縄県",
];

/// A Japanese city piece that runs on past its `市` into a ward, town, or lot
/// (`浜松市中区和合町`, `横浜市戸塚`): the source's label covers several components.
/// `四日市市`, `市川市`, and the towns `上市町` and `余市町` are single names.
fn city_holds_more(text: &str) -> bool {
    text.char_indices()
        .find(|&(k, c)| c == '市' && k > 0)
        .is_some_and(|(k, c)| !matches!(&text[k + c.len_utf8()..], "" | "市" | "町" | "村"))
}

/// A city piece that begins with a prefecture (`東京都調布市`) gives it to a region piece, or
/// drops it when the row already has that region.
fn strip_prefecture_from_city(p: &mut Pieces) {
    use AddressLabel as L;
    let Some(c) = p.items.iter().position(|i| i.label == Some(L::City)) else {
        return;
    };
    let Some(pref) = JP_PREFECTURES
        .iter()
        .find(|pref| p.items[c].text.starts_with(**pref) && p.items[c].text.len() > pref.len())
    else {
        return;
    };
    let rest = p.items[c].text[pref.len()..].to_string();
    // `京都府` is a prefecture and `京都市` its city: only a full prefecture name is cut.
    let has_region = p.items.iter().any(|i| i.label == Some(L::Region));
    p.items[c].text = rest;
    if !has_region {
        p.items.insert(
            c,
            Piece {
                label: Some(L::Region),
                text: pref.to_string(),
            },
        );
        p.seps.insert(c, String::new());
    }
}

const NYC_BOROUGHS: &[&str] = &[
    "manhattan",
    "brooklyn",
    "queens",
    "the bronx",
    "bronx",
    "staten island",
];

/// Fixes what the source writes in a way nobody does, per country, before the country rules:
/// Japanese in its written large-to-small order, GB with the postcode before the country, and
/// labels normalized where the source is inconsistent.
fn normalize_source(country: &str, p: &mut Pieces) -> Result<(), Rejection> {
    use AddressLabel as L;
    let lower = |t: &str| t.to_lowercase().trim_end_matches('.').to_string();
    match country {
        "GB" | "US" => {
            for road in p.items.iter().filter(|i| i.label == Some(L::Road)) {
                let Some(last) = road.text.split_whitespace().next_back().map(lower) else {
                    continue;
                };
                // `Devereux R`, `Coles L`: a single letter that is not a compass point, unless a
                // road type comes before it, as in `Avenue J` or `County Road F`.
                let before = road
                    .text
                    .split_whitespace()
                    .rev()
                    .nth(1)
                    .map(lower)
                    .unwrap_or_default();
                let lettered = [
                    "avenue", "ave", "road", "rd", "highway", "hwy", "route", "county", "street",
                    "st", "lane",
                ]
                .contains(&before.as_str());
                let letter = last.chars().count() == 1
                    && last.chars().all(|c| c.is_alphabetic())
                    && !matches!(last.as_str(), "n" | "s" | "e" | "w")
                    && !lettered;
                let cut = letter
                    || ROAD_TYPES
                        .iter()
                        .any(|t| t.len() > last.len() && last.len() >= 2 && t.starts_with(&last));
                if cut && !ROAD_TYPE_SHORT.contains(&last.as_str()) {
                    return Err(Rejection::TruncatedRoad);
                }
            }
        }
        _ => {}
    }
    match country {
        "GB" => {
            drop_pieces(p, |i| {
                i.label == Some(L::District)
                    && GB_STATISTICAL_REGIONS.contains(&lower(&i.text).as_str())
            });
            let at = |l: L, p: &Pieces| p.items.iter().position(|i| i.label == Some(l));
            if let (Some(c), Some(pc)) = (at(L::Country, p), at(L::Postcode, p))
                && pc == c + 1
            {
                p.items.swap(c, pc);
            }
            if let Some(c) = at(L::Country, p)
                && c > 0
                && p.seps[c - 1] == " "
            {
                p.seps[c - 1] = "\n".into();
            }
        }
        "US" => {
            // A borough is written as the city (`Brooklyn, NY`) unless the row already has one
            // (`Brooklyn / New York City, NY`), where it stays a district.
            if !p.items.iter().any(|i| i.label == Some(L::City)) {
                for i in &mut p.items {
                    if i.label == Some(L::District)
                        && NYC_BOROUGHS.contains(&lower(&i.text).as_str())
                    {
                        i.label = Some(L::City);
                    }
                }
            }
        }
        "GE" => {
            if p.items
                .iter()
                .any(|i| i.label == Some(L::Country) && i.text == "AB")
            {
                return Err(Rejection::ForeignCountry);
            }
            // A map-export log (`TURA (16: 47: 37 ...) Export ... Postscript`) as a road.
            if p.items.iter().any(|i| {
                i.label == Some(L::Road) && (i.text.chars().count() > 80 || i.text.contains(": "))
            }) {
                return Err(Rejection::Mislabelled);
            }
        }
        "JP" if p.items.iter().any(|i| {
            i.label == Some(L::HouseNumber)
                && i.text.chars().any(|c| {
                    is_cjk(c) && !"番号地丁目の－ー".contains(c) && !('０'..='９').contains(&c)
                })
        }) =>
        {
            return Err(Rejection::Mislabelled);
        }
        "JP" => {
            // Readings also come tagged `ja`: `にほん / 佐賀 / さがし`. A country, region, or city
            // written only in hiragana is a reading, not an address.
            let reading = p.items.iter().any(|i| {
                matches!(i.label, Some(L::Country | L::Region | L::City))
                    && !i.text.is_empty()
                    && i.text
                        .chars()
                        .all(|c| ('\u{3040}'..='\u{309F}').contains(&c))
            });
            if reading {
                return Err(Rejection::KanaReading);
            }
            normalize_japanese(p);
            if p.items
                .iter()
                .any(|i| i.label == Some(L::City) && city_holds_more(&i.text))
            {
                return Err(Rejection::Mislabelled);
            }
        }
        "DE" => {
            // `D-85049`: the old country prefix is printed before the postcode, not part of it.
            if let Some(k) = p.items.iter().position(|i| i.label == Some(L::Postcode)) {
                let prefix = ["DE-", "D-"]
                    .into_iter()
                    .find(|x| p.items[k].text.starts_with(x));
                if let Some(prefix) = prefix {
                    p.items[k].text = p.items[k].text[prefix.len()..].to_string();
                    p.items.insert(
                        k,
                        Piece {
                            label: None,
                            text: prefix.to_string(),
                        },
                    );
                    p.seps.insert(k, String::new());
                }
            }
        }
        _ => {}
    }
    if p.items.iter().all(|i| i.label.is_none()) {
        return Err(Rejection::NoComponent);
    }
    Ok(())
}

/// Removes every piece matching `f` with one of its separators.
fn drop_pieces(p: &mut Pieces, f: impl Fn(&Piece) -> bool) {
    let mut i = 0;
    while i < p.items.len() {
        if f(&p.items[i]) {
            p.items.remove(i);
            if i < p.seps.len() {
                p.seps.remove(i);
            } else if i > 0 {
                p.seps.remove(i - 1);
            }
        } else {
            i += 1;
        }
    }
}

/// A city that repeats its prefecture in Japanese script (`東京都` then `東京都港区`) keeps
/// only its own name. When the source wrote the prefecture without its suffix (`愛知`, then
/// `愛知県江南市`), the suffix moves to the prefecture. A city named after its prefecture
/// (`京都` then `京都市`) and romanized names are left alone.
fn strip_repeated_prefecture(p: &mut Pieces) {
    use AddressLabel as L;
    let Some(r) = p.items.iter().position(|i| i.label == Some(L::Region)) else {
        return;
    };
    if !p.items[r].text.chars().all(is_cjk) {
        return;
    }
    let region = p.items[r].text.clone();
    let Some(c) = p.items.iter().position(|i| {
        i.label == Some(L::City) && i.text.starts_with(region.as_str()) && i.text != region
    }) else {
        return;
    };
    let rest = p.items[c].text[region.len()..].to_string();
    let mut chars = rest.chars();
    let (first, second) = (chars.next(), chars.next());
    let suffix = |ch: char| "都道府県".contains(ch);
    let lone = |ch: char| "市区町村".contains(ch);
    match (first, second) {
        (Some(f), Some(_)) if suffix(f) => {
            p.items[r].text.push(f);
            p.items[c].text = rest[f.len_utf8()..].to_string();
        }
        (Some(f), _) if is_cjk(f) && !lone(f) => {
            p.items[c].text = rest;
        }
        _ => {}
    }
}

/// Whether more of the row's labelled words are in Japanese script than in Latin; digits and
/// postcodes count for neither.
pub(crate) fn japanese_script(p: &Pieces) -> bool {
    // Han and kana only: a fullwidth digit or `〒` is written in any script.
    let letter =
        |c: char| is_cjk(c) && !matches!(c, '\u{3000}'..='\u{303F}' | '\u{FF01}'..='\u{FF60}');
    let labelled = || p.items.iter().filter(|i| i.label.is_some());
    let cjk = labelled().filter(|i| i.text.chars().any(letter)).count();
    let latin = labelled()
        .filter(|i| i.text.chars().any(|c| c.is_ascii_alphabetic()))
        .count();
    cjk > latin
}

/// Japanese labelled one way and, in Japanese script, in the order it is written: country,
/// postcode, prefecture, city, ward, town and block, street, lot, any building or venue name,
/// then floor and room; the source writes most rows the other way round. A designated city's
/// ward (`区`) is a district and a Tokyo ward a city; a town with its block is one suburb.
/// Regional groupings (`関東地方`), `（丁目なし）` placeholders and the source's synthetic PO
/// boxes are dropped. Romanized rows keep their Western order.
fn normalize_japanese(p: &mut Pieces) {
    use AddressLabel as L;
    drop_pieces(p, |i| {
        let t = i.text.as_str();
        let lower = t.to_lowercase();
        t.ends_with("地方")
            || ["chiho", "chihō", "chuho", "chūhō", " region", "région"]
                .iter()
                .any(|w| lower.contains(w))
            || t.contains("丁目なし")
            || i.label == Some(L::PoBox)
    });
    for i in &mut p.items {
        // The source cuts prefectures short (`北海`, `長野`, `徳島`); written without its
        // suffix a prefecture teaches that any two kanji before a city are one (`福知` of
        // `福知山市`).
        if i.label == Some(L::Region)
            && let Some(full) = JP_PREFECTURES
                .iter()
                .find(|full| full.strip_suffix(['県', '府', '都', '道']) == Some(i.text.as_str()))
        {
            i.text = full.to_string();
        }
        // `豊洲豊洲2`: the source writes some towns twice in one piece.
        if i.label == Some(L::Suburb) {
            let chars: Vec<char> = i.text.chars().collect();
            let doubled = (2..=chars.len() / 2)
                .rev()
                .find(|&n| chars[..n] == chars[n..2 * n] && chars[..n].iter().all(|&c| is_cjk(c)));
            if let Some(n) = doubled {
                i.text = chars[n..].iter().collect();
            }
        }
    }
    strip_repeated_prefecture(p);
    strip_prefecture_from_city(p);
    split_ward_from_city(p);
    split_county_from_town(p);
    let has_city = p.items.iter().any(|i| {
        let lower = i.text.to_lowercase();
        i.label == Some(L::City)
            && (i.text.ends_with('市') || lower.ends_with("-shi") || lower.ends_with(" shi"))
    });
    let any_city = p.items.iter().any(|i| i.label == Some(L::City));
    let tokyo = p.items.iter().any(|i| {
        let lower = i.text.to_lowercase();
        i.label == Some(L::Region)
            && (i.text.starts_with("東京")
                || lower.starts_with("tokyo")
                || lower.starts_with("tōkyō"))
    });
    let named_town = p
        .items
        .iter()
        .any(|i| i.label == Some(L::Suburb) && !is_bare_chome(&i.text));
    let bare_block = |k: usize| {
        p.items
            .get(k)
            .is_some_and(|i| i.label == Some(L::Suburb) && is_bare_chome(&i.text))
    };
    let beside_block: Vec<bool> = (0..p.items.len())
        .map(|k| bare_block(k + 1) || (k > 0 && bare_block(k - 1)))
        .collect();
    // Towns relabelled from roads, which follow the source's own town when both are present.
    let mut from_road = vec![false; p.items.len()];
    for ((i, beside_block), from_road) in p.items.iter_mut().zip(beside_block).zip(&mut from_road) {
        let lower = i.text.to_lowercase();
        let ward = (i.text.ends_with('区') && !i.text.ends_with("地区"))
            || lower.ends_with("-ku")
            || lower.ends_with(" ku");
        if ward && (i.label == Some(L::Suburb) || (i.label == Some(L::City) && has_city)) {
            i.label = Some(L::District);
        }
        // A ward of Tokyo with no city above it is one of the special wards, which are cities.
        if ward && i.label == Some(L::District) && !any_city && tokyo {
            i.label = Some(L::City);
        }
        // Japan addresses by town and block, not by street; the source's `road` is mostly
        // the town (`錦町`, `Maruyama-cho`), the town with its block (`栄三丁目`), the town
        // beside a bare block (`栄` with `３丁目`), or a town or 字 name in Japanese script
        // (`烏ヶ辻`, `石脇字下長老沼`) that no street word ends.
        let japanese_name = i.text.chars().all(|c| is_cjk(c) || japanese_numeral(c))
            && i.text.chars().any(|c| is_cjk(c) && !japanese_numeral(c));
        let town = !is_japanese_street(&i.text)
            && (is_japanese_town(&i.text)
                || japanese_name
                || (beside_block && !named_town && i.text.chars().all(is_cjk)));
        if i.label == Some(L::Road) && town {
            i.label = Some(L::Suburb);
            *from_road = true;
        }
    }
    // A road that only repeated the town is now a second copy of it.
    let mut k = 1;
    while k < p.items.len() {
        if p.items[k].label == Some(L::Suburb) && p.items[..k].contains(&p.items[k]) {
            p.items.remove(k);
            p.seps.remove(k - 1);
            from_road.remove(k);
        } else {
            k += 1;
        }
    }
    // A road that repeated the town with more or less of it (`銀座` and `銀座4`, `南池袋3` and
    // `南池袋2`) names the same town: the fuller one stays, the source's own when they differ.
    loop {
        let suburbs: Vec<usize> = (0..p.items.len())
            .filter(|&k| p.items[k].label == Some(L::Suburb))
            .collect();
        let same_town = suburbs
            .iter()
            .flat_map(|&c| suburbs.iter().map(move |&o| (c, o)))
            .filter(|&(c, o)| c != o && from_road[c])
            .find_map(|(c, o)| {
                let (road, town) = (&p.items[c].text, &p.items[o].text);
                let shared = road
                    .chars()
                    .zip(town.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                if town.starts_with(road.as_str())
                    || shared >= 2 && !road.starts_with(town.as_str())
                {
                    Some(c)
                } else if road.starts_with(town.as_str()) {
                    Some(o)
                } else {
                    None
                }
            });
        let Some(k) = same_town else {
            break;
        };
        p.items.remove(k);
        p.seps
            .remove(k.saturating_sub(1).min(p.seps.len().saturating_sub(1)));
        from_road.remove(k);
    }
    if !japanese_script(p) {
        return;
    }
    let rank = |(i, from_road): &(Piece, bool)| match i.label {
        Some(L::Country) => (0, 0),
        Some(L::Postcode) => (1, 0),
        Some(L::Region) => (2, 0),
        Some(L::City) => (3, 0),
        Some(L::District) => (4, 0),
        Some(L::Suburb) if is_bare_chome(&i.text) => (5, 2),
        Some(L::Suburb) => (5, u8::from(*from_road)),
        Some(L::Road) => (6, 0),
        Some(L::HouseNumber) => (7, 0),
        Some(L::PoBox) => (8, 0),
        Some(L::Unknown) | None => (9, 0),
        // The floor and room follow the building's name: `AKIBA Place 2階`.
        Some(L::Level) => (10, 0),
        Some(L::Unit) => (11, 0),
    };
    let mut ranked: Vec<(Piece, bool)> = std::mem::take(&mut p.items)
        .into_iter()
        .zip(from_road)
        .collect();
    ranked.sort_by_key(rank);
    p.items = ranked.into_iter().map(|(i, _)| i).collect();
    p.seps = (1..p.items.len())
        .map(|k| japanese_separator(&p.items[k - 1], &p.items[k]).to_string())
        .collect();
    merge_town_and_block(p);
}

/// Numerals a Japanese block number is written with.
fn japanese_numeral(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c) || "一二三四五六七八九十".contains(c)
}

/// `３丁目`, `三丁目`, `2 chome`: a block number with no town before it.
fn is_bare_chome(text: &str) -> bool {
    let rest = text.trim_start_matches(japanese_numeral).trim_start();
    rest.len() < text.len() && (rest == "丁目" || rest.eq_ignore_ascii_case("chome"))
}

/// A named street (`職安通り`, `松戸停車場線`, `杉戸バイパス`), or a piece that repeats the
/// prefecture, city, or ward before the town (`東京都世田谷区太子堂`, `世田谷区太子堂`). Towns
/// named with those characters (`府中町`, `市谷田町`, `八日市町`) are not.
fn is_japanese_street(text: &str) -> bool {
    let town = text.trim_end_matches(|c: char| japanese_numeral(c) || c == '丁' || c == '目');
    let repeats_place = JP_PREFECTURES.iter().any(|p| text.starts_with(p))
        || text.char_indices().any(|(k, c)| {
            k > 0
                && (c == '市' || c == '区')
                && !matches!(&text[k + c.len_utf8()..], "" | "町" | "村")
        });
    let route = ["国道", "県道", "府道", "都道", "道道"]
        .iter()
        .any(|w| text.starts_with(w))
        || text
            .strip_suffix('号')
            .is_some_and(|t| t.ends_with(japanese_numeral));
    ["通り", "通", "線", "街道", "道", "バイパス", "筋"]
        .iter()
        .any(|w| town.ends_with(w))
        || route
        || repeats_place
}

/// A town name, with or without its block: `錦町`, `栄三丁目`, `Maruyama-cho`, `1 chome`, or
/// a romanized name with no street word in it (`Marunouchi`, `Minami Ikebukuro`).
fn is_japanese_town(text: &str) -> bool {
    let lower = text.to_lowercase();
    let romanized = !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_alphabetic() && !is_cjk(c) || c == ' ' || c == '-')
        && !lower.split([' ', '-']).any(|w| {
            [
                "dori",
                "dōri",
                "doori",
                "douri",
                "tori",
                "toori",
                "tōri",
                "suji",
                "michi",
                "lane",
                "street",
                "st",
                "avenue",
                "ave",
                "road",
                "rd",
                "kaido",
                "kaidō",
                "sen",
                "line",
                "bypass",
                "highway",
                "expressway",
                "route",
                "ku",
                "shi",
            ]
            .contains(&w)
        });
    if romanized {
        return true;
    }
    let cjk_town = text.chars().all(|c| is_cjk(c) || japanese_numeral(c))
        && (text.ends_with('町') || text.ends_with('村') || text.ends_with("丁目"));
    cjk_town
        || ["-cho", "-chō", "-machi", " chome", "-chome"]
            .iter()
            .any(|s| lower.ends_with(s))
}

/// A county written into its town's piece (`中川郡池田町`) becomes its own district piece.
fn split_county_from_town(p: &mut Pieces) {
    use AddressLabel as L;
    let Some(c) = p.items.iter().position(|i| {
        i.label == Some(L::City)
            && i.text.chars().all(is_cjk)
            && (i.text.ends_with('町') || i.text.ends_with('村'))
    }) else {
        return;
    };
    let Some(cut) = p.items[c]
        .text
        .find('郡')
        .filter(|&k| k > 0)
        .map(|k| k + '郡'.len_utf8())
    else {
        return;
    };
    if cut >= p.items[c].text.len() {
        return;
    }
    let town = p.items[c].text.split_off(cut);
    let county = std::mem::replace(&mut p.items[c].text, town);
    p.items.insert(
        c,
        Piece {
            label: Some(L::District),
            text: county,
        },
    );
    p.seps.insert(c, String::new());
}

/// A designated city's ward written into the city piece (`札幌市中央区`) becomes its own
/// district piece; where the row already has that district (`横浜市中区` then `中区`), the
/// ward is only cut from the city.
fn split_ward_from_city(p: &mut Pieces) {
    use AddressLabel as L;
    let Some(c) = p.items.iter().position(|i| {
        i.label == Some(L::City) && i.text.chars().all(is_cjk) && i.text.ends_with('区')
    }) else {
        return;
    };
    let Some(cut) = p.items[c]
        .text
        .find('市')
        .filter(|&k| k > 0)
        .map(|k| k + '市'.len_utf8())
    else {
        return;
    };
    if cut >= p.items[c].text.len() {
        return;
    }
    let ward = p.items[c].text.split_off(cut);
    if p.items
        .get(c + 1)
        .is_some_and(|n| n.label == Some(L::District) && n.text == ward)
    {
        p.items[c + 1].text = ward;
        p.seps[c] = String::new();
    } else if !p
        .items
        .iter()
        .any(|i| i.label == Some(L::District) && i.text == ward)
    {
        p.items.insert(
            c + 1,
            Piece {
                label: Some(L::District),
                text: ward,
            },
        );
        p.seps.insert(c, String::new());
    }
}

/// A town and what follows it written together, its block (`栄` `３丁目`) or a 字 or
/// district name (`応神町` `中原字宮前`), become one suburb, `栄３丁目`, as they are read. A
/// town that already carries a number (`飯寺北一丁目`) takes no further name.
fn merge_town_and_block(p: &mut Pieces) {
    use AddressLabel as L;
    let mut k = 1;
    while k < p.items.len() {
        let (prev, next) = (&p.items[k - 1].text, &p.items[k].text);
        let glued = p.items[k - 1].label == Some(L::Suburb)
            && p.items[k].label == Some(L::Suburb)
            && p.seps[k - 1].is_empty()
            && !is_bare_chome(prev);
        let block = is_bare_chome(next);
        if glued && (block || !prev.chars().any(japanese_numeral)) {
            let next = p.items.remove(k);
            p.seps.remove(k - 1);
            // `栄三丁目` then `４丁目`: two blocks for one town; the second is dropped.
            if !(block && p.items[k - 1].text.ends_with("丁目")) {
                p.items[k - 1].text.push_str(&next.text);
            }
        } else {
            k += 1;
        }
    }
}

/// The separator between two Japanese components: none where the text runs on, a space
/// around a postcode and after the country, a line break before a venue name.
pub(crate) fn japanese_separator(a: &Piece, b: &Piece) -> &'static str {
    use AddressLabel as L;
    if b.label.is_none() {
        "\n"
    } else if matches!(a.label, Some(L::Postcode | L::Country)) || b.label == Some(L::Postcode) {
        " "
    } else if glued(&a.text, &b.text) {
        ""
    } else {
        " "
    }
}

/// `330-0802`, `〒044-0054`, `３３０－０８０２`: three digits, a hyphen, four digits.
fn is_jp_postcode(text: &str) -> bool {
    let digits: String = text
        .strip_prefix('〒')
        .unwrap_or(text)
        .trim()
        .chars()
        .map(|c| match c {
            '０'..='９' => char::from_u32(c as u32 - '０' as u32 + '0' as u32).unwrap_or(c),
            '－' | 'ー' | '‐' | '−' | '‑' => '-',
            c => c,
        })
        .collect();
    let Some((a, b)) = digits.split_once('-') else {
        return false;
    };
    a.len() == 3 && b.len() == 4 && (a.to_string() + b).bytes().all(|x| x.is_ascii_digit())
}

/// Text and spans from pieces: each labelled piece yields one span over its text.
pub fn render_pieces(p: &Pieces) -> (String, Vec<Span>) {
    let mut text = String::new();
    let mut spans = Vec::new();
    for (i, item) in p.items.iter().enumerate() {
        if i > 0 {
            text.push_str(&p.seps[i - 1]);
        }
        let start = text.len() as u32;
        text.push_str(&item.text);
        if let Some(label) = item.label {
            spans.push(Span {
                label,
                start,
                end: text.len() as u32,
            });
        }
    }
    (text, spans)
}

fn is_sep_char(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | ';' | '-')
}

/// Inverse of [`render_pieces`] for text built by it: labelled spans become pieces, and the
/// text between them is split into separators and unlabelled pieces.
pub fn decompose(text: &str, spans: &[Span]) -> Pieces {
    let mut items: Vec<Piece> = Vec::new();
    let mut seps: Vec<String> = Vec::new();
    let mut pending = String::new();
    let gap = |gap: &str, items: &mut Vec<Piece>, seps: &mut Vec<String>, pending: &mut String| {
        let core = gap.trim_matches(is_sep_char);
        if core.is_empty() {
            pending.push_str(gap);
            return;
        }
        let lead_len = gap.find(core).unwrap_or(0);
        pending.push_str(&gap[..lead_len]);
        if !items.is_empty() {
            seps.push(std::mem::take(pending));
        }
        items.push(Piece {
            label: None,
            text: core.to_string(),
        });
        pending.push_str(&gap[lead_len + core.len()..]);
    };
    let mut pos = 0usize;
    for s in spans {
        let (a, b) = (s.start as usize, s.end as usize);
        gap(&text[pos..a], &mut items, &mut seps, &mut pending);
        if !items.is_empty() {
            seps.push(std::mem::take(&mut pending));
        }
        items.push(Piece {
            label: Some(s.label),
            text: text[a..b].to_string(),
        });
        pos = b;
    }
    gap(&text[pos..], &mut items, &mut seps, &mut pending);
    Pieces { items, seps }
}

/// Lowercase, drop punctuation and whitespace tokens, join with one space.
pub fn normalize(text: &str) -> String {
    let mut out = String::new();
    for t in tokenize(text) {
        if matches!(
            t.class,
            TokenClass::Punct | TokenClass::Space | TokenClass::Newline
        ) {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.extend(text[t.start..t.end].chars().flat_map(char::to_lowercase));
    }
    out
}

fn labelled_text(text: &str, spans: &[Span], label: AddressLabel) -> String {
    spans
        .iter()
        .filter(|s| s.label == label)
        .map(|s| &text[s.start as usize..s.end as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Hash of the normalized text: the same address up to case, spacing, and punctuation.
pub fn fingerprint(text: &str) -> u64 {
    fnv1a(normalize(text).as_bytes(), 0)
}

/// The key that groups variants of one source record, so a record never has variants in two
/// splits. libpostal emits one record with and without its postcode, country, region, and
/// often city or unit, and in either order, so with a road the key is house number and road
/// with abbreviations folded; without a road it is the set of the remaining labelled pieces.
/// Venue names stay out of the key: libpostal writes one place with and without its venue, and
/// those variants must share a split. `prepare` keeps one row per place and layout, so one
/// town's rows do not form one large group.
pub fn group_key(country: &str, text: &str, spans: &[Span]) -> String {
    use AddressLabel as L;
    let house = normalize(&labelled_text(text, spans, L::HouseNumber));
    let road = abbrev::fold_road(country, &normalize(&labelled_text(text, spans, L::Road)));
    if !road.is_empty() {
        return format!("{country}\u{1}{house}\u{1}{road}");
    }
    // Hyphens become spaces only in the key: `Stoke-on-Trent` and `Stoke on Trent` are one
    // place, and the tokenizer keeps a hyphenated word whole.
    let mut pieces: Vec<String> = decompose(text, spans)
        .items
        .into_iter()
        .filter(|p| {
            p.label.is_some() && !matches!(p.label, Some(L::Postcode | L::Country | L::Region))
        })
        .map(|p| normalize(&p.text.replace('-', " ")))
        .filter(|p| !p.is_empty())
        .collect();
    pieces.sort_unstable();
    format!("{country}\u{1}\u{2}{}", pieces.join("\u{1}"))
}

/// A hash of the country and the labelled components in order, venue text left out.
pub fn labelled_fingerprint(country: &str, text: &str, spans: &[Span]) -> u64 {
    let mut key = country.to_string();
    for s in spans {
        key.push('\u{1}');
        key.push_str(s.label.as_str());
        key.push('\u{2}');
        key.push_str(&text[s.start as usize..s.end as usize]);
    }
    fnv1a(key.as_bytes(), 0)
}

/// `fnv1a(key, seed) % 12`: buckets 0 to 9 train, 10 valid, 11 test.
pub fn split_for(seed: u64, group_key: &str) -> Split {
    match fnv1a(group_key.as_bytes(), seed) % 12 {
        10 => Split::Valid,
        11 => Split::Test,
        _ => Split::Train,
    }
}

/// Road-term abbreviations per country, lowercase, whole tokens only.
pub(crate) mod abbrev {
    /// Road-type words and their common abbreviations, per country.
    pub const ROAD_ABBREVIATIONS: &[(&str, &[(&str, &str)])] = &[
        (
            "GB",
            &[
                ("street", "st"),
                ("road", "rd"),
                ("avenue", "ave"),
                ("lane", "ln"),
                ("drive", "dr"),
                ("court", "ct"),
                ("place", "pl"),
                ("square", "sq"),
                ("crescent", "cres"),
                ("gardens", "gdns"),
                ("terrace", "terr"),
            ],
        ),
        (
            "DE",
            &[("straße", "str."), ("strasse", "str."), ("platz", "pl.")],
        ),
        (
            "US",
            &[
                ("street", "st"),
                ("avenue", "ave"),
                ("boulevard", "blvd"),
                ("road", "rd"),
                ("drive", "dr"),
                ("lane", "ln"),
                ("court", "ct"),
                ("place", "pl"),
                ("parkway", "pkwy"),
                ("highway", "hwy"),
                ("north", "n"),
                ("south", "s"),
                ("east", "e"),
                ("west", "w"),
            ],
        ),
        (
            "GE",
            &[
                ("avenue", "ave"),
                ("street", "st"),
                ("გამზირი", "გამზ."),
                ("ქუჩა", "ქ."),
            ],
        ),
    ];

    /// The abbreviation pairs for `country`; empty for a country without a table.
    pub fn table(country: &str) -> &'static [(&'static str, &'static str)] {
        ROAD_ABBREVIATIONS
            .iter()
            .find(|(c, _)| *c == country)
            .map_or(&[], |(_, t)| t)
    }

    /// Maps abbreviated road words back to their full form, on normalized text, so the group
    /// key of `Baker St` and `Baker Street` is the same. Normalization has already dropped the
    /// trailing dot of `str.`. German writes the road type into the name, so there the suffix
    /// is folded too: `hauptstr`, `hauptstrasse`, and `hauptstraße` share a key.
    pub fn fold_road(country: &str, normalized: &str) -> String {
        let t = table(country);
        normalized
            .split(' ')
            .map(|w| {
                let w = t
                    .iter()
                    .find(|(_, short)| short.trim_end_matches('.') == w)
                    .map_or(w, |(full, _)| full);
                if country == "DE" && w != "straße" {
                    for suffix in ["strasse", "str"] {
                        if let Some(stem) = w.strip_suffix(suffix) {
                            return format!("{stem}straße");
                        }
                    }
                }
                w.to_string()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Counts of rows dropped for each reason, for the manifest.
#[derive(Debug, Default, Serialize)]
struct DroppedRows {
    malformed: u64,
    excluded_query: u64,
    no_component: u64,
    foreign_postcode: u64,
    po_box_with_road: u64,
    kana_reading: u64,
    cuts_token: u64,
    truncated_road: u64,
    foreign_country: u64,
    mislabelled: u64,
    invalid_span: u64,
    locality_over_cap: u64,
    /// Augmented copies, not source rows, dropped because a boundary cut a token.
    unencodable_copies: u64,
    duplicate: u64,
    /// Place-only rows whose labelled components, in order, match an earlier row's: the same
    /// place with another venue line.
    same_place: u64,
    over_quota: u64,
}

impl DroppedRows {
    fn add(&mut self, r: Rejection) {
        match r {
            Rejection::Malformed => self.malformed += 1,
            Rejection::Excluded => self.excluded_query += 1,
            Rejection::NoComponent => self.no_component += 1,
            Rejection::ForeignPostcode => self.foreign_postcode += 1,
            Rejection::PoBoxWithRoad => self.po_box_with_road += 1,
            Rejection::KanaReading => self.kana_reading += 1,
            Rejection::CutsToken => self.cuts_token += 1,
            Rejection::TruncatedRoad => self.truncated_road += 1,
            Rejection::ForeignCountry => self.foreign_country += 1,
            Rejection::Mislabelled => self.mislabelled += 1,
            Rejection::InvalidSpan => self.invalid_span += 1,
        }
    }
}

/// The source manifest written in Milestone 0.
#[derive(Deserialize)]
struct SourceManifest {
    source: String,
    url: String,
    sha256: String,
    license: String,
    attribution: String,
    downloaded: String,
}

impl SourceManifest {
    fn load(path: &Path) -> anyhow::Result<SourceManifest> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// The downloaded file lives under `data/raw/libpostal/`, named as in its URL.
    fn local_path(&self) -> PathBuf {
        let name = self.url.rsplit('/').next().unwrap_or("source.tsv.gz");
        PathBuf::from("data/raw/libpostal").join(name)
    }
}

/// `trainer prepare`: streams the source corpus into split, deduplicated, augmented shards
/// and writes the sample manifest.
pub fn prepare(config_path: &Path, date: &str) -> anyhow::Result<()> {
    let cfg = crate::config::load(config_path)?;
    let manifests = PathBuf::from(&cfg.data.manifests);
    let map = LabelMap::load(&manifests.join("label-map.json"))?;
    let fc = cfg.features.to_tessera();
    let source = SourceManifest::load(&manifests.join(format!("{}.json", cfg.data.source)))?;
    let path = source.local_path();
    if !path.exists() {
        bail!(
            "{} is missing; download {} first",
            path.display(),
            source.url
        );
    }
    let quota = |split: Split| match split {
        Split::Train => cfg.data.train_per_country,
        Split::Valid => cfg.data.valid_per_country,
        Split::Test => cfg.data.test_per_country,
    };

    let mut counts: BTreeMap<(String, Split), usize> = BTreeMap::new();
    let mut places: BTreeMap<(String, Split), usize> = BTreeMap::new();
    let mut seen = HashSet::<u64>::new();
    let mut places_seen = HashSet::<u64>::new();
    let mut outside_tags = BTreeMap::<String, u64>::new();
    let mut dropped = DroppedRows::default();
    let mut rows: Vec<LabelledExample> = Vec::new();
    let mut lines_read: u64 = 0;

    let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let reader =
        std::io::BufReader::with_capacity(1 << 20, flate2::read::MultiGzDecoder::new(file));
    for line in reader.lines() {
        let line = line?;
        lines_read += 1;
        if lines_read.is_multiple_of(1_000_000) {
            let filled: Vec<String> = cfg
                .data
                .countries
                .iter()
                .map(|c| {
                    format!(
                        "{c} {}",
                        counts.get(&(c.clone(), Split::Train)).copied().unwrap_or(0)
                    )
                })
                .collect();
            eprintln!(
                "{}M lines, train rows: {}",
                lines_read / 1_000_000,
                filled.join(", ")
            );
        }
        // Cheap country check before any parsing: the second tab-separated column.
        let Some(country_col) = line.split('\t').nth(1) else {
            continue;
        };
        let Some(country) = cfg
            .data
            .countries
            .iter()
            .find(|c| c.eq_ignore_ascii_case(country_col))
        else {
            continue;
        };
        let Some(record) = parse_line(&line) else {
            dropped.malformed += 1;
            continue;
        };
        let pieces = match pieces_from_record(&record, &map, &mut outside_tags)? {
            Ok(p) => p,
            Err(r) => {
                dropped.add(r);
                continue;
            }
        };
        let mut pieces = pieces;
        if let Err(r) = normalize_source(country, &mut pieces) {
            dropped.add(r);
            continue;
        }
        if let Err(r) = check_country_rules(country, &pieces) {
            dropped.add(r);
            continue;
        }
        let (text, spans) = render_pieces(&pieces);
        // Split and quota first: once a country's split is full, the costly checks below and
        // the duplicate set are skipped for the rest of the file.
        let key = group_key(country, &text, &spans);
        let split = split_for(cfg.seed, &key);
        if counts.get(&(country.clone(), split)).copied().unwrap_or(0) >= quota(split) {
            dropped.over_quota += 1;
            continue;
        }
        // Rows that are only a place name (`Leeds United Kingdom`) or a venue and a town are
        // a third of the source; left uncapped they crowd out street addresses.
        let street = spans.iter().any(|s| {
            matches!(
                s.label,
                AddressLabel::Road | AddressLabel::HouseNumber | AddressLabel::PoBox
            )
        });
        let n_places = places.entry((country.clone(), split)).or_default();
        if !street && *n_places as f64 >= quota(split) as f64 * MAX_PLACE_ONLY_SHARE {
            dropped.locality_over_cap += 1;
            continue;
        }
        if crate::dataset::encode(&text, &spans, &fc).is_err() {
            dropped.add(Rejection::CutsToken);
            continue;
        }
        if !crate::check::spans_valid(&text, &spans) {
            dropped.add(Rejection::InvalidSpan);
            continue;
        }
        if !seen.insert(fingerprint(&text)) {
            dropped.duplicate += 1;
            continue;
        }
        // `野田市18` with a venue line and without it is one place written twice.
        if !street && !places_seen.insert(labelled_fingerprint(country, &text, &spans)) {
            dropped.same_place += 1;
            continue;
        }
        if !street {
            *n_places += 1;
        }
        *counts.entry((country.clone(), split)).or_default() += 1;
        rows.push(LabelledExample {
            id: rows.len() as u64,
            group_id: fnv1a(key.as_bytes(), 0),
            country: country.clone(),
            language: record.language.to_string(),
            text,
            spans,
            split,
            augmented: false,
        });
        let full = cfg.data.countries.iter().all(|c| {
            Split::ALL
                .iter()
                .all(|s| counts.get(&(c.clone(), *s)).copied().unwrap_or(0) >= quota(*s))
        });
        if full {
            break;
        }
    }

    for c in &cfg.data.countries {
        for s in Split::ALL {
            let n = counts.get(&(c.clone(), s)).copied().unwrap_or(0);
            if n < quota(s) {
                eprintln!(
                    "{c} {}: the source ran out at {n} of {} rows",
                    s.name(),
                    quota(s)
                );
            }
        }
    }
    let augmented: Vec<LabelledExample> = rows
        .iter()
        .filter(|r| r.split == Split::Train)
        .flat_map(|r| {
            augment::augment_row(
                cfg.seed,
                r,
                cfg.augment.copies,
                &fc,
                &mut dropped.unencodable_copies,
            )
        })
        .collect();
    let mut augmented_counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in &augmented {
        *augmented_counts.entry(r.country.clone()).or_default() += 1;
    }
    rows.extend(augmented);

    let (checks, examples) =
        crate::check::verify(&rows, cfg.augment.copies, &cfg.features.to_tessera(), 4);
    anyhow::ensure!(
        checks.passed(),
        "the prepared sample fails its checks, nothing was written: {checks:?}\n{}",
        examples.join("\n")
    );
    let processed = PathBuf::from(&cfg.data.processed);
    write_shards(&processed, &rows)?;
    write_manifest(
        &cfg,
        &source,
        date,
        &counts,
        &augmented_counts,
        &outside_tags,
        &dropped,
        lines_read,
        &checks,
    )?;
    for c in &cfg.data.countries {
        let got: Vec<String> = Split::ALL
            .iter()
            .map(|s| {
                format!(
                    "{} {}",
                    s.name(),
                    counts.get(&(c.clone(), *s)).copied().unwrap_or(0)
                )
            })
            .collect();
        eprintln!(
            "{c}: {}, augmented {}",
            got.join(", "),
            augmented_counts.get(c).copied().unwrap_or(0)
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_manifest(
    cfg: &Config,
    source: &SourceManifest,
    date: &str,
    counts: &BTreeMap<(String, Split), usize>,
    augmented: &BTreeMap<String, usize>,
    outside_tags: &BTreeMap<String, u64>,
    dropped: &DroppedRows,
    lines_read: u64,
    checks: &crate::check::Checks,
) -> anyhow::Result<()> {
    let mut per_country = serde_json::Map::new();
    for c in &cfg.data.countries {
        let mut m = serde_json::Map::new();
        for s in Split::ALL {
            m.insert(
                s.name().into(),
                counts.get(&(c.clone(), s)).copied().unwrap_or(0).into(),
            );
        }
        m.insert(
            "augmented".into(),
            augmented.get(c).copied().unwrap_or(0).into(),
        );
        per_country.insert(c.clone(), m.into());
    }
    let manifest = serde_json::json!({
        "source": source.source,
        "url": source.url,
        "sha256": source.sha256,
        "license": source.license,
        "attribution": source.attribution,
        "downloaded": source.downloaded,
        "prepared": date,
        "filters": {
            "countries": cfg.data.countries,
            "label_map": "data/manifests/label-map.json",
            "rules": "GE postcodes four digits, DE postcodes five digits, US ZIP or ZIP+4, JP 〒nnn-nnnn; no DE Postfach with a street; JP island, regional-grouping, placeholder and PO box pieces dropped, Japanese-script rows reordered large to small, kana readings rejected; GB statistical regions dropped, postcode before country; NYC boroughs as cities; truncated GB and US road types rejected; GE rows marked AB rejected; rows whose spans cut a token or are invalid rejected; place-only rows capped at 15% of each quota; no rows without an address component; no search queries",
        },
        "counts": per_country,
        "split_seed": cfg.seed,
        "split_rule": "fnv1a64(seed, group_key) % 12: 0-9 train, 10 valid, 11 test",
        "augment_copies": cfg.augment.copies,
        "outside_tags": outside_tags,
        "dropped_rows": dropped,
        "lines_read": lines_read,
        "checks": { "passed": checks.passed(), "counts": checks },
    });
    let path = PathBuf::from(&cfg.data.sample_manifest);
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)? + "\n")
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Writes `train`, `valid`, and `test` Parquet shards under `dir`.
pub fn write_shards(dir: &Path, rows: &[LabelledExample]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for split in Split::ALL {
        let subset: Vec<&LabelledExample> = rows.iter().filter(|r| r.split == split).collect();
        shard::write(&dir.join(format!("{}.parquet", split.name())), &subset)?;
    }
    Ok(())
}

/// Reads one Parquet shard written by [`write_shards`].
pub fn read_shard(path: &Path, split: Split) -> anyhow::Result<Vec<LabelledExample>> {
    shard::read(path, split)
}

/// Parquet layout: flat scalar columns plus three parallel list columns for the spans.
mod shard {
    use std::path::Path;

    use anyhow::Context;
    use polars::prelude::*;
    use tessera::AddressLabel;

    use super::{LabelledExample, Span, Split};

    fn label_index(l: AddressLabel) -> u8 {
        AddressLabel::ALL.iter().position(|x| *x == l).unwrap_or(0) as u8
    }

    /// Writes rows as one Parquet file, spans as three list columns.
    pub fn write(path: &Path, rows: &[&LabelledExample]) -> anyhow::Result<()> {
        let list = |name: &str, f: &dyn Fn(&[Span]) -> Series| -> Column {
            let items: Vec<Series> = rows.iter().map(|r| f(&r.spans)).collect();
            Series::new(name.into(), items).into()
        };
        let mut df = DataFrame::new_infer_height(vec![
            Column::new("id".into(), rows.iter().map(|r| r.id).collect::<Vec<_>>()),
            Column::new(
                "group_id".into(),
                rows.iter().map(|r| r.group_id).collect::<Vec<_>>(),
            ),
            Column::new(
                "country".into(),
                rows.iter().map(|r| r.country.as_str()).collect::<Vec<_>>(),
            ),
            Column::new(
                "language".into(),
                rows.iter().map(|r| r.language.as_str()).collect::<Vec<_>>(),
            ),
            Column::new(
                "text".into(),
                rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
            ),
            list("span_label", &|spans| {
                Series::new(
                    PlSmallStr::EMPTY,
                    spans
                        .iter()
                        .map(|s| label_index(s.label))
                        .collect::<Vec<u8>>(),
                )
            }),
            list("span_start", &|spans| {
                Series::new(
                    PlSmallStr::EMPTY,
                    spans.iter().map(|s| s.start).collect::<Vec<u32>>(),
                )
            }),
            list("span_end", &|spans| {
                Series::new(
                    PlSmallStr::EMPTY,
                    spans.iter().map(|s| s.end).collect::<Vec<u32>>(),
                )
            }),
            Column::new(
                "augmented".into(),
                rows.iter().map(|r| r.augmented).collect::<Vec<_>>(),
            ),
        ])?;
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        ParquetWriter::new(file).finish(&mut df)?;
        Ok(())
    }

    /// Spans must be in order, non-overlapping, inside the text, and on character boundaries,
    /// so a stale or corrupt shard fails here with its path rather than later in slicing.
    fn check_spans(text: &str, spans: Vec<Span>) -> anyhow::Result<Vec<Span>> {
        let mut at = 0usize;
        for sp in &spans {
            let (start, end) = (sp.start as usize, sp.end as usize);
            anyhow::ensure!(
                at <= start && start <= end && end <= text.len(),
                "span {start}..{end} is out of order or outside a text of {} bytes",
                text.len()
            );
            anyhow::ensure!(
                text.is_char_boundary(start) && text.is_char_boundary(end),
                "span {start}..{end} splits a character"
            );
            at = end;
        }
        Ok(spans)
    }

    /// Reads a file written by [`write`], tagging every row with `split`.
    pub fn read(path: &Path, split: Split) -> anyhow::Result<Vec<LabelledExample>> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let df = ParquetReader::new(file).finish()?;
        let id = df.column("id")?.u64()?.clone();
        let group = df.column("group_id")?.u64()?.clone();
        let country = df.column("country")?.str()?.clone();
        let language = df.column("language")?.str()?.clone();
        let text = df.column("text")?.str()?.clone();
        let labels = df.column("span_label")?.list()?.clone();
        let starts = df.column("span_start")?.list()?.clone();
        let ends = df.column("span_end")?.list()?.clone();
        let augmented = df.column("augmented")?.bool()?.clone();
        let mut out = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            let row_text = text.get(i).context("text")?.to_string();
            let l = labels.get_as_series(i).context("span_label")?;
            let s = starts.get_as_series(i).context("span_start")?;
            let e = ends.get_as_series(i).context("span_end")?;
            let spans = l
                .u8()?
                .into_no_null_iter()
                .zip(s.u32()?.into_no_null_iter())
                .zip(e.u32()?.into_no_null_iter())
                .map(|((l, s), e)| {
                    let label = AddressLabel::ALL
                        .get(usize::from(l))
                        .copied()
                        .filter(|l| *l != AddressLabel::Unknown)
                        .with_context(|| format!("span label id {l} is not a training label"))?;
                    Ok(Span {
                        label,
                        start: s,
                        end: e,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .and_then(|spans| check_spans(&row_text, spans))
                .with_context(|| format!("{} row {i}", path.display()))?;
            out.push(LabelledExample {
                id: id.get(i).context("id")?,
                group_id: group.get(i).context("group_id")?,
                country: country.get(i).context("country")?.to_string(),
                language: language.get(i).context("language")?.to_string(),
                text: row_text,
                spans,
                split,
                augmented: augmented.get(i).context("augmented")?,
            });
        }
        Ok(out)
    }
}

/// Conservative augmentation of training rows. Every function edits pieces and separators
/// and never invents a format nobody writes; `reorder` only swaps pairs a country allows.
pub(crate) mod augment {
    use rand_chacha::ChaCha8Rng;
    use tessera::AddressLabel as L;

    use super::{
        LabelledExample, Pieces, Rng, SeedableRng, decompose, glued, japanese_script,
        japanese_separator, render_pieces,
    };

    type Augmentation = fn(&str, &mut Pieces, &mut ChaCha8Rng) -> bool;

    /// In application order, with the probability each is tried on one copy.
    pub const AUGMENTATIONS: &[(Augmentation, f32)] = &[
        (separators_commas, 0.5),
        (separators_newlines, 0.3),
        (one_line_commas, 0.35),
        (separator_variants, 0.10),
        (insert_postcode_ge, 0.5),
        (po_box_digits, 0.4),
        (insert_banchi_jp, 0.6),
        (digit_width_jp, 0.5),
        (ideographic_comma_jp, 0.15),
        (abbreviate_road, 0.4),
        (casing_upper, 0.15),
        (casing_lower, 0.10),
        (omit_country, 0.5),
        (omit_region, 0.3),
        (official_format_ge, 0.35),
        (floor_room_ge, 0.2),
        (official_layout_jp, 0.4),
        (office_unit_us, 0.2),
        (number_range_de, 0.15),
        (postcode_prefix_de, 0.05),
        (insert_unit_line, 0.15),
        (ocr_substitute, 0.10),
        (unicode_variant, 0.10),
        (reorder, 0.20),
    ];

    /// The id of copy `c` of original `id`; ids count down from `u64::MAX` so they never meet
    /// the originals' ids.
    pub fn copy_id(id: u64, copies: usize, c: usize) -> u64 {
        u64::MAX - (id * copies as u64 + c as u64)
    }

    /// Inverse of `copy_id`: the original an augmented row was made from.
    pub fn original_id(copy: u64, copies: usize) -> u64 {
        (u64::MAX - copy) / copies.max(1) as u64
    }

    /// Up to `copies` augmented variants of `row`, each different from the row and each other.
    ///
    /// A copy whose spans would cut a token is dropped and counted in `unencodable`: one bad
    /// copy must not fail a whole sample.
    pub fn augment_row(
        seed: u64,
        row: &LabelledExample,
        copies: usize,
        fc: &tessera::internal::FeatureConfig,
        unencodable: &mut u64,
    ) -> Vec<LabelledExample> {
        let mut out = Vec::new();
        for c in 0..copies {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ row.id ^ ((c as u64) << 48));
            let mut p = decompose(&row.text, &row.spans);
            let mut changed = false;
            for (f, rate) in AUGMENTATIONS {
                if rng.random::<f32>() < *rate {
                    changed |= f(&row.country, &mut p, &mut rng);
                }
            }
            if !changed {
                continue;
            }
            let (text, spans) = render_pieces(&p);
            // Exact text, not the normalized fingerprint: a copy that differs only in its
            // separators or casing is exactly what these augmentations are for.
            if text == row.text || out.iter().any(|o: &LabelledExample| o.text == text) {
                continue;
            }
            if !crate::check::spans_valid(&text, &spans)
                || crate::dataset::encode(&text, &spans, fc).is_err()
            {
                *unencodable += 1;
                continue;
            }
            out.push(LabelledExample {
                text,
                spans,
                augmented: true,
                id: copy_id(row.id, copies, c),
                ..row.clone()
            });
        }
        out
    }

    fn labelled(p: &Pieces, i: usize) -> bool {
        p.items[i].label.is_some()
    }

    /// Pairs never split by a separator augmentation: a house number and its road in every
    /// country, and a postcode with its locality where the country writes them together.
    fn kept_together(country: &str, p: &Pieces, i: usize) -> bool {
        let (a, b) = (p.items[i].label, p.items[i + 1].label);
        matches!(
            (a, b),
            (Some(L::HouseNumber), Some(L::Road)) | (Some(L::Road), Some(L::HouseNumber))
        ) || locality_pair(country, a, b)
    }

    /// Every plain space between two components becomes a comma, decided once for the row so
    /// one line never mixes the two.
    pub fn separators_commas(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        // Japanese is written without commas between its components.
        if country == "JP" && japanese_script(p) {
            return false;
        }
        let mut changed = false;
        for i in 0..p.seps.len() {
            if p.seps[i] == " "
                && labelled(p, i)
                && labelled(p, i + 1)
                && !kept_together(country, p, i)
            {
                p.seps[i] = ", ".into();
                changed = true;
            }
        }
        changed
    }

    /// A postcode and the locality it belongs to, written on one line with a space: in
    /// Germany the postcode leads its whole locality (`45149 Heißen`, `10115 Berlin`); in
    /// Georgia it follows or leads the town; in the US the ZIP follows the state (`IL 62701`).
    /// GB writes `Leeds LS1 4AB`, `Leeds, LS1 4AB`, and the two on separate lines, so it has no
    /// such pair.
    fn locality_pair(country: &str, a: Option<L>, b: Option<L>) -> bool {
        let locality = |l: Option<L>| matches!(l, Some(L::Suburb | L::District | L::City));
        match country {
            "DE" => a == Some(L::Postcode) && locality(b),
            "GE" => {
                (a == Some(L::Postcode) && locality(b))
                    || (a == Some(L::City) && b == Some(L::Postcode))
            }
            "US" => a == Some(L::Region) && b == Some(L::Postcode),
            _ => false,
        }
    }

    /// Whether separator `i` lies between a US city and its state, a county in between or not:
    /// `Atlanta, Fulton County, GA 30314` stays on one line.
    fn within_us_locality(country: &str, p: &Pieces, i: usize) -> bool {
        let at = |l: L| p.items.iter().position(|it| it.label == Some(l));
        country == "US"
            && matches!((at(L::City), at(L::Region)), (Some(c), Some(r)) if c <= i && i < r)
    }

    /// Pairs written on one line but with a comma: a US city and its state, `Springfield, IL`.
    fn same_line(country: &str, a: Option<L>, b: Option<L>) -> bool {
        country == "US" && a == Some(L::City) && b == Some(L::Region)
    }

    fn gb_town_postcode(country: &str, a: Option<L>, b: Option<L>) -> bool {
        country == "GB" && a == Some(L::City) && b == Some(L::Postcode)
    }

    /// The address on one line: field breaks become commas, except that a postcode and its
    /// locality share a segment, as in `12 Elm Rd, Norwich NR2 3HU` or `10115 Berlin`.
    pub fn one_line_commas(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        let mut changed = false;
        for i in 0..p.seps.len() {
            if p.seps[i] == "\n" {
                let (a, b) = (p.items[i].label, p.items[i + 1].label);
                // Japanese is written as one run: no separators, except around a postcode, after
                // the country, and before a venue name.
                let unlabelled = p.items[i].label.is_none() || p.items[i + 1].label.is_none();
                p.seps[i] = if country == "JP" && japanese_script(p) {
                    match japanese_separator(&p.items[i], &p.items[i + 1]) {
                        "\n" => " ",
                        sep => sep,
                    }
                } else if glued(&p.items[i].text, &p.items[i + 1].text) && !unlabelled {
                    ""
                } else if kept_together(country, p, i)
                    || (gb_town_postcode(country, a, b) && rng.random::<f32>() < 0.6)
                {
                    " "
                } else {
                    ", "
                }
                .into();
                changed = true;
            }
        }
        changed
    }

    /// A one-line address with its commas written as a less common mark throughout:
    /// `Tudor Hill · Birmingham · B14 5AA`. Multi-line addresses keep their commas.
    pub fn separator_variants(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        const MARKS: [&str; 4] = [" · ", " | ", "; ", " - "];
        if country == "JP" || p.seps.iter().any(|s| s.contains('\n')) {
            return false;
        }
        let mark = MARKS[rng.random_range(0..MARKS.len())];
        let mut changed = false;
        for i in 0..p.seps.len() {
            if p.seps[i] == ", " && !within_us_locality(country, p, i) {
                p.seps[i] = mark.into();
                changed = true;
            }
        }
        changed
    }

    /// Japanese written with the ideographic comma where the source has a space between
    /// components: `日本、神奈川県横浜市`, `〒220-0012、神奈川県`. The source almost never has it.
    pub fn ideographic_comma_jp(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        if country != "JP" || !japanese_script(p) {
            return false;
        }
        let mut changed = false;
        for i in 0..p.seps.len() {
            if p.seps[i] == " " && labelled(p, i) && labelled(p, i + 1) {
                p.seps[i] = "、".into();
                changed = true;
            }
        }
        changed
    }

    /// Georgian rows rarely carry a postcode, though people write one after or before the
    /// town. Only towns with known codes get one, and only one of their own codes.
    pub fn insert_postcode_ge(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        const TOWNS: [(&str, &str, &[&str]); 7] = [
            (
                "tbilisi",
                "თბილისი",
                &[
                    "0102", "0105", "0108", "0112", "0131", "0144", "0160", "0177", "0186", "0194",
                ],
            ),
            ("kutaisi", "ქუთაისი", &["4600"]),
            ("batumi", "ბათუმი", &["6000", "6010"]),
            ("rustavi", "რუსთავი", &["3700"]),
            ("zugdidi", "ზუგდიდი", &["2100"]),
            ("gori", "გორი", &["1400"]),
            ("telavi", "თელავი", &["2200"]),
        ];
        if country != "GE" || p.items.iter().any(|i| i.label == Some(L::Postcode)) {
            return false;
        }
        let Some(i) = p.items.iter().position(|it| it.label == Some(L::City)) else {
            return false;
        };
        let town = p.items[i].text.to_lowercase();
        let Some((_, _, codes)) = TOWNS
            .iter()
            .find(|(latin, native, _)| town == *latin || town == *native)
        else {
            return false;
        };
        let piece = super::Piece {
            label: Some(L::Postcode),
            text: codes[rng.random_range(0..codes.len())].to_string(),
        };
        // One-line Georgian rows in the source put the code first; after the town is the
        // less common but real order.
        if rng.random::<f32>() < 0.4 {
            p.items.insert(i + 1, piece);
        } else {
            p.items.insert(i, piece);
        }
        p.seps.insert(i, " ".into());
        true
    }

    /// A lot number after the block of a Japanese address that has none: `梅田3丁目1-1`,
    /// `栄3丁目5-12`, `12番4号`. The source almost never carries one; real addresses nearly
    /// always do.
    pub fn insert_banchi_jp(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        let has = |l: L| p.items.iter().any(|i| i.label == Some(l));
        if country != "JP" || has(L::HouseNumber) || has(L::PoBox) || !japanese_script(p) {
            return false;
        }
        // A route name (`宇治淀線`, `国道8号`) takes no lot number, and a road already ending
        // in one (`字伊原間26-9`) has it.
        let route = |t: &str| {
            ["線", "県道", "国道", "府道", "都道", "道道"]
                .iter()
                .any(|w| t.contains(w))
        };
        if p.items
            .iter()
            .any(|i| i.label == Some(L::Road) && route(&i.text))
        {
            return false;
        }
        let Some(at) = p.items.iter().rposition(|i| {
            matches!(i.label, Some(L::Suburb | L::Road)) && i.text.chars().any(super::is_cjk)
        }) else {
            return false;
        };
        let ends_in_number = p.items[at].text.chars().next_back().is_some_and(|c| {
            c.is_ascii_digit() || ('０'..='９').contains(&c) || "番号地".contains(c)
        });
        if ends_in_number || p.items[at].text.chars().any(|c| c.is_ascii_alphabetic()) {
            return false;
        }
        let chome = p.items.iter().any(|i| i.text.contains("丁目"));
        let (a, b, c) = (
            rng.random_range(1..=30),
            rng.random_range(1..=40),
            rng.random_range(1..=20),
        );
        // After `丁目` the block is already written, so only the lot and building remain.
        let text = match (chome, rng.random_range(0..4)) {
            (true, 0 | 1) => format!("{a}-{b}"),
            (true, _) => format!("{a}番{b}号"),
            (false, 0) => format!("{a}-{b}"),
            (false, 1) => format!("{a}-{b}-{c}"),
            (false, 2) => format!("{a}番{b}号"),
            (false, _) => match rng.random_range(0..3) {
                0 => format!("{a}番地{b}"),
                1 => format!("{a}番地の{b}"),
                _ => format!("{a}番地"),
            },
        };
        // Glued to a block ending in a Japanese character; after a digit or a romanized word,
        // gluing would merge the two into one token.
        let sep = match p.items[at].text.chars().next_back() {
            Some(c) if super::is_cjk(c) && !('０'..='９').contains(&c) => "",
            _ => " ",
        };
        p.items.insert(
            at + 1,
            super::Piece {
                label: Some(L::HouseNumber),
                text,
            },
        );
        p.seps.insert(at, sep.to_string());
        true
    }

    /// A German Postfach number written as it is printed, often in pairs: `Postfach 10 02 34`.
    pub fn po_box_digits(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        if country != "DE" {
            return false;
        }
        let Some(item) = p.items.iter_mut().find(|i| i.label == Some(L::PoBox)) else {
            return false;
        };
        let digits: String = (0..rng.random_range(4..=6))
            .map(|k| {
                char::from(
                    b'0' + if k == 0 {
                        rng.random_range(1..10)
                    } else {
                        rng.random_range(0..10)
                    },
                )
            })
            .collect();
        let number = if digits.len() == 6 && rng.random::<f32>() < 0.6 {
            format!("{} {} {}", &digits[0..2], &digits[2..4], &digits[4..6])
        } else {
            digits
        };
        let word = if rng.random::<f32>() < 0.8 {
            "Postfach"
        } else {
            "Pf."
        };
        item.text = format!("{word} {number}");
        true
    }

    /// One or two separators become line breaks, never inside a pair kept together.
    pub fn separators_newlines(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        // A Japanese address runs on; its only break is before a venue name.
        if country == "JP" && japanese_script(p) {
            return false;
        }
        let candidates: Vec<usize> = (0..p.seps.len())
            .filter(|&i| p.seps[i] != "\n" && !kept_together(country, p, i))
            .filter(|&i| !same_line(country, p.items[i].label, p.items[i + 1].label))
            .filter(|&i| !within_us_locality(country, p, i))
            .collect();
        if candidates.is_empty() {
            return false;
        }
        let n = rng.random_range(1..=2.min(candidates.len()));
        for _ in 0..n {
            let i = candidates[rng.random_range(0..candidates.len())];
            p.seps[i] = "\n".into();
        }
        true
    }

    const DIRECTIONS: [&str; 8] = [
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    ];

    /// Road-type words and directions shortened with the country's abbreviation table.
    pub fn abbreviate_road(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        let table = super::abbrev::table(country);
        let mut changed = false;
        for item in p.items.iter_mut().filter(|i| i.label == Some(L::Road)) {
            let all: Vec<&str> = item.text.split(' ').collect();
            let direction = |w: &str| {
                DIRECTIONS
                    .iter()
                    .any(|d| d.eq_ignore_ascii_case(w.trim_end_matches('.')))
            };
            // Only the road type and directions are shortened: the road type is the last word,
            // or the one before a trailing direction, and never the name in `The Drive` or the
            // first word of `Avenue Road`; a direction leads or trails a longer name
            // (`North Main Street`, `Main Street North`).
            let shortened = |k: usize| {
                let n = all.len();
                if direction(all[k]) {
                    n > 2 && (k == 0 || k + 1 == n)
                } else {
                    let last = k + 1 == n || (k + 2 == n && direction(all[n - 1]));
                    last && k > 0 && !all[k - 1].eq_ignore_ascii_case("the")
                }
            };
            let words: Vec<String> = all
                .iter()
                .enumerate()
                .map(|(k, &w)| {
                    let lower = w.to_lowercase();
                    let entry = table
                        .iter()
                        .find(|(full, _)| *full == lower)
                        .filter(|_| shortened(k));
                    match entry {
                        Some((_, short)) => {
                            changed = true;
                            if w.len() > 1 && w.chars().all(|c| !c.is_lowercase()) {
                                short.to_uppercase()
                            } else if w.chars().next().is_some_and(char::is_uppercase) {
                                let mut c = short.chars();
                                c.next()
                                    .map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
                            } else {
                                short.to_string()
                            }
                        }
                        // German writes the road type inside the word: `Mühlenstraße` is
                        // shortened to `Mühlenstr.`.
                        None if country == "DE" => {
                            // Cut by characters: `HAUPTSTRAẞE` lowercases to a string of
                            // another byte length.
                            let keep = |suffix: &str| {
                                let n = w.chars().count().checked_sub(suffix.chars().count())?;
                                (n > 0 && lower.ends_with(suffix))
                                    .then(|| w.char_indices().nth(n).map(|(b, _)| b))
                                    .flatten()
                            };
                            match ["straße", "strasse"].iter().find_map(|suffix| keep(suffix)) {
                                Some(cut) => {
                                    changed = true;
                                    format!("{}str.", &w[..cut])
                                }
                                None => w.to_string(),
                            }
                        }
                        None => w.to_string(),
                    }
                })
                .collect();
            item.text = words.join(" ");
        }
        changed
    }

    /// Not for Georgian: Mkhedruli upper-cases to Mtavruli, which addresses are not written in.
    pub fn casing_upper(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        country != "GE" && recase(p, |s| s.to_uppercase())
    }

    /// The whole address in lower case, except in Georgia.
    pub fn casing_lower(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        country != "GE" && recase(p, |s| s.to_lowercase())
    }

    /// Only when the byte length is unchanged; a recasing that grows a letter into several
    /// (`İ` to `i̇`) is skipped rather than written.
    fn recase(p: &mut Pieces, f: impl Fn(&str) -> String) -> bool {
        let mut changed = false;
        for item in &mut p.items {
            let next = f(&item.text);
            if next != item.text && next.len() == item.text.len() {
                item.text = next;
                changed = true;
            }
        }
        changed
    }

    /// Drops the country line, as most addresses written for a local reader do.
    pub fn omit_country(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        omit(country, p, L::Country)
    }

    /// Drops the region, except in Georgia, where it is usually written, and in the US when a
    /// ZIP follows it, which is never written without its state.
    pub fn omit_region(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        let us_zip = country == "US" && p.items.iter().any(|i| i.label == Some(L::Postcode));
        country != "GE" && !us_zip && omit(country, p, L::Region)
    }

    /// Removes the first `label` piece with one of its separators. The two pieces that become
    /// neighbours get a space when they are a pair that is written together.
    fn omit(country: &str, p: &mut Pieces, label: L) -> bool {
        let labelled = p.items.iter().filter(|i| i.label.is_some()).count();
        let Some(i) = p.items.iter().position(|it| it.label == Some(label)) else {
            return false;
        };
        if labelled < 3 {
            return false;
        }
        p.items.remove(i);
        if i < p.seps.len() {
            p.seps.remove(i);
        } else if i > 0 {
            p.seps.remove(i - 1);
        }
        if i > 0 && i < p.items.len() {
            let (a, b) = (&p.items[i - 1], &p.items[i]);
            // The kept separator was chosen for other neighbours; `""` between pieces that do
            // not run on would merge them into one token.
            let sep = if kept_together(country, p, i - 1) {
                " "
            } else if country == "JP" && p.seps[i - 1] != "\n" {
                match japanese_separator(a, b) {
                    "\n" => " ",
                    sep => sep,
                }
            } else if p.seps[i - 1].is_empty() && !glued(&a.text, &b.text) {
                " "
            } else {
                return true;
            };
            p.seps[i - 1] = sep.into();
        }
        true
    }

    /// A unit or floor added to a street address that has neither, placed where the country
    /// writes it: in GB on its own line or before the street (`Flat 12, 10 Downing Street`),
    /// in Germany after the house number (`Parkstraße 8, 3. OG`), in the US after the street
    /// on the same line (`123 Main St Apt 4B`), in Georgia after the house number in the
    /// street's own script (`ქ. 12, ბინა 5`).
    pub fn insert_unit_line(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        let has = |l: L| p.items.iter().any(|i| i.label == Some(l));
        if has(L::Unit) || has(L::Level) || has(L::PoBox) || !has(L::HouseNumber) {
            return false;
        }
        let street = |i: &super::Piece| matches!(i.label, Some(L::Road | L::HouseNumber));
        let (Some(first), Some(last)) = (
            p.items.iter().position(street),
            p.items.iter().rposition(street),
        ) else {
            return false;
        };
        let n = rng.random_range(1..=40);
        let floor = rng.random_range(1..=9);
        let letter = char::from(b'A' + rng.random_range(0..4u8));
        let pick = |options: &[(L, String)], rng: &mut ChaCha8Rng| {
            options[rng.random_range(0..options.len())].clone()
        };
        let road_text = p
            .items
            .iter()
            .find(|i| i.label == Some(L::Road))
            .map_or("", |i| i.text.as_str());
        let (label, text, at, before, after) = match country {
            "GB" => {
                let (l, t) = pick(
                    &[
                        (L::Unit, format!("Flat {n}")),
                        (L::Unit, format!("Flat {n}{letter}")),
                        (L::Unit, format!("Apartment {n}")),
                        (L::Unit, format!("Unit {n}")),
                        (L::Level, format!("{} Floor", ordinal(floor))),
                        (L::Level, "Ground Floor".to_string()),
                    ],
                    rng,
                );
                let sep = if rng.random::<f32>() < 0.5 {
                    "\n"
                } else {
                    ", "
                };
                (l, t, first, None, Some(sep))
            }
            "DE" => {
                let (l, t) = pick(
                    &[
                        (L::Unit, format!("Wohnung {n}")),
                        (L::Unit, format!("App. {n}")),
                        (L::Level, format!("{floor}. OG")),
                        (L::Level, format!("{floor}. Etage")),
                        (L::Level, "EG".to_string()),
                    ],
                    rng,
                );
                let sep = if rng.random::<f32>() < 0.5 {
                    "\n"
                } else {
                    ", "
                };
                (l, t, last + 1, Some(sep), None)
            }
            "US" => {
                let (l, t) = pick(
                    &[
                        (L::Unit, format!("Apt {n}{letter}")),
                        (L::Unit, format!("Apt {n}")),
                        (L::Unit, format!("Suite {}", n * 10)),
                        (L::Unit, format!("Unit {n}")),
                        (L::Unit, format!("#{}", n * 10 + floor)),
                        (L::Level, format!("Floor {floor}")),
                    ],
                    rng,
                );
                let sep = if t.starts_with("Suite") { ", " } else { " " };
                (l, t, last + 1, Some(sep), None)
            }
            "GE" => {
                let word = if road_text.is_ascii() {
                    "Apt"
                } else if road_text
                    .chars()
                    .any(|c| ('\u{0400}'..='\u{04FF}').contains(&c))
                {
                    "кв."
                } else {
                    "ბინა"
                };
                (L::Unit, format!("{word} {n}"), last + 1, Some(", "), None)
            }
            _ => return false,
        };
        p.items.insert(
            at,
            super::Piece {
                label: Some(label),
                text,
            },
        );
        match (before, after) {
            (Some(sep), _) => p.seps.insert(at - 1, sep.to_string()),
            (_, Some(sep)) => p.seps.insert(at, sep.to_string()),
            _ => {}
        }
        true
    }

    fn piece(label: Option<L>, text: impl Into<String>) -> super::Piece {
        super::Piece {
            label,
            text: text.into(),
        }
    }

    fn position(p: &Pieces, label: L) -> Option<usize> {
        p.items.iter().position(|i| i.label == Some(label))
    }

    fn georgian(text: &str) -> bool {
        text.chars().any(|c| ('\u{10D0}'..='\u{10FF}').contains(&c))
    }

    /// The layout Georgian public bodies print: the town first, in Georgian often after `ქ.`
    /// (city), `დაბა` (small town) or `სოფ.` (village), then the street and its number written
    /// `№71`, `№ 71`, `N12`, `N 12` or `#5`, then any floor and room: `ქ. ქუთაისი, წერეთლის
    /// ქ. №15`. Everything else (country, postcode, region, district, suburb, venue) is left
    /// off. A glued mark is part of the number; a spaced one is not.
    pub fn official_format_ge(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        if country != "GE" {
            return false;
        }
        let (Some(city), Some(road), Some(house)) = (
            position(p, L::City),
            position(p, L::Road),
            position(p, L::HouseNumber),
        ) else {
            return false;
        };
        let number = p.items[house].text.clone();
        if !number.starts_with(|c: char| c.is_ascii_digit()) {
            return false;
        }
        let native = georgian(&p.items[road].text) && georgian(&p.items[city].text);
        let mut items = Vec::new();
        let mut seps = Vec::new();
        if native && rng.random::<f32>() < 0.5 {
            let prefix = match rng.random_range(0..10) {
                0 => "სოფ.",
                1 => "დაბა",
                _ => "ქ.",
            };
            let glue = if prefix == "ქ." && rng.random::<f32>() < 0.2 {
                ""
            } else {
                " "
            };
            items.push(piece(None, prefix));
            seps.push(glue.to_string());
        }
        items.push(p.items[city].clone());
        seps.push(", ".into());
        // Village and suburb streets are often numbered, not named: `1-ლი ქ.`, `მე-3 ქ.`,
        // `30-ე ქ.`, `მე-17 ქუჩის I შესახვევი`.
        let mut road_piece = p.items[road].clone();
        if native && rng.random::<f32>() < 0.25 {
            let n = rng.random_range(1..=40);
            let kind = ["ქ.", "ქუჩა", "შესახვევი", "ჩიხი"][rng.random_range(0..4)];
            road_piece.text = match (n, rng.random_range(0..3)) {
                (1, _) => format!("1-ლი {kind}"),
                (_, 0) => format!("{n}-ე {kind}"),
                (_, 1) => format!("მე-{n} ქუჩის I შესახვევი"),
                _ => format!("მე-{n} {kind}"),
            };
        }
        items.push(road_piece);
        seps.push(if rng.random::<f32>() < 0.2 { ", " } else { " " }.into());
        match rng.random_range(0..20) {
            0..=6 => items.push(piece(Some(L::HouseNumber), format!("№{number}"))),
            7..=9 => {
                items.push(piece(None, "№"));
                seps.push(" ".into());
                items.push(piece(Some(L::HouseNumber), number));
            }
            10..=11 => items.push(piece(Some(L::HouseNumber), format!("N{number}"))),
            12..=13 => {
                items.push(piece(None, "N"));
                seps.push(" ".into());
                items.push(piece(Some(L::HouseNumber), number));
            }
            14 => items.push(piece(Some(L::HouseNumber), format!("#{number}"))),
            _ => items.push(piece(Some(L::HouseNumber), number)),
        }
        for i in &p.items {
            if matches!(i.label, Some(L::Level | L::Unit)) {
                seps.push(", ".into());
                items.push(i.clone());
            }
        }
        p.items = items;
        p.seps = seps;
        true
    }

    /// A floor and a room after a Georgian street address, as offices print them:
    /// `მე-3 სართული`, `III სართული`, `ოთახი №309`, `ოფისი 12`.
    pub fn floor_room_ge(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        let has = |l: L| p.items.iter().any(|i| i.label == Some(l));
        if country != "GE" || has(L::Level) || has(L::Unit) || !has(L::HouseNumber) {
            return false;
        }
        let Some(last) = p
            .items
            .iter()
            .rposition(|i| i.label == Some(L::HouseNumber))
        else {
            return false;
        };
        if !p
            .items
            .iter()
            .any(|i| i.label == Some(L::Road) && georgian(&i.text))
        {
            return false;
        }
        const ROMAN: [&str; 9] = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
        let n = rng.random_range(1..=9usize);
        let floor = match (n, rng.random_range(0..4)) {
            (1, 0) => "პირველი სართული".to_string(),
            (1, 1) => "1-ლი სართული".to_string(),
            (_, 0 | 1) => format!("მე-{n} სართული"),
            (_, 2) => format!("{} სართული", ROMAN[n - 1]),
            _ => format!("სართ. {n}"),
        };
        let room = rng.random_range(1..=40) + 100 * n;
        let room = match rng.random_range(0..3) {
            0 => format!("ოთახი №{room}"),
            1 => format!("ოფისი {room}"),
            _ => format!("ოთ. {room}"),
        };
        let mut added = vec![];
        // A campus or estate block, and the institution the office is in, are not address
        // components: `თსუ, III კორპუსი, ოთახი №206`.
        if rng.random::<f32>() < 0.4 {
            if rng.random::<f32>() < 0.5 {
                let place = ["თსუ", "სტუ", "თსსუ", "ბიზნესცენტრი", "სავაჭრო ცენტრი"];
                added.push(piece(None, place[rng.random_range(0..place.len())]));
            }
            let block = match rng.random_range(0..3) {
                0 => format!("{} კორპუსი", ROMAN[rng.random_range(0..ROMAN.len())]),
                1 => format!("კორპუსი {}", rng.random_range(1..=12)),
                _ => format!("მე-{} კორპუსი", rng.random_range(2..=12)),
            };
            added.push(piece(None, block));
        }
        match rng.random_range(0..3) {
            0 => added.push(piece(Some(L::Level), floor)),
            1 => added.push(piece(Some(L::Unit), room)),
            _ => {
                added.push(piece(Some(L::Level), floor));
                added.push(piece(Some(L::Unit), room));
            }
        }
        for (k, item) in added.into_iter().enumerate() {
            p.items.insert(last + 1 + k, item);
            p.seps.insert(last + k, ", ".into());
        }
        true
    }

    /// The layout of Japanese public offices' addresses: the postcode without `〒`, often on a
    /// line of its own, no country, and at times the building with its floor after the lot
    /// number (`横浜第２合同庁舎3階`, `日進ビル1・2階`, `センタービル10F`).
    pub fn official_layout_jp(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        if country != "JP" || !japanese_script(p) {
            return false;
        }
        let mut changed = false;
        if let Some(c) = position(p, L::Country)
            && c < p.seps.len()
        {
            p.items.remove(c);
            p.seps.remove(c);
            changed = true;
        }
        if let Some(k) = position(p, L::Postcode) {
            let text = p.items[k].text.trim_start_matches('〒').to_string();
            if text != p.items[k].text {
                p.items[k].text = text;
                changed = true;
            }
            if rng.random::<f32>() < 0.2 {
                let dash = if rng.random::<f32>() < 0.5 {
                    "−"
                } else {
                    "－"
                };
                p.items[k].text = p.items[k].text.replace('-', dash);
            }
            if k < p.seps.len() && rng.random::<f32>() < 0.5 {
                p.seps[k] = "\n".into();
                changed = true;
            }
        }
        let has = |l: L| p.items.iter().any(|i| i.label == Some(l));
        let Some(lot) = position(p, L::HouseNumber) else {
            return changed;
        };
        if has(L::Level) || has(L::Unit) || rng.random::<f32>() < 0.5 {
            return changed;
        }
        let place: String = p
            .items
            .iter()
            .find(|i| i.label == Some(L::City))
            .map(|i| {
                i.text
                    .trim_end_matches(['市', '町', '村', '区'])
                    .to_string()
            })
            .filter(|t| !t.is_empty() && t.chars().all(super::is_cjk))
            .unwrap_or_else(|| "中央".to_string());
        let n = rng.random_range(1..=12);
        let building = match rng.random_range(0..5) {
            0 => format!("{place}合同庁舎"),
            1 => format!("{place}第2合同庁舎"),
            2 => format!("中央合同庁舎{}号館", rng.random_range(1..=8)),
            3 => format!("{place}センタービル"),
            _ => format!("{place}ビル"),
        };
        let level = match rng.random_range(0..4) {
            0 => format!("{n}F"),
            1 => format!("{n}・{}階", n + 1),
            _ => format!("{n}階"),
        };
        let sep = match rng.random_range(0..3) {
            0 => "\n",
            1 => "\u{3000}",
            _ => " ",
        };
        p.items.insert(lot + 1, piece(None, building));
        p.seps.insert(lot, sep.into());
        if rng.random::<f32>() < 0.8 {
            p.items.insert(lot + 2, piece(Some(L::Level), level));
            p.seps.insert(lot + 1, String::new());
        }
        true
    }

    /// A US federal office's room, suite, or mail stop, before the street or after it:
    /// `Room 6D-033`, `Suite CC-5610`, `Mail Stop H21-8`, `Mail Code 28221T`.
    pub fn office_unit_us(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        let has = |l: L| p.items.iter().any(|i| i.label == Some(l));
        if country != "US" || has(L::Unit) || has(L::Level) || has(L::PoBox) {
            return false;
        }
        let street = |i: &super::Piece| matches!(i.label, Some(L::Road | L::HouseNumber));
        let (Some(first), Some(last)) = (
            p.items.iter().position(street),
            p.items.iter().rposition(street),
        ) else {
            return false;
        };
        let (a, b, c) = (
            rng.random_range(1..=12),
            rng.random_range(10..=999),
            rng.random_range(1000..=9999),
        );
        let (x, y) = (
            char::from(b'A' + rng.random_range(0..26u8)),
            char::from(b'A' + rng.random_range(0..26u8)),
        );
        let text = match rng.random_range(0..9) {
            0 => format!("Room {a}{x}-{b:03}"),
            1 => format!("Room {x}{a}-{b}"),
            2 => format!("Suite {x}{y}-{c}"),
            3 => format!("Rm. {c}"),
            4 => format!("Mail Stop {x}{a}-{}", a + 3),
            5 => format!("MS {x}{a}-{a}"),
            6 => format!("Mailstop {a}{x}"),
            7 => format!("Mail Code {}{x}", c * 10 + a),
            _ => format!("Room {x}{a}-{b}-{:02}", a + 1),
        };
        if rng.random::<f32>() < 0.5 {
            p.items.insert(first, piece(Some(L::Unit), text));
            p.seps.insert(first, ", ".into());
        } else {
            p.items.insert(last + 1, piece(Some(L::Unit), text));
            p.seps.insert(last, ", ".into());
        }
        true
    }

    /// A German house number as a range of numbers: `52–54`, `2 - 4`, `16/18`.
    pub fn number_range_de(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        if country != "DE" {
            return false;
        }
        let Some(k) = position(p, L::HouseNumber) else {
            return false;
        };
        let Ok(n) = p.items[k].text.parse::<u32>() else {
            return false;
        };
        let dash = ["–", "-", " - ", " – ", "/"][rng.random_range(0..5)];
        let step = if dash == "/" {
            2
        } else {
            rng.random_range(1..=6)
        };
        p.items[k].text = format!("{n}{dash}{}", n + step);
        true
    }

    /// The old country prefix before a German postcode, printed outside the code: `D-70565`.
    pub fn postcode_prefix_de(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        if country != "DE" {
            return false;
        }
        let Some(k) = position(p, L::Postcode) else {
            return false;
        };
        let before = if k > 0 {
            format!("{}{}", p.items[k - 1].text, p.seps[k - 1])
        } else {
            String::new()
        };
        if !p.items[k].text.starts_with(|c: char| c.is_ascii_digit()) || before.ends_with('-') {
            return false;
        }
        p.items.insert(k, piece(None, "D-"));
        p.seps.insert(k, String::new());
        true
    }

    fn ordinal(n: u32) -> String {
        let suffix = match (n % 10, n % 100) {
            (1, x) if x != 11 => "st",
            (2, x) if x != 12 => "nd",
            (3, x) if x != 13 => "rd",
            _ => "th",
        };
        format!("{n}{suffix}")
    }

    /// One character in one ASCII component, swapped for a common OCR confusion.
    pub fn ocr_substitute(country: &str, p: &mut Pieces, rng: &mut ChaCha8Rng) -> bool {
        const SWAPS: [(char, char); 7] = [
            ('0', 'O'),
            ('O', '0'),
            ('1', 'l'),
            ('l', '1'),
            ('5', 'S'),
            ('S', '5'),
            ('8', 'B'),
        ];
        if country == "GE" {
            return false;
        }
        // Postcodes and house numbers keep their exact form: a misread one is no longer valid.
        let candidates: Vec<usize> = (0..p.items.len())
            .filter(|&i| {
                matches!(p.items[i].label, Some(l) if !matches!(l, L::Postcode | L::HouseNumber))
                    && p.items[i].text.is_ascii()
            })
            .collect();
        if candidates.is_empty() {
            return false;
        }
        let i = candidates[rng.random_range(0..candidates.len())];
        let text = &p.items[i].text;
        let positions: Vec<(usize, char)> = text
            .char_indices()
            .filter_map(|(k, c)| {
                SWAPS
                    .iter()
                    .find(|(from, _)| *from == c)
                    .map(|(_, to)| (k, *to))
            })
            .collect();
        if positions.is_empty() {
            return false;
        }
        let (k, to) = positions[rng.random_range(0..positions.len())];
        let mut next = text.clone();
        next.replace_range(k..k + 1, &to.to_string());
        p.items[i].text = next;
        true
    }

    /// Japanese digits in the other width: the source writes blocks fullwidth (`１丁目`) and
    /// lots in ASCII, while typed text uses either. A row with fullwidth digits in its
    /// components gets them in ASCII, hyphens between digits too; a Japanese-script row
    /// without gets its block fullwidth. Venue text is left as written.
    pub fn digit_width_jp(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        const OFFSET: u32 = '０' as u32 - '0' as u32;
        if country != "JP" {
            return false;
        }
        let labelled = |i: &super::Piece| i.label.is_some();
        let fullwidth = p
            .items
            .iter()
            .filter(|i| labelled(i))
            .any(|i| i.text.chars().any(|c| ('０'..='９').contains(&c)));
        if !fullwidth && !japanese_script(p) {
            return false;
        }
        let mut changed = false;
        for item in p.items.iter_mut().filter(|i| labelled(i)) {
            let to_ascii = fullwidth;
            if !to_ascii && item.label != Some(L::Suburb) {
                continue;
            }
            let chars: Vec<char> = item.text.chars().collect();
            let digit = |k: usize| {
                chars
                    .get(k)
                    .is_some_and(|c| c.is_ascii_digit() || ('０'..='９').contains(c))
            };
            let text: String = chars
                .iter()
                .enumerate()
                .map(|(k, &c)| match c {
                    '０'..='９' if to_ascii => char::from_u32(c as u32 - OFFSET).unwrap_or(c),
                    '0'..='9' if !to_ascii => char::from_u32(c as u32 + OFFSET).unwrap_or(c),
                    '－' if to_ascii && k > 0 && digit(k - 1) && digit(k + 1) => '-',
                    c => c,
                })
                .collect();
            changed |= text != item.text;
            item.text = text;
        }
        changed
    }

    /// One German umlaut written decomposed, as some systems store it.
    pub fn unicode_variant(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        const DECOMPOSED: [(char, &str); 6] = [
            ('ä', "a\u{308}"),
            ('ö', "o\u{308}"),
            ('ü', "u\u{308}"),
            ('Ä', "A\u{308}"),
            ('Ö', "O\u{308}"),
            ('Ü', "U\u{308}"),
        ];
        if country != "DE" {
            return false;
        }
        for item in p.items.iter_mut().filter(|i| i.label.is_some()) {
            if let Some((k, c)) = item
                .text
                .char_indices()
                .find(|(_, c)| DECOMPOSED.iter().any(|(f, _)| f == c))
                && let Some((_, to)) = DECOMPOSED.iter().find(|(f, _)| *f == c)
            {
                item.text.replace_range(k..k + c.len_utf8(), to);
                return true;
            }
        }
        false
    }

    /// Swaps a Latin-script Georgian street and its number, the one pair written in both orders.
    pub fn reorder(country: &str, p: &mut Pieces, _: &mut ChaCha8Rng) -> bool {
        // Georgian addresses written in Latin script put the number first as often as last;
        // in Georgian or Cyrillic script the number follows the street.
        if country != "GE" {
            return false;
        }
        for i in 0..p.seps.len() {
            let (a, b) = (&p.items[i], &p.items[i + 1]);
            let pair = matches!(
                (a.label, b.label),
                (Some(L::Road), Some(L::HouseNumber)) | (Some(L::HouseNumber), Some(L::Road))
            );
            let road = if a.label == Some(L::Road) { a } else { b };
            if pair && road.text.is_ascii() {
                p.items.swap(i, i + 1);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use tessera::internal::FeatureConfig;

    fn map() -> LabelMap {
        LabelMap::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/manifests/label-map.json"),
        )
        .unwrap()
    }

    fn render_line(line: &str) -> Result<(String, Vec<(AddressLabel, String)>), Rejection> {
        let rec = parse_line(line).unwrap();
        let mut dropped = BTreeMap::new();
        let pieces = pieces_from_record(&rec, &map(), &mut dropped).unwrap()?;
        let (text, spans) = render_pieces(&pieces);
        let named = spans
            .iter()
            .map(|s| (s.label, text[s.start as usize..s.end as usize].to_string()))
            .collect();
        Ok((text, named))
    }

    #[test]
    fn parses_lines_and_slash_tokens() {
        let rec = parse_line("en\tgb\t14/house_number Rustaveli/road Avenue/road London/city SW1A/postcode 1AA/postcode\r\n").unwrap();
        assert_eq!(rec.pairs.len(), 6);
        assert_eq!(rec.pairs[5], ("1AA", "postcode"));
        let rec = parse_line("en\tgb\t1/2/house_number").unwrap();
        assert_eq!(rec.pairs[0], ("1/2", "house_number"));
    }

    #[test]
    fn renders_text_a_person_would_write() {
        let (text, spans) = render_line("de\tde\tParkstraße/road 8/house_number |/FSEP 10117/postcode Berlin/city ,/SEP Deutschland/country").unwrap();
        assert_eq!(text, "Parkstraße 8\n10117 Berlin, Deutschland");
        assert_eq!(
            spans,
            vec![
                (AddressLabel::Road, "Parkstraße".into()),
                (AddressLabel::HouseNumber, "8".into()),
                (AddressLabel::Postcode, "10117".into()),
                (AddressLabel::City, "Berlin".into()),
                (AddressLabel::Country, "Deutschland".into()),
            ]
        );
    }

    #[test]
    fn venue_text_is_kept_unlabelled_and_queries_are_dropped() {
        let (text, spans) = render_line("en\tgb\tLidl/house |/FSEP Leeds/city").unwrap();
        assert_eq!(text, "Lidl\nLeeds");
        assert_eq!(spans, vec![(AddressLabel::City, "Leeds".into())]);
        assert_eq!(
            render_line("en\tgb\trestaurants/category near/near Leeds/city"),
            Err(Rejection::Excluded)
        );
        assert_eq!(
            render_line("en\tgb\tLidl/house"),
            Err(Rejection::NoComponent)
        );
    }

    #[test]
    fn brackets_and_punctuation_are_not_padded() {
        let (text, _) = render_line(
            "ka\tge\tლიბერთი/house (/house Liberty/house Bank/house )/house |/FSEP თბილისი/city",
        )
        .unwrap();
        assert_eq!(text, "ლიბერთი (Liberty Bank)\nთბილისი");
    }

    #[test]
    fn country_rules() {
        let rec = |s| {
            let r = parse_line(s).unwrap();
            pieces_from_record(&r, &map(), &mut BTreeMap::new())
                .unwrap()
                .unwrap()
        };
        assert_eq!(
            check_country_rules("GE", &rec("ru\tge\tСухум/city 384900/postcode")),
            Err(Rejection::ForeignPostcode)
        );
        assert_eq!(
            check_country_rules("GE", &rec("ka\tge\tთბილისი/city 0108/postcode")),
            Ok(())
        );
        assert_eq!(
            check_country_rules("DE", &rec("de\tde\tD-60320/postcode Frankfurt/city")),
            Ok(())
        );
        assert_eq!(
            check_country_rules("DE", &rec("de\tde\tDE-67691/postcode Hochspeyer/city")),
            Ok(())
        );
        assert_eq!(
            check_country_rules("DE", &rec("de\tde\tD32051/postcode Herford/city")),
            Err(Rejection::ForeignPostcode)
        );
        assert_eq!(
            check_country_rules(
                "DE",
                &rec("de\tde\tPostfach/po_box 12/po_box |/FSEP Hauptstraße/road")
            ),
            Err(Rejection::PoBoxWithRoad)
        );
    }

    #[test]
    fn decompose_inverts_render() {
        for line in [
            "de\tde\tParkstraße/road 8/house_number |/FSEP 10117/postcode Berlin/city ,/SEP Deutschland/country",
            "en\tgb\tLidl/house |/FSEP Leeds/city",
            "ka\tge\tლიბერთი/house (/house Liberty/house Bank/house )/house |/FSEP თბილისი/city 0108/postcode",
            "en\tgb\tFlat/unit 4/unit ,/SEP 221B/house_number Baker/road Street/road",
        ] {
            let rec = parse_line(line).unwrap();
            let p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
                .unwrap()
                .unwrap();
            let (text, spans) = render_pieces(&p);
            assert_eq!(
                render_pieces(&decompose(&text, &spans)),
                (text, spans),
                "{line}"
            );
        }
    }

    #[test]
    fn normalize_and_group_keys() {
        assert_eq!(normalize("14, Rustaveli Ave.\n"), "14 rustaveli ave");
        let key = |t: &str, road: (u32, u32)| {
            group_key(
                "GB",
                t,
                &[
                    Span {
                        label: AddressLabel::HouseNumber,
                        start: 0,
                        end: 2,
                    },
                    Span {
                        label: AddressLabel::Road,
                        start: road.0,
                        end: road.1,
                    },
                ],
            )
        };
        assert_eq!(key("10 Baker Street", (3, 15)), key("10 Baker St", (3, 11)));
        assert_eq!(abbrev::fold_road("DE", "hauptstr"), "hauptstraße");
        assert_eq!(abbrev::fold_road("DE", "hauptstrasse"), "hauptstraße");
        assert_eq!(abbrev::fold_road("DE", "am markt str"), "am markt straße");
        assert_eq!(abbrev::fold_road("GB", "hauptstr"), "hauptstr");
    }

    #[test]
    fn splits_are_deterministic_and_balanced() {
        let mut n = BTreeMap::new();
        for k in 0..12_000 {
            let key = format!("key {k}");
            assert_eq!(split_for(42, &key), split_for(42, &key));
            *n.entry(split_for(42, &key)).or_insert(0) += 1;
        }
        assert!((900..=1100).contains(&n[&Split::Valid]), "{n:?}");
        assert!((900..=1100).contains(&n[&Split::Test]), "{n:?}");
    }

    #[test]
    fn parquet_round_trip() {
        let rows: Vec<LabelledExample> = (0..5)
            .map(|i| LabelledExample {
                id: i,
                group_id: i * 7,
                country: "GB".into(),
                language: "en".into(),
                text: format!("{i} Baker Street"),
                spans: if i == 3 {
                    vec![]
                } else {
                    vec![
                        Span {
                            label: AddressLabel::HouseNumber,
                            start: 0,
                            end: 1,
                        },
                        Span {
                            label: AddressLabel::Road,
                            start: 2,
                            end: 14,
                        },
                    ]
                },
                split: Split::Valid,
                augmented: i % 2 == 0,
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("tessera-shard-{}", std::process::id()));
        write_shards(&dir, &rows).unwrap();
        let back = read_shard(&dir.join("valid.parquet"), Split::Valid).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back, rows);
    }

    #[test]
    fn a_shard_with_bad_spans_is_an_error() {
        let row = |spans: Vec<Span>| LabelledExample {
            id: 0,
            group_id: 0,
            country: "DE".into(),
            language: "de".into(),
            text: "Straße 1".into(),
            spans,
            split: Split::Test,
            augmented: false,
        };
        let span = |label, start, end| Span { label, start, end };
        for (name, spans) in [
            ("past the end", vec![span(AddressLabel::Road, 0, 40)]),
            ("inside ß", vec![span(AddressLabel::Road, 0, 5)]),
            (
                "out of order",
                vec![
                    span(AddressLabel::HouseNumber, 8, 9),
                    span(AddressLabel::Road, 0, 7),
                ],
            ),
            ("unknown label", vec![span(AddressLabel::Unknown, 0, 7)]),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "tessera-bad-shard-{}-{}",
                std::process::id(),
                name.replace(' ', "-")
            ));
            write_shards(&dir, &[row(spans)]).unwrap();
            let got = read_shard(&dir.join("test.parquet"), Split::Test);
            std::fs::remove_dir_all(&dir).unwrap();
            let err = format!("{:#}", got.unwrap_err());
            assert!(err.contains("test.parquet row 0"), "{name}: {err}");
        }
    }

    fn example(country: &str, line: &str) -> LabelledExample {
        let rec = parse_line(line).unwrap();
        let p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
            .unwrap()
            .unwrap();
        let (text, spans) = render_pieces(&p);
        LabelledExample {
            id: 1,
            group_id: 1,
            country: country.into(),
            language: "en".into(),
            text,
            spans,
            split: Split::Train,
            augmented: false,
        }
    }

    fn assert_invariant(e: &LabelledExample) {
        let mut last = 0;
        for s in &e.spans {
            assert!(s.start < s.end && s.start >= last, "{e:?}");
            let t = &e.text[s.start as usize..s.end as usize];
            assert_eq!(t, t.trim(), "{e:?}");
            last = s.end;
        }
    }

    #[test]
    fn each_augmentation_keeps_spans_valid() {
        use augment::*;
        let e = example(
            "GB",
            "en\tgb\t10/house_number Downing/road Street/road |/FSEP London/city |/FSEP SW1A/postcode 2AA/postcode |/FSEP United/country Kingdom/country",
        );
        for (i, (f, _)) in AUGMENTATIONS.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
            let mut p = decompose(&e.text, &e.spans);
            f("GB", &mut p, &mut rng);
            let (text, spans) = render_pieces(&p);
            assert_invariant(&LabelledExample {
                text,
                spans,
                ..e.clone()
            });
        }
        let mut p = decompose(&e.text, &e.spans);
        assert!(abbreviate_road(
            "GB",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert!(render_pieces(&p).0.contains("Downing St"));
    }

    #[test]
    fn japanese_town_names_the_source_calls_roads_are_towns() {
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t104-1/house_number |/FSEP 吉/road 成/road 有/road 天/road |/FSEP 応/suburb 神/suburb 町/suburb |/FSEP 徳/city 島/city 市/city |/FSEP 徳/state 島/state 県/state"
            ),
            pairs(&[
                ("region", "徳島県"),
                ("city", "徳島市"),
                ("suburb", "応神町吉成有天"),
                ("house_number", "104-1"),
            ])
        );
        let suburbs = |line: &str| -> Vec<String> {
            labelled("JP", line)
                .into_iter()
                .filter(|(l, _)| l == "suburb")
                .map(|(_, t)| t)
                .collect()
        };
        assert_eq!(
            suburbs(
                "ja\tjp\t1/house_number |/FSEP 銀/road 座/road 4/road |/FSEP 銀/suburb 座/suburb |/FSEP 中/city 央/city 区/city |/FSEP 東/state 京/state 都/state"
            ),
            ["銀座4"]
        );
        assert_eq!(
            suburbs(
                "ja\tjp\t住/road 吉/road 町/road |/FSEP 飯/suburb 寺/suburb 北/suburb 一/suburb 丁/suburb 目/suburb |/FSEP 会/city 津/city 若/city 松/city 市/city"
            ),
            ["飯寺北一丁目", "住吉町"]
        );
        let roads = |line: &str| -> Vec<String> {
            labelled("JP", line)
                .into_iter()
                .filter(|(l, _)| l == "road")
                .map(|(_, t)| t)
                .collect()
        };
        assert_eq!(
            roads("ja\tjp\t国/road 道/road 1/road 号/road |/FSEP 箱/city 根/city 町/city"),
            ["国道1号"]
        );
        assert_eq!(
            labelled(
                "DE",
                "de\tde\tHauptstraße/road 5/house_number |/FSEP D-85049/postcode Ingolstadt/city"
            ),
            pairs(&[
                ("road", "Hauptstraße"),
                ("house_number", "5"),
                ("postcode", "85049"),
                ("city", "Ingolstadt"),
            ])
        );
        assert_eq!(
            suburbs(
                "unk\tjp\t豊/suburb 洲/suburb 豊/suburb 洲/suburb 2/suburb |/FSEP 江/city 東/city 区/city"
            ),
            ["豊洲2"]
        );
    }

    #[test]
    fn official_layouts_print_what_public_bodies_print() {
        use augment::*;
        let spans_of = |p: &Pieces| {
            let (text, spans) = render_pieces(p);
            let parts: Vec<(AddressLabel, String)> = spans
                .iter()
                .map(|s| (s.label, text[s.start as usize..s.end as usize].to_string()))
                .collect();
            (text, spans, parts)
        };
        let ge = example(
            "GE",
            "ka\tge\tწერეთლის/road ქ./road 15/house_number |/FSEP ქუთაისი/city |/FSEP საქართველო/country",
        );
        let mut numbers = std::collections::BTreeSet::new();
        for seed in 0..60 {
            let mut p = decompose(&ge.text, &ge.spans);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            assert!(official_format_ge("GE", &mut p, &mut rng));
            floor_room_ge("GE", &mut p, &mut rng);
            let (text, spans, parts) = spans_of(&p);
            assert!(text.contains("ქუთაისი, "), "{text}");
            assert!(!text.contains("საქართველო"), "{text}");
            assert_invariant(&LabelledExample {
                text,
                spans,
                ..ge.clone()
            });
            for (label, part) in parts {
                if label == AddressLabel::HouseNumber {
                    numbers.insert(part);
                }
            }
        }
        assert!(
            numbers.contains("№15") && numbers.contains("15"),
            "{numbers:?}"
        );

        let jp = example(
            "JP",
            "ja\tjp\t〒100-8926/postcode |/FSEP 東/state 京/state 都/state |/FSEP 千/city 代/city 田/city 区/city |/FSEP 霞/suburb が/suburb 関/suburb 2/suburb 丁/suburb 目/suburb |/FSEP 1-2/house_number",
        );
        let mut p = decompose(&jp.text, &jp.spans);
        assert!(official_layout_jp(
            "JP",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(3)
        ));
        let (text, spans, parts) = spans_of(&p);
        assert!(
            parts
                .iter()
                .any(|(l, t)| *l == AddressLabel::Postcode && !t.contains('〒'))
        );
        assert_invariant(&LabelledExample {
            text,
            spans,
            ..jp.clone()
        });

        let us = example(
            "US",
            "en\tus\t1000/house_number Independence/road Avenue/road SW/road |/FSEP Washington/city DC/state 20585/postcode",
        );
        let mut p = decompose(&us.text, &us.spans);
        assert!(office_unit_us(
            "US",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(1)
        ));
        let (text, spans, parts) = spans_of(&p);
        assert!(
            parts.iter().any(|(l, _)| *l == AddressLabel::Unit),
            "{text}"
        );
        assert_invariant(&LabelledExample {
            text,
            spans,
            ..us.clone()
        });

        let de = example(
            "DE",
            "de\tde\tMarzahner/road Promenade/road 52/house_number |/FSEP 12679/postcode Berlin/city",
        );
        let mut p = decompose(&de.text, &de.spans);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        assert!(number_range_de("DE", &mut p, &mut rng));
        assert!(postcode_prefix_de("DE", &mut p, &mut rng));
        let (text, spans, parts) = spans_of(&p);
        assert!(text.contains("D-12679"), "{text}");
        assert!(parts.contains(&(AddressLabel::Postcode, "12679".to_string())));
        assert!(
            parts.iter().any(|(l, t)| *l == AddressLabel::HouseNumber
                && t.starts_with("52")
                && t.len() > 2)
        );
        assert_invariant(&LabelledExample {
            text,
            spans,
            ..de.clone()
        });
    }

    #[test]
    fn japanese_is_written_without_spaces() {
        let e = example(
            "JP",
            "ja\tjp\t日/country 本/country |/FSEP 静/state 岡/state 県/state |/FSEP 浜/city 松/city 市/city |/FSEP 北/suburb 3/suburb 条/suburb 西/suburb 28/suburb",
        );
        assert_eq!(e.text, "日本\n静岡県\n浜松市\n北3条西28");
        let island = example(
            "JP",
            "ja\tjp\t日/country 本/country |/FSEP 静/state 岡/state 県/state |/FSEP 本/island 州/island |/FSEP 浜/city 松/city 市/city",
        );
        assert_eq!(island.text, "日本\n静岡県\n浜松市");
        let labels: Vec<_> = e
            .spans
            .iter()
            .map(|s| &e.text[s.start as usize..s.end as usize])
            .collect();
        assert_eq!(labels, ["日本", "静岡県", "浜松市", "北3条西28"]);
        let enc = crate::dataset::encode(
            &e.text,
            &e.spans,
            &tessera::internal::FeatureConfig::default(),
        );
        assert!(enc.is_ok(), "a label boundary cut a token: {enc:?}");
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::one_line_commas(
            "JP",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(render_pieces(&p).0, "日本 静岡県浜松市北3条西28");
    }

    #[test]
    fn same_kana_components_keep_a_space() {
        let e = example("GB", "ja\tgb\tロンドン/city イギリス/country");
        assert_eq!(e.text, "ロンドン イギリス");
        assert!(
            crate::dataset::encode(
                &e.text,
                &e.spans,
                &tessera::internal::FeatureConfig::default()
            )
            .is_ok()
        );
    }

    #[test]
    fn us_state_and_zip_stay_together() {
        let e = example(
            "US",
            "en\tus\t7811/house_number Overbrook/road Road/road |/FSEP Towson/city ,/SEP MD/state 21204/postcode",
        );
        assert_eq!(e.text, "7811 Overbrook Road\nTowson, MD 21204");
        for seed in 0..16 {
            let mut p = decompose(&e.text, &e.spans);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            augment::separators_newlines("US", &mut p, &mut rng);
            augment::separator_variants("US", &mut p, &mut rng);
            assert!(
                render_pieces(&p).0.contains("MD 21204"),
                "{:?}",
                render_pieces(&p).0
            );
        }
    }

    #[test]
    fn postcodes_follow_their_country() {
        assert!(is_jp_postcode("330-0802"));
        assert!(is_jp_postcode("〒044-0054"));
        assert!(is_jp_postcode("３３０－０８０２"));
        assert!(!is_jp_postcode("3300802"));
        assert!(!is_jp_postcode("〒〒329-0511"));
        assert!(!is_jp_postcode("10115"));
        let reject = |country: &str, line: &str| {
            let rec = parse_line(line).unwrap();
            let p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
                .unwrap()
                .unwrap();
            check_country_rules(country, &p).is_err()
        };
        assert!(!reject(
            "US",
            "en\tus\tWoburn/city ,/SEP MA/state 01801/postcode"
        ));
        assert!(!reject(
            "US",
            "en\tus\tWoburn/city ,/SEP MA/state 01801-2345/postcode"
        ));
        assert!(reject(
            "US",
            "en\tus\tToronto/city ,/SEP ON/state M5V/postcode 3L9/postcode"
        ));
        assert!(reject("JP", "ja\tjp\t東/city 京/city 10115/postcode"));
    }

    #[test]
    fn units_go_where_each_country_writes_them() {
        let cases = [
            (
                "GB",
                "en\tgb\tLidl/house |/FSEP 10/house_number Downing/road Street/road |/FSEP London/city",
                "Lidl\n",
                "10 Downing Street",
            ),
            (
                "DE",
                "de\tde\tParkstraße/road 8/house_number |/FSEP 10117/postcode Berlin/city",
                "Parkstraße 8",
                "10117 Berlin",
            ),
            (
                "US",
                "en\tus\t123/house_number Main/road St/road |/FSEP Springfield/city ,/SEP IL/state 62701/postcode",
                "123 Main St",
                "Springfield, IL 62701",
            ),
            (
                "GE",
                "ka\tge\tრუსთაველის/road გამზირი/road 14/house_number |/FSEP თბილისი/city",
                "გამზირი 14, ბინა ",
                "თბილისი",
            ),
        ];
        for (country, line, before, after) in cases {
            let e = example(country, line);
            for seed in 0..12 {
                let mut p = decompose(&e.text, &e.spans);
                assert!(augment::insert_unit_line(
                    country,
                    &mut p,
                    &mut ChaCha8Rng::seed_from_u64(seed)
                ));
                let (text, spans) = render_pieces(&p);
                let unit = spans
                    .iter()
                    .find(|s| matches!(s.label, AddressLabel::Unit | AddressLabel::Level))
                    .unwrap();
                assert!(
                    text.contains(before) && text.contains(after),
                    "{country}: {text:?}"
                );
                let (a, b) = (text.find(before).unwrap(), text.find(after).unwrap());
                match country {
                    // Before the street, after the venue line.
                    "GB" => assert!(
                        a < unit.start as usize
                            && (unit.end as usize) < text.find("10 Downing").unwrap(),
                        "{text:?}"
                    ),
                    // After the street, before the locality.
                    _ => assert!(
                        a < unit.start as usize && (unit.end as usize) <= b,
                        "{country}: {text:?}"
                    ),
                }
                assert_invariant(&LabelledExample {
                    text,
                    spans,
                    ..e.clone()
                });
            }
        }
    }

    fn normalized(country: &str, line: &str) -> Result<String, Rejection> {
        let rec = parse_line(line).unwrap();
        let mut p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
            .unwrap()
            .unwrap();
        normalize_source(country, &mut p)?;
        Ok(render_pieces(&p).0)
    }

    #[test]
    fn japanese_rows_are_written_large_to_small() {
        assert_eq!(
            normalized(
                "JP",
                "ja\tjp\t１/suburb 丁/suburb 目/suburb |/FSEP 新/city_district 宿/city_district 区/city_district ,/SEP 東/state 京/state 都/state |/FSEP 日/country 本/country"
            ),
            Ok("日本 東京都新宿区１丁目".to_string())
        );
        assert_eq!(
            normalized(
                "JP",
                "ja\tjp\tローソン/house |/FSEP 滋/state 賀/state 県/state 522-0007/postcode |/FSEP 関/state 東/state 地/state 方/state"
            ),
            Ok("522-0007 滋賀県\nローソン".to_string())
        );
        assert_eq!(
            normalized(
                "JP",
                "ja_rm\tjp\t1-1/house_number Marunouchi/suburb |/FSEP Tokyo/state"
            ),
            Ok("1-1 Marunouchi\nTokyo".to_string())
        );
    }

    fn labelled(country: &str, line: &str) -> Vec<(String, String)> {
        let rec = parse_line(line).unwrap();
        let mut p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
            .unwrap()
            .unwrap();
        normalize_source(country, &mut p).unwrap();
        let (text, spans) = render_pieces(&p);
        spans
            .iter()
            .map(|s| {
                (
                    s.label.as_str().to_string(),
                    text[s.start as usize..s.end as usize].to_string(),
                )
            })
            .collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn japanese_towns_blocks_and_wards_follow_one_convention() {
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t３/suburb 丁/suburb 目/suburb |/FSEP 栄/road |/FSEP 名/city 古/city 屋/city 市/city 中/city 区/city |/FSEP 愛/state 知/state 県/state"
            ),
            pairs(&[
                ("region", "愛知県"),
                ("city", "名古屋市"),
                ("district", "中区"),
                ("suburb", "栄３丁目"),
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t日/country 本/country 横/city 浜/city 市/city 中/city 区/city 中/city_district 区/city_district １/suburb 丁/suburb 目/suburb"
            ),
            pairs(&[
                ("country", "日本"),
                ("city", "横浜市"),
                ("district", "中区"),
                ("suburb", "１丁目"),
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t錦/road 町/road |/FSEP 国/city 立/city 市/city"
            ),
            pairs(&[("city", "国立市"), ("suburb", "錦町")])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja_rm\tjp\t1-1/house_number Maruyama-cho/road |/FSEP Chuo-ku/city |/FSEP Sapporo-shi/city"
            )[1..3],
            pairs(&[("suburb", "Maruyama-cho"), ("district", "Chuo-ku")])[..]
        );
        assert_eq!(
            labelled(
                "JP",
                "ja_rm\tjp\t1-1/house_number Marunouchi/suburb |/FSEP Chiyoda-ku/city_district |/FSEP Tokyo/state"
            )[2],
            ("city".to_string(), "Chiyoda-ku".to_string())
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t日/country 本/country 神/city 奈/city 川/city 県/city 横/city 浜/city 市/city 中/city 区/city 中/city_district 区/city_district １/suburb 丁/suburb 目/suburb"
            ),
            pairs(&[
                ("country", "日本"),
                ("region", "神奈川県"),
                ("city", "横浜市"),
                ("district", "中区"),
                ("suburb", "１丁目"),
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t日/country 本/country 福/city 岡/city 市/city 南/city 区/city Minami/city_district Ku/city_district 井/suburb 尻/suburb 三/suburb 丁/suburb 目/suburb"
            ),
            pairs(&[
                ("country", "日本"),
                ("city", "福岡市"),
                ("district", "南区"),
                ("district", "Minami Ku"),
                ("suburb", "井尻三丁目"),
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t３/suburb 丁/suburb 目/suburb |/FSEP 松/road 戸/road 停/road 車/road 場/road 線/road |/FSEP 松/city 戸/city 市/city"
            ),
            pairs(&[
                ("city", "松戸市"),
                ("suburb", "３丁目"),
                ("road", "松戸停車場線")
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t４/suburb 丁/suburb 目/suburb |/FSEP 栄/road 三/road 丁/road 目/road |/FSEP 名/city 古/city 屋/city 市/city"
            ),
            pairs(&[("city", "名古屋市"), ("suburb", "栄三丁目")])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t飯/road 寺/road 北/road 一/road 丁/road 目/road |/FSEP 飯/suburb 寺/suburb 北/suburb 一/suburb 丁/suburb 目/suburb |/FSEP 会/city 津/city 若/city 松/city 市/city"
            ),
            pairs(&[("city", "会津若松市"), ("suburb", "飯寺北一丁目")])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t東/city_district 区/city_district |/FSEP 福/city 岡/city 市/city 東/city 区/city |/FSEP 福/state 岡/state"
            ),
            pairs(&[
                ("region", "福岡県"),
                ("city", "福岡市"),
                ("district", "東区")
            ])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja_rm\tjp\t1-1/house_number Sakae/suburb |/FSEP Naka/city_district Ku/city_district |/FSEP Aichi/state"
            )[2],
            ("district".to_string(), "Naka Ku".to_string())
        );
        assert_eq!(
            labelled(
                "JP",
                "ja_rm\tjp\t1-1/house_number Marunouchi/road |/FSEP Chiyoda-ku/city_district |/FSEP Tokyo/state"
            )[1],
            ("suburb".to_string(), "Marunouchi".to_string())
        );
        assert_eq!(
            labelled(
                "JP",
                "ja_rm\tjp\t5/house_number Meiji-dori/road |/FSEP Shibuya-ku/city_district |/FSEP Tokyo/state"
            )[1],
            ("road".to_string(), "Meiji-dori".to_string())
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t東/city 京/city 都/city 調/city 布/city 市/city"
            ),
            pairs(&[("region", "東京都"), ("city", "調布市")])
        );
        assert_eq!(
            labelled(
                "JP",
                "ja\tjp\t京/city 都/city 市/city |/FSEP 京/state 都/state 府/state"
            ),
            pairs(&[("region", "京都府"), ("city", "京都市")])
        );
        let rec =
            parse_line("ja\tjp\t浜/city 松/city 市/city 中/city 区/city 和/city 合/city 町/city")
                .unwrap();
        let mut p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
            .unwrap()
            .unwrap();
        assert_eq!(normalize_source("JP", &mut p), Err(Rejection::Mislabelled));
        assert!(!city_holds_more("四日市市") && !city_holds_more("市川市"));
        assert!(!city_holds_more("上市町") && !city_holds_more("余市町"));
        assert!(!is_japanese_street("府中町") && !is_japanese_street("市谷田町"));
        assert!(!is_japanese_street("八日市町") && is_japanese_street("世田谷区太子堂"));
        assert!(is_japanese_street("東京都世田谷区太子堂") && is_japanese_street("職安通り"));
        assert!(!is_japanese_town("Mido-suji") && !is_japanese_town("Sakura Lane"));
        assert!(city_holds_more("横浜市戸塚"));
        assert_eq!(
            labelled("JP", "ja\tjp\t市/city 原/city 区/city")[0],
            ("city".to_string(), "市原区".to_string())
        );
    }

    #[test]
    fn japanese_spaces_become_ideographic_commas() {
        let piece = |label, text: &str| Piece {
            label: Some(label),
            text: text.to_string(),
        };
        let mut p = Pieces {
            items: vec![
                piece(AddressLabel::Country, "日本"),
                piece(AddressLabel::Region, "神奈川県"),
                piece(AddressLabel::City, "横浜市"),
            ],
            seps: vec![" ".into(), String::new()],
        };
        assert!(augment::ideographic_comma_jp(
            "JP",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(render_pieces(&p).0, "日本、神奈川県横浜市");
    }

    #[test]
    fn japanese_digits_switch_width() {
        let rng = &mut ChaCha8Rng::seed_from_u64(0);
        let mut p = decompose(
            &example(
                "JP",
                "ja\tjp\t栄/suburb ３/suburb 丁/suburb 目/suburb ５－１２/house_number",
            )
            .text,
            &example(
                "JP",
                "ja\tjp\t栄/suburb ３/suburb 丁/suburb 目/suburb ５－１２/house_number",
            )
            .spans,
        );
        assert!(augment::digit_width_jp("JP", &mut p, rng));
        assert_eq!(render_pieces(&p).0, "栄3丁目5-12");
        assert!(augment::digit_width_jp("JP", &mut p, rng));
        assert_eq!(render_pieces(&p).0, "栄３丁目5-12");
        assert!(!augment::digit_width_jp("GB", &mut p, rng));
        let e = example(
            "JP",
            "ja_rm\tjp\t2/suburb chome/suburb |/FSEP Chuo-ku/city_district",
        );
        let mut p = decompose(&e.text, &e.spans);
        assert!(!augment::digit_width_jp("JP", &mut p, rng));
    }

    #[test]
    fn japanese_lot_numbers_follow_the_block() {
        let rec = parse_line("ja\tjp\t大/state 阪/state 府/state 大/city 阪/city 市/city 梅/suburb 田/suburb ３/suburb 丁/suburb 目/suburb").unwrap();
        let mut p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
            .unwrap()
            .unwrap();
        normalize_source("JP", &mut p).unwrap();
        assert!(augment::insert_banchi_jp(
            "JP",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        let (text, spans) = render_pieces(&p);
        assert!(text.starts_with("大阪府大阪市梅田３丁目"), "{text}");
        assert_eq!(
            spans.last().map(|s| s.label),
            Some(AddressLabel::HouseNumber)
        );
        assert!(
            crate::dataset::encode(&text, &spans, &tessera::internal::FeatureConfig::default())
                .is_ok()
        );
    }

    #[test]
    fn a_prefecture_repeated_in_the_city_is_stripped() {
        let strip = |region: &str, city: &str| {
            let piece = |label, text: &str| Piece {
                label: Some(label),
                text: text.to_string(),
            };
            let mut p = Pieces {
                items: vec![
                    piece(AddressLabel::Region, region),
                    piece(AddressLabel::City, city),
                ],
                seps: vec![String::new()],
            };
            strip_repeated_prefecture(&mut p);
            (p.items[0].text.clone(), p.items[1].text.clone())
        };
        let own = |a: &str, b: &str| (a.to_string(), b.to_string());
        assert_eq!(strip("愛知", "愛知県江南市"), own("愛知県", "江南市"));
        assert_eq!(strip("北海", "北海道札幌市"), own("北海道", "札幌市"));
        assert_eq!(strip("愛知県", "愛知県江南市"), own("愛知県", "江南市"));
        assert_eq!(strip("京都", "京都市"), own("京都", "京都市"));
        assert_eq!(strip("Toyama", "Toyama Shi"), own("Toyama", "Toyama Shi"));
    }

    #[test]
    fn lettered_roads_are_not_truncations() {
        assert!(normalized("US", "en\tus\t12/house_number Avenue/road J/road").is_ok());
        assert!(normalized("US", "en\tus\t3/house_number County/road Road/road F/road").is_ok());
        assert!(normalized("US", "en\tus\t40/house_number Main/road Street/road N/road").is_ok());
        assert_eq!(
            normalized("GB", "en\tgb\t5/house_number Devereux/road R/road"),
            Err(Rejection::TruncatedRoad)
        );
    }

    #[test]
    fn source_quirks_are_normalized() {
        assert_eq!(
            normalized(
                "GB",
                "en\tgb\t12/house_number Church/road Road/road |/FSEP Leeds/city |/FSEP West/state_district Midlands/state_district |/FSEP United/country Kingdom/country LS1/postcode 4AB/postcode"
            ),
            Ok("12 Church Road\nLeeds\nLS1 4AB\nUnited Kingdom".to_string())
        );
        assert_eq!(
            normalized("GB", "en\tgb\t4/house_number Kew/road Foot/road Roa/road"),
            Err(Rejection::TruncatedRoad)
        );
        assert_eq!(
            normalized(
                "US",
                "en\tus\t9/house_number Elm/road St/road |/FSEP Brooklyn/city_district"
            ),
            Ok("9 Elm St\nBrooklyn".to_string())
        );
        assert_eq!(
            labelled(
                "US",
                "en\tus\t9/house_number Elm/road St/road |/FSEP Brooklyn/city_district |/FSEP New/city York/city City/city"
            )[2..],
            pairs(&[("district", "Brooklyn"), ("city", "New York City")])[..]
        );
        assert_eq!(
            labelled(
                "US",
                "en\tus\t9/house_number Elm/road St/road |/FSEP Brooklyn/city_district"
            )[2],
            ("city".to_string(), "Brooklyn".to_string())
        );
        assert_eq!(
            normalized("GE", "ru\tge\tГагра/city |/FSEP AB/country"),
            Err(Rejection::ForeignCountry)
        );
    }

    #[test]
    fn lot_numbers_only_where_a_japanese_address_takes_one() {
        let pieces = |line: &str| {
            let rec = parse_line(line).unwrap();
            let mut p = pieces_from_record(&rec, &map(), &mut BTreeMap::new())
                .unwrap()
                .unwrap();
            normalize_source("JP", &mut p).unwrap();
            p
        };
        let refused = [
            "ja\tjp\t大/city 阪/city 市/city 大/suburb 戸/suburb 1/suburb",
            "ja_rm\tjp\t東/city 京/city ２/suburb chome/suburb",
            "ja\tjp\t豊/city 川/city 市/city 宇/road 治/road 淀/road 線/road",
        ];
        for line in refused {
            let mut p = pieces(line);
            for seed in 0..8 {
                assert!(
                    !augment::insert_banchi_jp("JP", &mut p, &mut ChaCha8Rng::seed_from_u64(seed)),
                    "{line}"
                );
            }
        }
        for seed in 0..16 {
            let mut p = pieces(
                "ja\tjp\t大/city 阪/city 市/city 梅/suburb 田/suburb ３/suburb 丁/suburb 目/suburb",
            );
            assert!(augment::insert_banchi_jp(
                "JP",
                &mut p,
                &mut ChaCha8Rng::seed_from_u64(seed)
            ));
            let lot = &p.items.last().unwrap().text;
            // After 丁目 only the lot and building remain: never three parts.
            assert!(
                lot.matches('-').count() <= 1 && !lot.ends_with("番地"),
                "{lot}"
            );
        }
    }

    #[test]
    fn omitting_a_piece_never_glues_same_kana_neighbours() {
        let piece = |label, text: &str| Piece {
            label: Some(label),
            text: text.into(),
        };
        let mut p = Pieces {
            items: vec![
                piece(AddressLabel::Country, "ニホン"),
                piece(AddressLabel::Region, "東京都"),
                piece(AddressLabel::City, "ナカノ"),
            ],
            seps: vec![String::new(), String::new()],
        };
        assert!(augment::omit_region(
            "JP",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        let (text, spans) = render_pieces(&p);
        assert_eq!(text, "ニホン ナカノ");
        assert!(crate::dataset::encode(&text, &spans, &FeatureConfig::default()).is_ok());
    }

    #[test]
    fn readings_and_joiners_in_japanese() {
        assert_eq!(
            normalized(
                "JP",
                "ja\tjp\tに/country ほ/country ん/country |/FSEP 佐/state 賀/state |/FSEP さ/city が/city し/city"
            ),
            Err(Rejection::KanaReading)
        );
        let e = example(
            "JP",
            "ja\tjp\tセブン/house -/house イレブン/house |/FSEP 東/city 京/city",
        );
        assert_eq!(e.text, "セブン-イレブン\n東京");
    }

    #[test]
    fn only_road_types_and_directions_are_abbreviated() {
        let short = |country: &str, line: &str| {
            let e = example(country, line);
            let mut p = decompose(&e.text, &e.spans);
            augment::abbreviate_road(country, &mut p, &mut ChaCha8Rng::seed_from_u64(0));
            render_pieces(&p).0
        };
        assert_eq!(
            short(
                "US",
                "en\tus\t12/house_number North/road Main/road Street/road"
            ),
            "12 N Main St"
        );
        assert_eq!(
            short(
                "US",
                "en\tus\t12/house_number Main/road Street/road North/road"
            ),
            "12 Main St N"
        );
        assert_eq!(
            short("GB", "en\tgb\t26/house_number The/road Drive/road"),
            "26 The Drive"
        );
        assert_eq!(
            short("GB", "en\tgb\t3/house_number Avenue/road Road/road"),
            "3 Avenue Rd"
        );
        assert_eq!(
            short("DE", "de\tde\tHAUPTSTRAẞE/road 1/house_number"),
            "HAUPTstr. 1"
        );
    }

    #[test]
    fn capitals_stay_capitals_when_abbreviated() {
        let e = example("US", "en\tus\t12/house_number PINE/road STREET/road");
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::abbreviate_road(
            "US",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(render_pieces(&p).0, "12 PINE ST");
    }

    #[test]
    fn german_compound_roads_are_abbreviated() {
        let e = example("DE", "de\tde\tMühlenstraße/road 8/house_number");
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::abbreviate_road(
            "DE",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(render_pieces(&p).0, "Mühlenstr. 8");
    }

    #[test]
    fn layout_only_copies_are_kept() {
        let e = example(
            "GB",
            "en\tgb\t9/house_number Elm/road Rd/road |/FSEP Norwich/city |/FSEP NR2/postcode 3HU/postcode",
        );
        let mut p = decompose(&e.text, &e.spans);
        augment::one_line_commas("GB", &mut p, &mut ChaCha8Rng::seed_from_u64(0));
        let (text, _) = render_pieces(&p);
        assert_ne!(text, e.text);
        assert_eq!(fingerprint(&text), fingerprint(&e.text));
        let copies: Vec<_> = (0..64)
            .flat_map(|seed| augment::augment_row(seed, &e, 2, &FeatureConfig::default(), &mut 0))
            .collect();
        assert!(
            copies.iter().any(|c| c.text == "9 Elm Rd, Norwich NR2 3HU"),
            "no one-line copy survived"
        );
    }

    #[test]
    fn copy_ids_map_back_to_their_original() {
        let mut seen = std::collections::HashSet::new();
        for (id, copies) in [(0u64, 2usize), (7, 2), (123_456, 3), (40_000_000, 2)] {
            for c in 0..copies {
                let copy = augment::copy_id(id, copies, c);
                assert_eq!(augment::original_id(copy, copies), id);
                assert!(seen.insert((copies, copy)), "copy id {copy} repeats");
            }
        }
    }

    #[test]
    fn unit_line_needs_a_street_address() {
        for line in [
            "de\tde\tPostfach/po_box 1234/po_box |/FSEP 60002/postcode Frankfurt/city",
            "en\tgb\tLondon/city |/FSEP United/country Kingdom/country",
        ] {
            let e = example("DE", line);
            let mut p = decompose(&e.text, &e.spans);
            for seed in 0..8 {
                assert!(!augment::insert_unit_line(
                    "DE",
                    &mut p,
                    &mut ChaCha8Rng::seed_from_u64(seed)
                ));
            }
        }
    }

    #[test]
    fn one_line_keeps_postcode_and_locality_together() {
        let e = example(
            "DE",
            "de\tde\tParkstraße/road 8/house_number |/FSEP 45149/postcode |/FSEP Heißen/suburb |/FSEP Deutschland/country",
        );
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::one_line_commas(
            "DE",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(
            render_pieces(&p).0,
            "Parkstraße 8, 45149 Heißen, Deutschland"
        );
    }

    #[test]
    fn gb_town_and_postcode_take_every_common_layout() {
        let e = example(
            "GB",
            "en\tgb\t9/house_number Elm/road Rd/road |/FSEP Norwich/city |/FSEP NR2/postcode 3HU/postcode",
        );
        let layouts: std::collections::BTreeSet<String> = (0..32)
            .map(|seed| {
                let mut p = decompose(&e.text, &e.spans);
                augment::one_line_commas("GB", &mut p, &mut ChaCha8Rng::seed_from_u64(seed));
                render_pieces(&p).0
            })
            .collect();
        assert!(layouts.contains("9 Elm Rd, Norwich NR2 3HU"), "{layouts:?}");
        assert!(
            layouts.contains("9 Elm Rd, Norwich, NR2 3HU"),
            "{layouts:?}"
        );
    }

    #[test]
    fn omitting_a_piece_rejoins_a_pair_with_a_space() {
        let e = example(
            "GE",
            "ka\tge\t4600/postcode |/FSEP საქართველო/country |/FSEP ქუთაისი/city",
        );
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::omit_country(
            "GE",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(0)
        ));
        assert_eq!(render_pieces(&p).0, "4600 ქუთაისი");
    }

    #[test]
    fn rows_without_a_road_are_keyed_by_their_places_not_their_venue() {
        let key = |t: &str| {
            let e = example("GE", t);
            group_key("GE", &e.text, &e.spans)
        };
        let clinic = key("ka\tge\tკლინიკა/house |/FSEP ბათუმი/city |/FSEP საქართველო/country");
        let school = key("ka\tge\tსკოლა/house |/FSEP ბათუმი/city |/FSEP საქართველო/country");
        let clinic_bare = key("ka\tge\tკლინიკა/house |/FSEP ბათუმი/city");
        let clinic_reversed =
            key("ka\tge\tსაქართველო/country |/FSEP ბათუმი/city |/FSEP კლინიკა/house");
        assert_eq!(clinic, school);
        assert_eq!(clinic, clinic_bare);
        assert_ne!(clinic, key("ka\tge\tკლინიკა/house |/FSEP ქობულეთი/city"));
        assert_eq!(clinic, clinic_reversed);
        assert_eq!(
            key("en\tgb\tStoke-on-Trent/city |/FSEP ST4/postcode 1AA/postcode"),
            key("en\tgb\tStoke/city on/city Trent/city")
        );
    }

    #[test]
    fn georgian_postcode_sits_next_to_the_town() {
        let e = example(
            "GE",
            "ka\tge\tრუსთაველის/road გამზირი/road 14/house_number |/FSEP თბილისი/city",
        );
        for seed in 0..8 {
            let mut p = decompose(&e.text, &e.spans);
            assert!(augment::insert_postcode_ge(
                "GE",
                &mut p,
                &mut ChaCha8Rng::seed_from_u64(seed)
            ));
            let (text, spans) = render_pieces(&p);
            let labels: Vec<_> = spans.iter().map(|s| s.label).collect();
            let city = labels
                .iter()
                .position(|l| *l == AddressLabel::City)
                .unwrap();
            let post = labels
                .iter()
                .position(|l| *l == AddressLabel::Postcode)
                .unwrap();
            assert_eq!(city.abs_diff(post), 1, "{text}");
            assert_invariant(&LabelledExample {
                text,
                spans,
                ..e.clone()
            });
        }
    }

    #[test]
    fn german_postfach_numbers_are_digits() {
        let e = example(
            "DE",
            "de\tde\tPostfach/po_box A/po_box |/FSEP 60002/postcode Frankfurt/city",
        );
        let mut p = decompose(&e.text, &e.spans);
        assert!(augment::po_box_digits(
            "DE",
            &mut p,
            &mut ChaCha8Rng::seed_from_u64(3)
        ));
        let (text, spans) = render_pieces(&p);
        let po = &text[spans[0].start as usize..spans[0].end as usize];
        assert!(
            po.starts_with("Postfach ") || po.starts_with("Pf. "),
            "{po}"
        );
        assert!(
            po.split(' ')
                .skip(1)
                .all(|g| g.bytes().all(|b| b.is_ascii_digit())),
            "{po}"
        );
    }

    #[test]
    fn reorder_only_swaps_latin_georgian_streets() {
        let rng = || ChaCha8Rng::seed_from_u64(0);
        let de = example(
            "DE",
            "de\tde\tParkstraße/road 8/house_number |/FSEP 10117/postcode Berlin/city |/FSEP Deutschland/country",
        );
        assert!(!augment::reorder(
            "DE",
            &mut decompose(&de.text, &de.spans),
            &mut rng()
        ));
        let latin = example("GE", "en\tge\tRustaveli/road Avenue/road 14/house_number");
        let mut p = decompose(&latin.text, &latin.spans);
        assert!(augment::reorder("GE", &mut p, &mut rng()));
        assert_eq!(render_pieces(&p).0, "14 Rustaveli Avenue");
        let georgian = example("GE", "ka\tge\tრუსთაველის/road გამზირი/road 14/house_number");
        assert!(!augment::reorder(
            "GE",
            &mut decompose(&georgian.text, &georgian.spans),
            &mut rng()
        ));
    }

    #[test]
    fn augmented_rows_are_valid_and_distinct() {
        let e = example(
            "DE",
            "de\tde\tParkstraße/road 8/house_number |/FSEP 10117/postcode Berlin/city |/FSEP Deutschland/country",
        );
        let mut made = 0;
        for seed in 0..50 {
            for a in augment::augment_row(seed, &e, 2, &FeatureConfig::default(), &mut 0) {
                assert!(a.augmented);
                assert_ne!(a.text, e.text);
                assert_eq!(a.group_id, e.group_id);
                assert_invariant(&a);
                made += 1;
            }
        }
        assert!(made > 50, "only {made} copies from 50 seeds");
    }
}
