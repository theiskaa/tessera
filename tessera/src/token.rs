//! Reversible, Unicode-aware tokenizer shared by every stage and by the trainer.
//!
//! Every token carries the exact byte range of its text. Whitespace is kept as
//! tokens so that features can see line shape; consumers filter it. Script and
//! class detection use hand-written range tables covering only what the
//! product needs, because a general Unicode table would cost more in wasm than
//! the whole rules layer. Normalization never happens here: the source string
//! and its offsets are the contract everything else depends on.

/// Writing system of an alphabetic token. Digits, punctuation, and whitespace are `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Cyrillic,
    Georgian,
    Arabic,
    Hebrew,
    Greek,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Thai,
    Devanagari,
    Other,
}

/// Coarse character class of a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    Alpha,
    Digit,
    Alnum,
    Punct,
    Space,
    Newline,
    Other,
}

/// One token: a byte range into the source string plus its class and script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// UTF-8 byte offset where the token begins.
    pub start: usize,
    /// UTF-8 byte offset where the token ends, exclusive.
    pub end: usize,
    /// Coarse character class.
    pub class: TokenClass,
    /// Writing system of its letters; `Other` for digits, punctuation, and whitespace.
    pub script: Script,
}

impl Token {
    /// The token's text, sliced from the source it was produced from.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharKind {
    Alpha(Script),
    /// `set` distinguishes ASCII, fullwidth, and Arabic-Indic digits so runs do not mix.
    Digit {
        set: u8,
    },
    Punct,
    Space,
    Newline,
    Combining,
    Other,
}

fn classify(c: char) -> CharKind {
    let u = c as u32;
    if c == '\n' || c == '\r' || u == 0x85 || u == 0x2028 || u == 0x2029 {
        return CharKind::Newline;
    }
    if c == ' '
        || c == '\t'
        || u == 0x0B
        || u == 0x0C
        || u == 0xA0
        || u == 0x1680
        || (0x2000..=0x200A).contains(&u)
        || u == 0x202F
        || u == 0x205F
        || u == 0x3000
    {
        return CharKind::Space;
    }
    if c.is_ascii_digit() {
        return CharKind::Digit { set: 0 };
    }
    if (0xFF10..=0xFF19).contains(&u) {
        return CharKind::Digit { set: 1 };
    }
    if (0x0660..=0x0669).contains(&u) || (0x06F0..=0x06F9).contains(&u) {
        return CharKind::Digit { set: 2 };
    }
    if in_ranges(u, COMBINING) {
        return CharKind::Combining;
    }
    if c.is_ascii_punctuation() || in_ranges(u, PUNCT) {
        return CharKind::Punct;
    }
    if let Some(script) = script_of(u) {
        return CharKind::Alpha(script);
    }
    CharKind::Other
}

fn in_ranges(u: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(lo, hi)| u >= lo && u <= hi)
}

fn script_of(u: u32) -> Option<Script> {
    const TABLE: &[(Script, &[(u32, u32)])] = &[
        (Script::Latin, LATIN),
        (Script::Cyrillic, CYRILLIC),
        (Script::Georgian, GEORGIAN),
        (Script::Arabic, ARABIC),
        (Script::Hebrew, HEBREW),
        (Script::Greek, GREEK),
        (Script::Hiragana, HIRAGANA),
        (Script::Katakana, KATAKANA),
        (Script::Han, HAN),
        (Script::Hangul, HANGUL),
        (Script::Thai, THAI),
        (Script::Devanagari, DEVANAGARI),
    ];
    TABLE.iter().find(|(_, r)| in_ranges(u, r)).map(|(s, _)| *s)
}

const LATIN: &[(u32, u32)] = &[
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00AA, 0x00AA),
    (0x00B5, 0x00B5),
    (0x00BA, 0x00BA),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x024F),
    (0x0250, 0x02AF),
    (0x1E00, 0x1EFF),
    (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
];
const CYRILLIC: &[(u32, u32)] = &[
    (0x0400, 0x0482),
    (0x048A, 0x052F),
    (0x1C80, 0x1C8F),
    (0x2DE0, 0x2DFF),
    (0xA640, 0xA69F),
];
const GEORGIAN: &[(u32, u32)] = &[
    (0x10A0, 0x10FA),
    (0x10FC, 0x10FF),
    (0x1C90, 0x1CBF),
    (0x2D00, 0x2D2F),
];
const ARABIC: &[(u32, u32)] = &[
    (0x0620, 0x064A),
    (0x066E, 0x066F),
    (0x0671, 0x06D3),
    (0x06D5, 0x06D5),
    (0x06EE, 0x06EF),
    (0x06FA, 0x06FC),
    (0x06FF, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08C9),
    (0xFB50, 0xFBFF),
    (0xFC00, 0xFDFF),
    (0xFE70, 0xFEFC),
];
const HEBREW: &[(u32, u32)] = &[(0x05D0, 0x05EA), (0x05EF, 0x05F2), (0xFB1D, 0xFB4F)];
const GREEK: &[(u32, u32)] = &[
    (0x0370, 0x0373),
    (0x0376, 0x037D),
    (0x037F, 0x0383),
    (0x0386, 0x03FF),
    (0x1F00, 0x1FFF),
];
const HAN: &[(u32, u32)] = &[
    (0x2E80, 0x2FDF),
    (0x3005, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303B),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
];
const HIRAGANA: &[(u32, u32)] = &[(0x3041, 0x3096), (0x309D, 0x309F)];
const KATAKANA: &[(u32, u32)] = &[(0x30A1, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)];
const HANGUL: &[(u32, u32)] = &[
    (0x1100, 0x11FF),
    (0x3131, 0x318E),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7FF),
];
const THAI: &[(u32, u32)] = &[(0x0E01, 0x0E30), (0x0E32, 0x0E33), (0x0E40, 0x0E46)];
const DEVANAGARI: &[(u32, u32)] = &[
    (0x0904, 0x0939),
    (0x093D, 0x093D),
    (0x0950, 0x0950),
    (0x0958, 0x0961),
    (0x0972, 0x097F),
    (0xA8E0, 0xA8FF),
];

const COMBINING: &[(u32, u32)] = &[
    (0x0300, 0x036F),
    (0x0483, 0x0489),
    (0x0591, 0x05BD),
    (0x05BF, 0x05BF),
    (0x05C1, 0x05C2),
    (0x05C4, 0x05C5),
    (0x05C7, 0x05C7),
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06DC),
    (0x06DF, 0x06E4),
    (0x06E7, 0x06E8),
    (0x06EA, 0x06ED),
    (0x0900, 0x0903),
    (0x093A, 0x093C),
    (0x093E, 0x094F),
    (0x0951, 0x0957),
    (0x0962, 0x0963),
    (0x0E31, 0x0E31),
    (0x0E34, 0x0E3A),
    (0x0E47, 0x0E4E),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x200C, 0x200D),
    (0x20D0, 0x20FF),
    (0x3099, 0x309A),
    (0xFE00, 0xFE0F),
    (0xFE20, 0xFE2F),
    (0x1F3FB, 0x1F3FF),
    (0xE0100, 0xE01EF),
];

const PUNCT: &[(u32, u32)] = &[
    (0x00A1, 0x00A9),
    (0x00AB, 0x00B4),
    (0x00B6, 0x00B9),
    (0x00BB, 0x00BF),
    (0x00D7, 0x00D7),
    (0x00F7, 0x00F7),
    (0x055A, 0x055F),
    (0x0589, 0x058A),
    (0x05BE, 0x05BE),
    (0x05C0, 0x05C0),
    (0x05C3, 0x05C3),
    (0x05C6, 0x05C6),
    (0x05F3, 0x05F4),
    (0x0609, 0x060D),
    (0x061B, 0x061F),
    (0x066A, 0x066D),
    (0x06D4, 0x06D4),
    (0x0964, 0x0965),
    (0x0970, 0x0970),
    (0x0E4F, 0x0E4F),
    (0x0E5A, 0x0E5B),
    (0x10FB, 0x10FB),
    (0x2010, 0x2027),
    (0x2030, 0x205E),
    (0x20A0, 0x20CF),
    (0x2100, 0x214F),
    (0x2190, 0x21FF),
    (0x2200, 0x22FF),
    (0x2300, 0x23FF),
    (0x2500, 0x27BF),
    (0x2E00, 0x2E7F),
    (0x3001, 0x3004),
    (0x3008, 0x3020),
    (0x3030, 0x3037),
    (0x303D, 0x303F),
    (0x30FB, 0x30FB),
    (0xFE30, 0xFE4F),
    (0xFE50, 0xFE6B),
    (0xFF01, 0xFF0F),
    (0xFF1A, 0xFF20),
    (0xFF3B, 0xFF40),
    (0xFF5B, 0xFF65),
];

/// Tokenize `text`. The result is sorted, non-overlapping, and covers every byte:
/// concatenating the token texts in order reproduces `text` exactly.
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::with_capacity(text.len() / 4 + 1);
    let mut chars = text.char_indices().peekable();
    let mut cur: Option<(Token, u8)> = None; // token plus digit set, for Digit tokens

    while let Some((i, c)) = chars.next() {
        let end = i + c.len_utf8();
        let next = chars.peek().map(|&(_, n)| n);
        let kind = classify(c);

        match kind {
            CharKind::Newline => {
                flush(&mut tokens, &mut cur);
                let mut tok = Token {
                    start: i,
                    end,
                    class: TokenClass::Newline,
                    script: Script::Other,
                };
                if c == '\r' && next == Some('\n') {
                    chars.next();
                    tok.end = end + 1;
                }
                tokens.push(tok);
            }
            CharKind::Space => extend_or_start(
                &mut tokens,
                &mut cur,
                i,
                end,
                TokenClass::Space,
                Script::Other,
                0,
            ),
            CharKind::Combining => match &mut cur {
                Some((tok, _)) => tok.end = end,
                None => match tokens.last_mut() {
                    Some(last) if last.class != TokenClass::Newline => last.end = end,
                    _ => {
                        cur = Some((
                            Token {
                                start: i,
                                end,
                                class: TokenClass::Other,
                                script: Script::Other,
                            },
                            0,
                        ))
                    }
                },
            },
            CharKind::Punct => {
                let joiner = matches!(c, '\'' | '\u{2019}' | '-');
                let joins_alpha = joiner
                    && matches!(&cur, Some((t, _)) if t.class == TokenClass::Alpha && t.end == i)
                    && matches!(next.map(classify), Some(CharKind::Alpha(s)) if Some(s) == cur.as_ref().map(|(t, _)| t.script));
                let joins_digit = c == '.'
                    && matches!(&cur, Some((t, set)) if t.class == TokenClass::Digit && *set == 0 && t.end == i)
                    && next.is_some_and(|n| n.is_ascii_digit());
                if joins_alpha || joins_digit {
                    if let Some((tok, _)) = &mut cur {
                        tok.end = end;
                    }
                } else {
                    flush(&mut tokens, &mut cur);
                    tokens.push(Token {
                        start: i,
                        end,
                        class: TokenClass::Punct,
                        script: Script::Other,
                    });
                }
            }
            CharKind::Digit { set } => {
                let merge_alnum = set == 0
                    && matches!(&cur, Some((t, _)) if t.end == i
                        && (t.class == TokenClass::Alnum
                            || (t.class == TokenClass::Alpha && alnum_script(t.script))));
                match &mut cur {
                    Some((tok, cur_set))
                        if tok.class == TokenClass::Digit && *cur_set == set && tok.end == i =>
                    {
                        tok.end = end
                    }
                    Some((tok, _)) if merge_alnum => {
                        tok.class = TokenClass::Alnum;
                        tok.end = end;
                    }
                    _ => {
                        flush(&mut tokens, &mut cur);
                        cur = Some((
                            Token {
                                start: i,
                                end,
                                class: TokenClass::Digit,
                                script: Script::Other,
                            },
                            set,
                        ));
                    }
                }
            }
            CharKind::Alpha(script) => {
                let merge_alnum = alnum_script(script)
                    && matches!(&cur, Some((t, set)) if t.end == i
                        && ((t.class == TokenClass::Digit && *set == 0)
                            || (t.class == TokenClass::Alnum && t.script == script)));
                match &mut cur {
                    Some((tok, _))
                        if tok.class == TokenClass::Alpha
                            && tok.script == script
                            && script != Script::Han
                            && tok.end == i =>
                    {
                        tok.end = end
                    }
                    Some((tok, _)) if merge_alnum => {
                        tok.class = TokenClass::Alnum;
                        tok.script = script;
                        tok.end = end;
                    }
                    _ => {
                        flush(&mut tokens, &mut cur);
                        cur = Some((
                            Token {
                                start: i,
                                end,
                                class: TokenClass::Alpha,
                                script,
                            },
                            0,
                        ));
                    }
                }
            }
            CharKind::Other => extend_or_start(
                &mut tokens,
                &mut cur,
                i,
                end,
                TokenClass::Other,
                Script::Other,
                0,
            ),
        }
    }
    flush(&mut tokens, &mut cur);
    tokens
}

fn alnum_script(s: Script) -> bool {
    matches!(s, Script::Latin | Script::Cyrillic | Script::Greek)
}

fn flush(tokens: &mut Vec<Token>, cur: &mut Option<(Token, u8)>) {
    if let Some((tok, _)) = cur.take() {
        tokens.push(tok);
    }
}

fn extend_or_start(
    tokens: &mut Vec<Token>,
    cur: &mut Option<(Token, u8)>,
    start: usize,
    end: usize,
    class: TokenClass,
    script: Script,
    set: u8,
) {
    match cur {
        Some((tok, _)) if tok.class == class && tok.end == start => tok.end = end,
        _ => {
            flush(tokens, cur);
            *cur = Some((
                Token {
                    start,
                    end,
                    class,
                    script,
                },
                set,
            ));
        }
    }
}

/// UTF-16 code unit offset for every byte offset `0..=text.len()`.
///
/// Byte offsets inside a multi-byte character map to the start of that
/// character; only char boundaries are meaningful span ends.
pub fn utf16_offsets(text: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(text.len() + 1);
    let mut units: u32 = 0;
    for c in text.chars() {
        for _ in 0..c.len_utf8() {
            out.push(units);
        }
        units += c.len_utf16() as u32;
    }
    out.push(units);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(text: &str) -> Vec<(&str, TokenClass, Script)> {
        let toks = tokenize(text);
        let joined: String = toks.iter().map(|t| t.text(text)).collect();
        assert_eq!(joined, text, "tokens must reproduce the input");
        for w in toks.windows(2) {
            assert_eq!(w[0].end, w[1].start, "tokens must be contiguous");
        }
        toks.iter()
            .map(|t| (t.text(text), t.class, t.script))
            .collect()
    }

    use Script::{Arabic, Cyrillic, Georgian, Han, Hiragana, Katakana, Latin};
    use TokenClass::{Alnum, Alpha, Digit, Newline, Punct, Space};

    #[test]
    fn latin_words_and_spaces() {
        assert_eq!(
            parts("Nino Beridze"),
            vec![
                ("Nino", Alpha, Latin),
                (" ", Space, Script::Other),
                ("Beridze", Alpha, Latin)
            ]
        );
    }

    #[test]
    fn apostrophe_and_hyphen_join_same_script() {
        assert_eq!(parts("O'Brien"), vec![("O'Brien", Alpha, Latin)]);
        assert_eq!(parts("Jean-Luc"), vec![("Jean-Luc", Alpha, Latin)]);
        assert_eq!(
            parts("Jean- Luc"),
            vec![
                ("Jean", Alpha, Latin),
                ("-", Punct, Script::Other),
                (" ", Space, Script::Other),
                ("Luc", Alpha, Latin)
            ]
        );
        assert_eq!(
            parts("-Luc"),
            vec![("-", Punct, Script::Other), ("Luc", Alpha, Latin)]
        );
    }

    #[test]
    fn digits_glued_to_latin_letters_are_alnum() {
        assert_eq!(parts("221B"), vec![("221B", Alnum, Latin)]);
        assert_eq!(
            parts("NW1 6XE"),
            vec![
                ("NW1", Alnum, Latin),
                (" ", Space, Script::Other),
                ("6XE", Alnum, Latin)
            ]
        );
        assert_eq!(parts("14Rustaveli"), vec![("14Rustaveli", Alnum, Latin)]);
    }

    #[test]
    fn decimal_point_stays_inside_digits() {
        assert_eq!(
            parts("1.5 km"),
            vec![
                ("1.5", Digit, Script::Other),
                (" ", Space, Script::Other),
                ("km", Alpha, Latin)
            ]
        );
        assert_eq!(
            parts("1."),
            vec![("1", Digit, Script::Other), (".", Punct, Script::Other)]
        );
    }

    #[test]
    fn punctuation_is_split() {
        assert_eq!(
            parts("+995 (32)"),
            vec![
                ("+", Punct, Script::Other),
                ("995", Digit, Script::Other),
                (" ", Space, Script::Other),
                ("(", Punct, Script::Other),
                ("32", Digit, Script::Other),
                (")", Punct, Script::Other)
            ]
        );
        assert_eq!(
            parts("a@b.example"),
            vec![
                ("a", Alpha, Latin),
                ("@", Punct, Script::Other),
                ("b", Alpha, Latin),
                (".", Punct, Script::Other),
                ("example", Alpha, Latin)
            ]
        );
    }

    #[test]
    fn scripts_split_runs() {
        assert_eq!(
            parts("Tbilisiთბილისი"),
            vec![("Tbilisi", Alpha, Latin), ("თბილისი", Alpha, Georgian)]
        );
        assert_eq!(
            parts("რუსთაველის14"),
            vec![
                ("რუსთაველის", Alpha, Georgian),
                ("14", Digit, Script::Other)
            ]
        );
        assert_eq!(parts("Иван"), vec![("Иван", Alpha, Cyrillic)]);
        assert_eq!(parts("شارع"), vec![("شارع", Alpha, Arabic)]);
    }

    #[test]
    fn han_is_one_token_per_ideograph_kana_runs() {
        assert_eq!(
            parts("東京都"),
            vec![("東", Alpha, Han), ("京", Alpha, Han), ("都", Alpha, Han)]
        );
        assert_eq!(
            parts("さいたま市"),
            vec![("さいたま", Alpha, Hiragana), ("市", Alpha, Han)]
        );
        assert_eq!(parts("トヨタ"), vec![("トヨタ", Alpha, Katakana)]);
        assert_eq!(
            parts("3丁目"),
            vec![
                ("3", Digit, Script::Other),
                ("丁", Alpha, Han),
                ("目", Alpha, Han)
            ]
        );
    }

    #[test]
    fn newlines() {
        assert_eq!(
            parts("a\r\nb"),
            vec![
                ("a", Alpha, Latin),
                ("\r\n", Newline, Script::Other),
                ("b", Alpha, Latin)
            ]
        );
        assert_eq!(
            parts("a\n\nb"),
            vec![
                ("a", Alpha, Latin),
                ("\n", Newline, Script::Other),
                ("\n", Newline, Script::Other),
                ("b", Alpha, Latin)
            ]
        );
        assert_eq!(
            parts("a  \tb"),
            vec![
                ("a", Alpha, Latin),
                ("  \t", Space, Script::Other),
                ("b", Alpha, Latin)
            ]
        );
    }

    #[test]
    fn combining_marks_attach() {
        assert_eq!(parts("Jose\u{301}"), vec![("Jose\u{301}", Alpha, Latin)]);
        assert_eq!(
            parts("\u{301}x"),
            vec![
                ("\u{301}", TokenClass::Other, Script::Other),
                ("x", Alpha, Latin)
            ]
        );
        assert_eq!(parts("José"), vec![("José", Alpha, Latin)]);
    }

    #[test]
    fn emoji_and_astral() {
        assert_eq!(
            parts("📞 Nino"),
            vec![
                ("📞", TokenClass::Other, Script::Other),
                (" ", Space, Script::Other),
                ("Nino", Alpha, Latin)
            ]
        );
        assert_eq!(parts("👍🏽"), vec![("👍🏽", TokenClass::Other, Script::Other)]);
    }

    #[test]
    fn non_ascii_digits() {
        assert_eq!(parts("٠١٢"), vec![("٠١٢", Digit, Script::Other)]);
        assert_eq!(parts("１２３"), vec![("１２３", Digit, Script::Other)]);
        assert_eq!(
            parts("1١"),
            vec![("1", Digit, Script::Other), ("١", Digit, Script::Other)]
        );
    }

    #[test]
    fn empty_input() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn utf16_mapping() {
        assert_eq!(utf16_offsets(""), vec![0]);
        assert_eq!(utf16_offsets("ab"), vec![0, 1, 2]);
        assert_eq!(utf16_offsets("é"), vec![0, 0, 1]); // 2 bytes, 1 unit
        assert_eq!(utf16_offsets("ა"), vec![0, 0, 0, 1]); // 3 bytes, 1 unit
        assert_eq!(utf16_offsets("📞"), vec![0, 0, 0, 0, 2]); // 4 bytes, 2 units
        assert_eq!(utf16_offsets("📞a"), vec![0, 0, 0, 0, 2, 3]);
    }
}
