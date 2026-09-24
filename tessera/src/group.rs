//! Stage 4: block splitting and the deterministic contact grouper.
//!
//! A document is cut into lines at the tokenizer's newline tokens and the lines are grouped
//! into blocks. Blocks are the unit of locality for every grouping rule: details attach to an
//! anchor in the same block, and only a narrow set of rules crosses a block boundary.
//! Limitations: two-column tab-separated rows are not detected as table rows, and chat turns
//! are only recognised when a line starts with a timestamp.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "the grouper of phase 5.2 is the first caller")
)]

use crate::token::{Token, TokenClass};

/// What one line is, judged from its text alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineKind {
    Blank,
    Separator,
    Header,
    Quoted,
    /// A quote marker with nothing after it, which ends a block like a blank line.
    QuotedBlank,
    TableRow,
    ChatTurn,
    /// A sign-off such as `Regards,`, which does not count as a content line.
    Closing,
    Text,
}

/// One line of the document, without its newline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Line {
    pub start: usize,
    pub end: usize,
    pub kind: LineKind,
    /// Characters in the trimmed line, used for the short-line test.
    pub chars: usize,
}

/// What a block of lines is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockKind {
    Paragraph,
    /// A paragraph of short lines, the shape of a signature or letterhead.
    Signature,
    Header,
    QuotedReply,
    TableRow,
    Separator,
    ChatTurn,
}

/// Consecutive lines that belong together, with byte offsets into the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    pub start: usize,
    pub end: usize,
    pub kind: BlockKind,
    pub lines: Vec<Line>,
    /// At least two content lines, none longer than `SHORT_LINE_CHARS`.
    pub short_lines: bool,
    /// Blank lines between the previous block and this one.
    pub gap_lines: usize,
}

const SHORT_LINE_CHARS: usize = 64;
const MIN_SIGNATURE_LINES: usize = 2;

/// The lines of `text`, cut at the tokenizer's newline tokens; a newline belongs to no line,
/// and a last line without one is kept when it is not empty.
pub(crate) fn lines(text: &str, tokens: &[Token]) -> Vec<Line> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for token in tokens.iter().filter(|t| t.class == TokenClass::Newline) {
        out.push(make_line(text, cursor, token.start));
        cursor = token.end;
    }
    if cursor < text.len() {
        out.push(make_line(text, cursor, text.len()));
    }
    out
}

fn make_line(text: &str, start: usize, end: usize) -> Line {
    let raw = &text[start..end];
    let trimmed = raw.trim();
    Line {
        start,
        end,
        kind: classify_line(raw, trimmed),
        chars: trimmed.chars().count(),
    }
}

/// The checks run in a fixed order: blank, separator, quoted, chat turn, table row, header,
/// closing, text.
fn classify_line(raw: &str, trimmed: &str) -> LineKind {
    if trimmed.is_empty() {
        return LineKind::Blank;
    }
    if trimmed.len() >= 3
        && trimmed
            .bytes()
            .all(|b| matches!(b, b'-' | b'_' | b'=' | b'*' | b' '))
    {
        return LineKind::Separator;
    }
    if let Some(rest) = trimmed.strip_prefix('>') {
        return if rest.trim().is_empty() {
            LineKind::QuotedBlank
        } else {
            LineKind::Quoted
        };
    }
    if is_chat_turn(trimmed) {
        return LineKind::ChatTurn;
    }
    if trimmed.matches('|').count() >= 2 || raw.matches('\t').count() >= 2 {
        return LineKind::TableRow;
    }
    if is_header(trimmed) {
        return LineKind::Header;
    }
    if is_closing(trimmed) {
        return LineKind::Closing;
    }
    LineKind::Text
}

const HEADER_PREFIXES: &[&str] = &["From:", "Sent:", "To:", "Cc:", "Subject:", "Date:"];

fn is_header(t: &str) -> bool {
    HEADER_PREFIXES.iter().any(|p| t.starts_with(p))
        || (t.starts_with("On ") && t.ends_with("wrote:"))
}

/// `[10:02]`, `10:02`, `[10:02:33]`, followed by a space, at the start of the line.
fn is_chat_turn(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = 0;
    let bracket = b.first() == Some(&b'[');
    if bracket {
        i += 1;
    }
    let digits = |b: &[u8], i: &mut usize, min: usize, max: usize| -> bool {
        let s = *i;
        while *i < b.len() && b[*i].is_ascii_digit() && *i - s < max {
            *i += 1;
        }
        *i - s >= min
    };
    if !digits(b, &mut i, 1, 2) || b.get(i) != Some(&b':') {
        return false;
    }
    i += 1;
    if !digits(b, &mut i, 2, 2) {
        return false;
    }
    if b.get(i) == Some(&b':') {
        i += 1;
        if !digits(b, &mut i, 2, 2) {
            return false;
        }
    }
    if bracket {
        if b.get(i) != Some(&b']') {
            return false;
        }
        i += 1;
    }
    b.get(i) == Some(&b' ')
}

const CLOSINGS: &[&str] = &[
    "best",
    "best regards",
    "kind regards",
    "regards",
    "warm regards",
    "best wishes",
    "all the best",
    "thanks",
    "thank you",
    "many thanks",
    "cheers",
    "sincerely",
    "yours sincerely",
    "yours faithfully",
    "yours truly",
    "mit freundlichen grüßen",
    "viele grüße",
    "beste grüße",
    "freundliche grüße",
    "met vriendelijke groet",
    "პატივისცემით",
    "よろしくお願いいたします",
    "敬具",
];

fn is_closing(t: &str) -> bool {
    let stripped = t.trim_end_matches([',', '.', '!']).trim_end();
    CLOSINGS.contains(&stripped.to_lowercase().as_str())
}

/// Groups the lines of `text` into blocks. Blank and quoted-blank lines end a block;
/// separators and table rows are blocks of their own; consecutive header lines, quoted lines,
/// and a chat turn with the text lines after it each form one block; everything else is a
/// paragraph, reported as a signature when its lines are short.
pub(crate) fn split_blocks(text: &str, tokens: &[Token]) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut current: Vec<Line> = Vec::new();
    let mut current_kind = BlockKind::Paragraph;
    let mut gap = 0;

    for line in lines(text, tokens) {
        match line.kind {
            LineKind::Blank | LineKind::QuotedBlank => {
                flush(&mut blocks, &mut current, current_kind, &mut gap);
                gap += 1;
            }
            LineKind::Separator => {
                flush(&mut blocks, &mut current, current_kind, &mut gap);
                current.push(line);
                flush(&mut blocks, &mut current, BlockKind::Separator, &mut gap);
            }
            LineKind::TableRow => {
                flush(&mut blocks, &mut current, current_kind, &mut gap);
                current.push(line);
                flush(&mut blocks, &mut current, BlockKind::TableRow, &mut gap);
            }
            LineKind::ChatTurn => {
                flush(&mut blocks, &mut current, current_kind, &mut gap);
                current_kind = BlockKind::ChatTurn;
                current.push(line);
            }
            LineKind::Header => {
                if current_kind != BlockKind::Header {
                    flush(&mut blocks, &mut current, current_kind, &mut gap);
                    current_kind = BlockKind::Header;
                }
                current.push(line);
            }
            LineKind::Quoted => {
                if current_kind != BlockKind::QuotedReply {
                    flush(&mut blocks, &mut current, current_kind, &mut gap);
                    current_kind = BlockKind::QuotedReply;
                }
                current.push(line);
            }
            LineKind::Text | LineKind::Closing => {
                if matches!(current_kind, BlockKind::Header | BlockKind::QuotedReply) {
                    flush(&mut blocks, &mut current, current_kind, &mut gap);
                }
                if current.is_empty() {
                    current_kind = BlockKind::Paragraph;
                }
                current.push(line);
            }
        }
    }
    flush(&mut blocks, &mut current, current_kind, &mut gap);
    blocks
}

fn flush(blocks: &mut Vec<Block>, current: &mut Vec<Line>, kind: BlockKind, gap: &mut usize) {
    let Some((first, last)) = current.first().zip(current.last()) else {
        return;
    };
    let (start, end) = (first.start, last.end);
    let lines = std::mem::take(current);
    let content: Vec<&Line> = lines
        .iter()
        .filter(|l| {
            matches!(
                l.kind,
                LineKind::Text | LineKind::Quoted | LineKind::ChatTurn | LineKind::Header
            )
        })
        .collect();
    let short_lines =
        content.len() >= MIN_SIGNATURE_LINES && content.iter().all(|l| l.chars <= SHORT_LINE_CHARS);
    let kind = match kind {
        BlockKind::Paragraph if short_lines => BlockKind::Signature,
        other => other,
    };
    blocks.push(Block {
        start,
        end,
        kind,
        lines,
        short_lines,
        gap_lines: *gap,
    });
    *gap = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::tokenize;

    fn kinds(text: &str) -> Vec<(BlockKind, usize, usize, bool, usize)> {
        let tokens = tokenize(text);
        split_blocks(text, &tokens)
            .into_iter()
            .map(|b| (b.kind, b.start, b.end, b.short_lines, b.gap_lines))
            .collect()
    }

    #[test]
    fn blank_line_splits_paragraphs() {
        assert_eq!(
            kinds("one two three\n\nfour five"),
            vec![
                (BlockKind::Paragraph, 0, 13, false, 0),
                (BlockKind::Paragraph, 15, 24, false, 1)
            ]
        );
    }

    #[test]
    fn two_short_lines_are_a_signature() {
        assert_eq!(
            kinds("Nino Beridze\nKavkaz Freight LLC"),
            vec![(BlockKind::Signature, 0, 31, true, 0)]
        );
    }

    #[test]
    fn closing_line_does_not_count_as_content() {
        assert_eq!(
            kinds("Best,\nTom"),
            vec![(BlockKind::Paragraph, 0, 9, false, 0)]
        );
        assert_eq!(
            kinds("Best,\nTom\nAcme"),
            vec![(BlockKind::Signature, 0, 14, true, 0)]
        );
    }

    #[test]
    fn separator_line_is_its_own_block() {
        assert_eq!(
            kinds("a\n---\nb"),
            vec![
                (BlockKind::Paragraph, 0, 1, false, 0),
                (BlockKind::Separator, 2, 5, false, 0),
                (BlockKind::Paragraph, 6, 7, false, 0),
            ]
        );
    }

    #[test]
    fn quoted_lines_form_a_block_and_quoted_blank_splits() {
        assert_eq!(
            kinds("reply\n> a\n> b\n>\n> c"),
            vec![
                (BlockKind::Paragraph, 0, 5, false, 0),
                (BlockKind::QuotedReply, 6, 13, true, 0),
                (BlockKind::QuotedReply, 16, 19, false, 1),
            ]
        );
    }

    #[test]
    fn header_lines_form_one_block() {
        let text = "From: A <a@x.example>\nTo: B <b@x.example>\nSubject: hi\n\nbody";
        assert_eq!(
            kinds(text),
            vec![
                (BlockKind::Header, 0, 53, true, 0),
                (BlockKind::Paragraph, 55, 59, false, 1)
            ]
        );
    }

    #[test]
    fn on_wrote_line_is_a_header() {
        let text = "On Fri, 12 Sep 2026, A B <a@x.example> wrote:\n> hi";
        assert_eq!(
            kinds(text),
            vec![
                (BlockKind::Header, 0, 45, false, 0),
                (BlockKind::QuotedReply, 46, 50, false, 0)
            ]
        );
    }

    #[test]
    fn table_rows_are_separate_blocks() {
        assert_eq!(
            kinds("a | b | c\nd | e | f"),
            vec![
                (BlockKind::TableRow, 0, 9, false, 0),
                (BlockKind::TableRow, 10, 19, false, 0)
            ]
        );
        assert_eq!(
            kinds("a\tb\tc\nd\te\tf"),
            vec![
                (BlockKind::TableRow, 0, 5, false, 0),
                (BlockKind::TableRow, 6, 11, false, 0)
            ]
        );
    }

    #[test]
    fn timestamped_lines_are_chat_turns() {
        let text = "[10:02] Agent: hi\nmore of the same message\n[10:03] Customer: ok";
        assert_eq!(
            kinds(text),
            vec![
                (BlockKind::ChatTurn, 0, 42, true, 0),
                (BlockKind::ChatTurn, 43, 63, false, 0)
            ]
        );
    }

    #[test]
    fn crlf_lines_split_like_lf() {
        assert_eq!(
            kinds("a\r\n\r\nb"),
            vec![
                (BlockKind::Paragraph, 0, 1, false, 0),
                (BlockKind::Paragraph, 5, 6, false, 1)
            ]
        );
    }

    #[test]
    fn long_lines_are_not_a_signature() {
        let long = "x".repeat(65);
        let text = format!("{long}\n{long}");
        assert_eq!(kinds(&text)[0].0, BlockKind::Paragraph);
    }

    #[test]
    fn cjk_line_length_is_counted_in_chars() {
        let text = "株式会社ヘリオス照明\n〒100-0005 東京都千代田区丸の内1-1-1";
        assert_eq!(kinds(text)[0].0, BlockKind::Signature);
    }
}
