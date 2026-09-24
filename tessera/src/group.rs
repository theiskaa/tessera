//! Stage 4: block splitting and the deterministic contact grouper.
//!
//! A document is cut into lines at the tokenizer's newline tokens and the lines are grouped
//! into blocks. Blocks are the unit of locality for every grouping rule: details attach to an
//! anchor in the same block, and only a narrow set of rules crosses a block boundary.
//! Limitations: two-column tab-separated rows are not detected as table rows, and chat turns
//! are only recognised when a line starts with a timestamp.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "extract_contacts in phase 5.3 is the first caller"
    )
)]

use crate::policy::MEDIUM;
use crate::token::{Token, TokenClass};
use crate::{Contact, Entity, Extraction, Kind};

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

/// Where an entity sits: its block and the line (index into the block's `lines`) holding its
/// first byte.
#[derive(Debug, Clone, Copy)]
struct Placed {
    entity: usize,
    block: usize,
    line: usize,
}

/// Every entity's place, or `None` for one that starts before the first block or on no line,
/// which detector output never does; such an entity is left unassigned.
fn place(blocks: &[Block], entities: &[Entity]) -> Vec<Option<Placed>> {
    entities
        .iter()
        .enumerate()
        .map(|(entity, e)| {
            let block = blocks.iter().rposition(|b| b.start <= e.start)?;
            let line = blocks[block]
                .lines
                .iter()
                .rposition(|l| l.start <= e.start)?;
            Some(Placed {
                entity,
                block,
                line,
            })
        })
        .collect()
}

/// How much an attachment across `d` lines keeps of the entity's confidence. Three lines is the
/// normal span of a signature (name, title, company, then details), so it costs nothing.
fn distance_factor(d: usize) -> f32 {
    match d {
        0..=3 => 1.0,
        4..=6 => 0.90,
        _ => 0.70,
    }
}

fn is_attachable(kind: Kind) -> bool {
    matches!(kind, Kind::Address | Kind::Email | Kind::Phone)
}

/// Latin-script folding for the email name match. `None` for a letter outside the table, which
/// makes the match refuse rather than guess.
fn fold(c: char) -> Option<char> {
    if c.is_ascii_alphabetic() {
        return Some(c.to_ascii_lowercase());
    }
    let folded = match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å'
        | 'Ā' => 'a',
        'ç' | 'ć' | 'č' | 'Ç' | 'Ć' | 'Č' => 'c',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ě' | 'ę' | 'È' | 'É' | 'Ê' | 'Ë' | 'Ē' => 'e',
        'ì' | 'í' | 'î' | 'ï' | 'ī' | 'Ì' | 'Í' | 'Î' | 'Ï' => 'i',
        'ñ' | 'ń' | 'ň' | 'Ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' | 'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => {
            'o'
        }
        'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'Ù' | 'Ú' | 'Û' | 'Ü' => 'u',
        'ý' | 'ÿ' | 'Ý' => 'y',
        'ğ' | 'Ğ' => 'g',
        'ş' | 'š' | 'ś' | 'Ş' | 'Š' => 's',
        'ž' | 'ź' | 'ż' | 'Ž' => 'z',
        'ł' | 'Ł' => 'l',
        'ř' | 'Ř' => 'r',
        'ť' | 'Ť' => 't',
        'ď' | 'Ď' => 'd',
        _ => return None,
    };
    Some(folded)
}

/// A person span as folded name parts; `None` when any letter is outside the folding table
/// (non-Latin scripts) or no part remains.
fn name_parts(person: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for c in person.chars() {
        // `ß` is alphabetic but has no one-letter fold, so it is spelled out before the table.
        if c == 'ß' {
            current.push_str("ss");
        } else if c.is_alphabetic() {
            current.push(fold(c)?);
        } else if !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    (!parts.is_empty()).then_some(parts)
}

const HONORIFICS: &[&str] = &[
    "dr", "mr", "mrs", "ms", "mx", "prof", "sir", "herr", "frau", "hr",
];

/// Whether an email local part spells a person's name: given and family name in either order,
/// with an initial for either, joined or split by `.` `_` `-`; a lone given or family name of
/// at least four letters; or the initials of a name with at least two parts.
pub(crate) fn name_match(local: &str, person: &str) -> bool {
    let Some(mut parts) = name_parts(person) else {
        return false;
    };
    parts.retain(|p| !HONORIFICS.contains(&p.as_str()));
    let (Some(given), Some(family)) = (parts.first(), parts.last()) else {
        return false;
    };
    let (given, family) = (given.as_str(), family.as_str());
    let local = local
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_ascii_lowercase();
    let local_parts: Vec<&str> = local
        .split(['.', '_', '-'])
        .filter(|p| !p.is_empty())
        .collect();
    let joined: String = local_parts.concat();
    if joined.is_empty() {
        return false;
    }
    let initials: String = parts.iter().filter_map(|p| p.chars().next()).collect();
    let g_initial: String = given.chars().take(1).collect();
    let f_initial: String = family.chars().take(1).collect();

    if let [a, b] = local_parts[..] {
        if (a == given && b == family) || (a == family && b == given) {
            return true;
        }
        if (a == g_initial && b == family) || (a == given && b == f_initial) {
            return true;
        }
    }
    joined == format!("{given}{family}")
        || joined == format!("{family}{given}")
        || joined == format!("{g_initial}{family}")
        || joined == format!("{given}{f_initial}")
        || joined == format!("{family}{g_initial}")
        || (joined.len() >= 4 && (joined == given || joined == family))
        || (parts.len() >= 2 && joined == initials)
        || (local_parts.len() >= 3 && joined == parts.concat())
}

/// The grouper's working state: the contacts drafted so far, which contact each entity went
/// to, and each entity's assignment confidence.
struct Build {
    contacts: Vec<Draft>,
    assigned: Vec<Option<usize>>,
    confidence: Vec<f32>,
}

/// A contact being assembled, by entity index.
struct Draft {
    anchor: usize,
    block: usize,
    line: usize,
    person: Option<usize>,
    org: Option<usize>,
    members: Vec<usize>,
}

/// One contact per person, and per org in a block with no person.
fn anchors(blocks: &[Block], entities: &[Entity], placed: &[Option<Placed>]) -> Build {
    let mut contacts = Vec::new();
    for block in 0..blocks.len() {
        let in_block: Vec<Placed> = placed
            .iter()
            .flatten()
            .copied()
            .filter(|p| p.block == block)
            .collect();
        let has_person = in_block
            .iter()
            .any(|p| entities[p.entity].kind == Kind::Person);
        for p in &in_block {
            let kind = entities[p.entity].kind;
            if kind == Kind::Person || (kind == Kind::Org && !has_person) {
                contacts.push(Draft {
                    anchor: p.entity,
                    block: p.block,
                    line: p.line,
                    person: (kind == Kind::Person).then_some(p.entity),
                    org: (kind == Kind::Org).then_some(p.entity),
                    members: Vec::new(),
                });
            }
        }
    }
    let mut build = Build {
        contacts,
        assigned: vec![None; entities.len()],
        confidence: vec![1.0; entities.len()],
    };
    for (idx, draft) in build.contacts.iter().enumerate() {
        build.assigned[draft.anchor] = Some(idx);
        build.confidence[draft.anchor] = entities[draft.anchor].confidence;
    }
    build
}

/// The only place an assignment is made, so the `MEDIUM` floor holds for every rule.
fn try_assign(
    build: &mut Build,
    entities: &[Entity],
    entity: usize,
    contact: usize,
    factor: f32,
) -> bool {
    let conf = entities[entity].confidence * factor;
    if conf < MEDIUM {
        return false;
    }
    build.assigned[entity] = Some(contact);
    build.confidence[entity] = conf;
    build.contacts[contact].members.push(entity);
    true
}

/// An org in a block with people goes to the nearest person, preceding on a tie, within three
/// lines, with no person or org between them and no org of its own yet. Entities are sorted by
/// start, so "between" is an index range.
fn attach_orgs(build: &mut Build, entities: &[Entity], placed: &[Option<Placed>]) {
    for p in placed.iter().flatten() {
        if entities[p.entity].kind != Kind::Org || build.assigned[p.entity].is_some() {
            continue;
        }
        // (contact, line distance, whether the person precedes the org)
        let mut best: Option<(usize, usize, bool)> = None;
        for (ci, draft) in build.contacts.iter().enumerate() {
            let Some(person) = draft.person else {
                continue;
            };
            if draft.block != p.block || draft.org.is_some() {
                continue;
            }
            let (lo, hi) = (person.min(p.entity), person.max(p.entity));
            let blocked =
                (lo + 1..hi).any(|i| matches!(entities[i].kind, Kind::Person | Kind::Org));
            let d = draft.line.abs_diff(p.line);
            if blocked || d > 3 {
                continue;
            }
            let preceding = person < p.entity;
            let better = match best {
                None => true,
                Some((_, bd, bp)) => d < bd || (d == bd && preceding && !bp),
            };
            if better {
                best = Some((ci, d, preceding));
            }
        }
        if let Some((ci, d, _)) = best
            && try_assign(build, entities, p.entity, ci, distance_factor(d))
        {
            build.contacts[ci].org = Some(p.entity);
        }
    }
}

/// Each address, email, and phone goes to the first contact these rules name, in order: the one
/// anchor on its line; for an email, the one person its local part spells; the owner of the
/// details on the line above; the nearest preceding anchor in the block; in a short-line block,
/// the nearest anchor at most two lines below.
fn attach_details(
    build: &mut Build,
    text: &str,
    blocks: &[Block],
    entities: &[Entity],
    placed: &[Option<Placed>],
) {
    let slice = |i: usize| text.get(entities[i].start..entities[i].end).unwrap_or("");
    for p in placed.iter().flatten() {
        let e = &entities[p.entity];
        if !is_attachable(e.kind) || build.assigned[p.entity].is_some() {
            continue;
        }
        let block = &blocks[p.block];
        let anchors: Vec<(usize, usize)> = build
            .contacts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.block == p.block)
            .map(|(ci, d)| (ci, d.line))
            .collect();

        let same_line: Vec<usize> = anchors
            .iter()
            .filter(|(_, l)| *l == p.line)
            .map(|(ci, _)| *ci)
            .collect();
        if let [only] = same_line[..] {
            try_assign(build, entities, p.entity, only, 1.0);
            continue;
        }

        if e.kind == Kind::Email {
            let address = e.normalized.as_deref().unwrap_or(slice(p.entity));
            let local = address.split('@').next().unwrap_or("");
            let matches: Vec<usize> = anchors
                .iter()
                .map(|(ci, _)| *ci)
                .filter(|ci| {
                    build.contacts[*ci]
                        .person
                        .is_some_and(|person| name_match(local, slice(person)))
                })
                .collect();
            if let [only] = matches[..]
                && try_assign(build, entities, p.entity, only, 1.0)
            {
                continue;
            }
        }

        if let Some(prev) = previous_content_line(block, p.line) {
            let owners: Vec<usize> = placed
                .iter()
                .flatten()
                .filter(|q| {
                    q.block == p.block && q.line == prev && is_attachable(entities[q.entity].kind)
                })
                .filter_map(|q| build.assigned[q.entity])
                .collect();
            if let Some(&owner) = owners.first()
                && owners.iter().all(|c| *c == owner)
                && try_assign(build, entities, p.entity, owner, 1.0)
            {
                continue;
            }
        }

        let preceding = anchors
            .iter()
            .filter(|(ci, l)| *l <= p.line && entities[build.contacts[*ci].anchor].start <= e.start)
            .max_by_key(|(ci, l)| (*l, entities[build.contacts[*ci].anchor].start))
            .copied();
        if let Some((ci, line)) = preceding
            && try_assign(
                build,
                entities,
                p.entity,
                ci,
                distance_factor(p.line - line),
            )
        {
            continue;
        }

        if block.short_lines
            && let Some((ci, _)) = anchors
                .iter()
                .filter(|(_, l)| *l > p.line && *l - p.line <= 2)
                .min_by_key(|(_, l)| *l)
                .copied()
        {
            try_assign(build, entities, p.entity, ci, 0.9);
        }
    }
}

fn previous_content_line(block: &Block, line: usize) -> Option<usize> {
    (0..line)
        .rev()
        .find(|&i| !matches!(block.lines[i].kind, LineKind::Blank | LineKind::QuotedBlank))
}

/// A short-line block with no anchor, one blank line below a block that holds a contact (and is
/// not a separator), gives its details to that block's last contact: the signature layout with
/// a blank line between the name lines and the address lines.
fn adopt_orphans(
    build: &mut Build,
    blocks: &[Block],
    entities: &[Entity],
    placed: &[Option<Placed>],
) {
    for (bi, block) in blocks.iter().enumerate().skip(1) {
        if !block.short_lines
            || block.gap_lines != 1
            || blocks[bi - 1].kind == BlockKind::Separator
            || build.contacts.iter().any(|d| d.block == bi)
        {
            continue;
        }
        let Some(target) = build
            .contacts
            .iter()
            .enumerate()
            .filter(|(_, d)| d.block == bi - 1)
            .max_by_key(|(_, d)| entities[d.anchor].start)
            .map(|(ci, _)| ci)
        else {
            continue;
        };
        for p in placed.iter().flatten().filter(|p| p.block == bi) {
            if is_attachable(entities[p.entity].kind) && build.assigned[p.entity].is_none() {
                try_assign(build, entities, p.entity, target, 0.8);
            }
        }
    }
}

/// A contact that is only its anchor is a real record in a signature, a header, or a table row;
/// elsewhere (a greeting, a name in prose) it is noise.
fn keeps_lone_anchor(block: &Block) -> bool {
    block.short_lines || matches!(block.kind, BlockKind::Header | BlockKind::TableRow)
}

/// Groups `entities`, sorted by start, into contacts: anchored on each person, or on an org in a
/// block with no person, with details attached by the rules above. A contact's confidence is
/// the minimum of its anchor's and its assignments'. Everything left over is `unassigned`.
/// No confidence policy is applied here.
pub(crate) fn group(text: &str, tokens: &[Token], entities: Vec<Entity>) -> Extraction {
    let blocks = split_blocks(text, tokens);
    let placed = place(&blocks, &entities);
    let mut build = anchors(&blocks, &entities, &placed);
    attach_orgs(&mut build, &entities, &placed);
    attach_details(&mut build, text, &blocks, &entities, &placed);
    adopt_orphans(&mut build, &blocks, &entities, &placed);

    let dropped: Vec<bool> = build
        .contacts
        .iter()
        .map(|d| d.members.is_empty() && !keeps_lone_anchor(&blocks[d.block]))
        .collect();
    for (draft, _) in build.contacts.iter().zip(&dropped).filter(|(_, d)| **d) {
        build.assigned[draft.anchor] = None;
    }

    let mut slots: Vec<Option<Entity>> = entities.into_iter().map(Some).collect();
    let mut contacts = Vec::new();
    for (draft, _) in build.contacts.iter().zip(&dropped).filter(|(_, d)| !**d) {
        let mut contact = Contact {
            start: usize::MAX,
            end: 0,
            confidence: build.confidence[draft.anchor],
            review_recommended: false,
            person: None,
            org: None,
            addresses: Vec::new(),
            emails: Vec::new(),
            phones: Vec::new(),
        };
        let mut members = draft.members.clone();
        members.push(draft.anchor);
        members.sort_unstable();
        for idx in members {
            let Some(e) = slots[idx].take() else {
                continue;
            };
            contact.start = contact.start.min(e.start);
            contact.end = contact.end.max(e.end);
            contact.confidence = contact.confidence.min(build.confidence[idx]);
            match e.kind {
                Kind::Person => contact.person = Some(e),
                Kind::Org => contact.org = Some(e),
                Kind::Address => contact.addresses.push(e),
                Kind::Email => contact.emails.push(e),
                Kind::Phone => contact.phones.push(e),
            }
        }
        contacts.push(contact);
    }
    contacts.sort_by_key(|c| c.start);
    let mut unassigned: Vec<Entity> = slots.into_iter().flatten().collect();
    unassigned.sort_by_key(|e| e.start);
    Extraction {
        contacts,
        unassigned,
    }
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

    const SPEC: &str = "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example";

    const SPEC_ENTITIES: &[(Kind, &str)] = &[
        (Kind::Person, "Nino Beridze"),
        (Kind::Org, "Kavkaz Freight LLC"),
        (Kind::Address, "14 Rustaveli Avenue, Tbilisi 0108, Georgia"),
        (Kind::Phone, "+995 32 212 3456"),
        (Kind::Email, "nino@kavkaz-freight.example"),
    ];

    fn gold(text: &str, spec: &[(Kind, &str)]) -> Vec<Entity> {
        let mut from = 0;
        spec.iter()
            .map(|(kind, s)| {
                let start = text[from..].find(s).expect("gold text present") + from;
                from = start + s.len();
                Entity {
                    kind: *kind,
                    start,
                    end: start + s.len(),
                    confidence: 0.95,
                    source: crate::Source::Model,
                    components: Vec::new(),
                    normalized: (*kind == Kind::Email).then(|| s.to_ascii_lowercase()),
                    region: None,
                    review_recommended: false,
                }
            })
            .collect()
    }

    fn run(text: &str, spec: &[(Kind, &str)]) -> Extraction {
        group(text, &tokenize(text), gold(text, spec))
    }

    /// Each contact as its person or org text, then the texts of its details.
    fn cards<'a>(text: &'a str, x: &Extraction) -> Vec<Vec<&'a str>> {
        x.contacts
            .iter()
            .map(|c| {
                c.person
                    .iter()
                    .chain(&c.org)
                    .chain(&c.addresses)
                    .chain(&c.emails)
                    .chain(&c.phones)
                    .map(|e| e.text(text))
                    .collect()
            })
            .collect()
    }

    fn unassigned<'a>(text: &'a str, x: &Extraction) -> Vec<&'a str> {
        x.unassigned.iter().map(|e| e.text(text)).collect()
    }

    #[test]
    fn spec_example_groups_into_one_contact() {
        let x = run(SPEC, SPEC_ENTITIES);
        assert_eq!(
            cards(SPEC, &x),
            vec![vec![
                "Nino Beridze",
                "Kavkaz Freight LLC",
                "14 Rustaveli Avenue, Tbilisi 0108, Georgia",
                "nino@kavkaz-freight.example",
                "+995 32 212 3456",
            ]]
        );
        let c = &x.contacts[0];
        assert_eq!((c.start, c.end), (28, 147));
        assert!((c.confidence - 0.95).abs() < 1e-6);
        assert!(x.unassigned.is_empty());
    }

    #[test]
    fn same_line_rule_wins_over_order() {
        let text = "billing@okafor-consulting.example, attention Amelia Okafor\nx y";
        let x = run(
            text,
            &[
                (Kind::Email, "billing@okafor-consulting.example"),
                (Kind::Person, "Amelia Okafor"),
            ],
        );
        assert_eq!(
            cards(text, &x),
            vec![vec!["Amelia Okafor", "billing@okafor-consulting.example"]]
        );
    }

    const TWO_PEOPLE: &str = "Amelia Okafor\nNino Beridze\nnino@kavkaz-freight.example\n+995 32 212 3456\namelia.okafor@northgate.example\n+44 20 7946 0123";

    fn two_people() -> Extraction {
        run(
            TWO_PEOPLE,
            &[
                (Kind::Person, "Amelia Okafor"),
                (Kind::Person, "Nino Beridze"),
                (Kind::Email, "nino@kavkaz-freight.example"),
                (Kind::Phone, "+995 32 212 3456"),
                (Kind::Email, "amelia.okafor@northgate.example"),
                (Kind::Phone, "+44 20 7946 0123"),
            ],
        )
    }

    #[test]
    fn email_name_match_overrides_nearest_preceding() {
        let x = two_people();
        assert_eq!(
            cards(TWO_PEOPLE, &x),
            vec![
                vec![
                    "Amelia Okafor",
                    "amelia.okafor@northgate.example",
                    "+44 20 7946 0123"
                ],
                vec![
                    "Nino Beridze",
                    "nino@kavkaz-freight.example",
                    "+995 32 212 3456"
                ],
            ]
        );
        assert!(x.unassigned.is_empty());
    }

    #[test]
    fn trailing_phone_follows_email() {
        let x = two_people();
        let amelia = &x.contacts[0];
        assert_eq!(amelia.phones.len(), 1);
        assert_eq!(amelia.phones[0].text(TWO_PEOPLE), "+44 20 7946 0123");
    }

    #[test]
    fn org_attaches_to_nearest_person_without_person_between() {
        let text = "Amelia Okafor\nNino Beridze\nKavkaz Freight LLC";
        let x = run(
            text,
            &[
                (Kind::Person, "Amelia Okafor"),
                (Kind::Person, "Nino Beridze"),
                (Kind::Org, "Kavkaz Freight LLC"),
            ],
        );
        assert_eq!(
            cards(text, &x),
            vec![
                vec!["Amelia Okafor"],
                vec!["Nino Beridze", "Kavkaz Freight LLC"]
            ]
        );
    }

    #[test]
    fn org_anchors_when_block_has_no_person() {
        let text = "Customer Care\nHelios Lighting GmbH\nsupport@helios-lighting.example";
        let x = run(
            text,
            &[
                (Kind::Org, "Helios Lighting GmbH"),
                (Kind::Email, "support@helios-lighting.example"),
            ],
        );
        assert_eq!(
            cards(text, &x),
            vec![vec![
                "Helios Lighting GmbH",
                "support@helios-lighting.example"
            ]]
        );
        assert!(x.contacts[0].person.is_none());
    }

    const ORPHAN: &[(Kind, &str)] = &[
        (Kind::Person, "Nino Beridze"),
        (Kind::Org, "Kavkaz Freight LLC"),
        (Kind::Address, "14 Rustaveli Avenue, Tbilisi 0108, Georgia"),
        (Kind::Phone, "+995 32 212 3456"),
    ];

    #[test]
    fn orphan_block_is_adopted_across_one_blank_line() {
        let text = "Nino Beridze\nKavkaz Freight LLC\n\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456";
        let x = run(text, ORPHAN);
        assert_eq!(
            cards(text, &x),
            vec![vec![
                "Nino Beridze",
                "Kavkaz Freight LLC",
                "14 Rustaveli Avenue, Tbilisi 0108, Georgia",
                "+995 32 212 3456",
            ]]
        );
        assert!((x.contacts[0].confidence - 0.95 * 0.8).abs() < 1e-6);
    }

    #[test]
    fn orphan_block_not_adopted_across_two_blank_lines() {
        let text = "Nino Beridze\nKavkaz Freight LLC\n\n\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456";
        let x = run(text, ORPHAN);
        assert_eq!(
            cards(text, &x),
            vec![vec!["Nino Beridze", "Kavkaz Freight LLC"]]
        );
        assert_eq!(
            unassigned(text, &x),
            vec![
                "14 Rustaveli Avenue, Tbilisi 0108, Georgia",
                "+995 32 212 3456"
            ]
        );
    }

    #[test]
    fn lone_greeting_is_unassigned() {
        let text = "Hi Nino,\n\nSee you Monday.";
        let x = run(text, &[(Kind::Person, "Nino")]);
        assert!(x.contacts.is_empty());
        assert_eq!(unassigned(text, &x), vec!["Nino"]);
    }

    #[test]
    fn lone_person_in_signature_is_kept() {
        let text = "Amelia Okafor\nSenior Buyer";
        let x = run(text, &[(Kind::Person, "Amelia Okafor")]);
        assert_eq!(cards(text, &x), vec![vec!["Amelia Okafor"]]);
    }

    #[test]
    fn low_confidence_assignment_is_refused() {
        let text = SPEC.replace(
            "Kavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n",
            "Kavkaz Freight LLC\nx\nx\nx\nx\nx\nx\n",
        );
        let mut entities = gold(
            &text,
            &[
                (Kind::Person, "Nino Beridze"),
                (Kind::Org, "Kavkaz Freight LLC"),
                (Kind::Phone, "+995 32 212 3456"),
            ],
        );
        entities[2].confidence = 0.5;
        let x = group(&text, &tokenize(&text), entities);
        assert_eq!(unassigned(&text, &x), vec!["+995 32 212 3456"]);
    }

    #[test]
    fn contact_confidence_is_minimum_not_average() {
        let mut entities = gold(SPEC, SPEC_ENTITIES);
        entities[4].confidence = 0.6;
        let x = group(SPEC, &tokenize(SPEC), entities);
        assert_eq!(x.contacts.len(), 1);
        assert!((x.contacts[0].confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn name_match_cases() {
        let cases = [
            ("nino", "Nino Beridze", true),
            ("amelia.okafor", "Dr Amelia Okafor", true),
            ("aokafor", "Amelia Okafor", true),
            ("okafora", "Amelia Okafor", true),
            ("ao", "Amelia Okafor", true),
            ("support", "Amelia Okafor", false),
            ("nino", "ნინო ბერიძე", false),
            ("tom", "Tom", false),
            ("jean-luc.picard", "Jean-Luc Picard", true),
            ("juergen.weiss", "Jürgen Weiß", false),
            ("jurgen.weiss", "Jürgen Weiß", true),
            ("nino42", "Nino Beridze", true),
        ];
        for (local, person, want) in cases {
            assert_eq!(name_match(local, person), want, "{local} / {person}");
        }
    }
}
