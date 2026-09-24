//! Feature `markdown`: which bytes of a Markdown document may be scanned.
//!
//! The source is never rewritten. `select` walks pulldown-cmark's offset
//! events once and returns byte ranges into the original string: runs of
//! inline content that the pipeline may scan, ranges inside them that must
//! be skipped (code, raw HTML, link and image destinations, front matter),
//! and link targets. Emphasis markers stay inside runs; the tokenizer treats
//! them as punctuation and the detector was trained with them present.
//!
//! pulldown-cmark builds a block tree for its whole input, about nine times the
//! input's size, so the pipeline parses a document in [`segments`] of about
//! [`SEGMENT_BYTES`], cut where the parser itself reports a top-level block
//! starting after a blank line. Reference links whose definition lies in another
//! piece resolve through the definitions the parser found while planning the
//! cuts. Labels there are matched lowercased rather than Unicode case folded, so
//! a label differing from its definition only in a character such as `ẞ`, with
//! the definition in another piece, stays unresolved.

use std::collections::HashMap;
use std::ops::Range;

use pulldown_cmark::{BrokenLink, CowStr, Event, LinkType, Options, Parser, Tag, TagEnd};

use crate::MarkdownOptions;
use crate::chunk::Mask;

/// A link whose destination may carry a contact detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkTarget {
    /// The destination as written, with `mailto:` prefixed for autolinked emails.
    pub dest: String,
    /// Byte offset where the whole link, markers included, begins.
    pub start: usize,
    /// Byte offset after the whole link, exclusive.
    pub end: usize,
    /// Byte offset where the visible link text begins.
    pub text_start: usize,
    /// Byte offset after the visible link text, exclusive.
    pub text_end: usize,
}

/// Byte ranges of a Markdown document, all into the original string.
///
/// `scan` ranges are sorted and disjoint. `skip` ranges are sorted and lie
/// either inside a `scan` range (inline code, a link destination) or outside
/// every `scan` range (a code block, front matter). Bytes in neither are
/// structural: bullets, heading hashes, table pipes, reference definitions.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Selection {
    pub scan: Vec<(usize, usize)>,
    pub skip: Vec<(usize, usize)>,
    pub links: Vec<LinkTarget>,
}

impl Selection {
    /// The bytes entities may lie in: scan minus skip. For the trainer, which featurizes
    /// Markdown the way the pipeline does; `Mask` is not a stable type.
    #[doc(hidden)]
    pub fn mask(&self) -> Mask {
        Mask::new(&self.scan, &self.skip)
    }

    /// [`Selection::mask`] relative to a piece that starts at `base` in the document.
    pub(crate) fn mask_from(&self, base: usize) -> Mask {
        let back = |ranges: &[(usize, usize)]| -> Vec<(usize, usize)> {
            ranges.iter().map(|&(s, e)| (s - base, e - base)).collect()
        };
        Mask::new(&back(&self.scan), &back(&self.skip))
    }
}

struct LinkFrame {
    dest: String,
    start: usize,
    end: usize,
    text: Option<(usize, usize)>,
}

#[derive(Default)]
struct Builder {
    selection: Selection,
    run: Option<(usize, usize)>,
    /// Depth inside a code block or HTML block being skipped whole.
    opaque: usize,
    /// Depth inside an image, whose alt text is never scanned.
    image: usize,
    links: Vec<LinkFrame>,
}

/// Parser options shared by every parse in this module. Metadata blocks are off: pulldown-cmark
/// recognises them anywhere, which would hide a contact written between two `---` lines, so
/// front matter is found by [`front_matter`] at the document's start only.
fn parser_options(options: &MarkdownOptions) -> Options {
    let mut parser_options =
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS | Options::ENABLE_FOOTNOTES;
    if options.gfm_tables {
        parser_options |= Options::ENABLE_TABLES;
    }
    parser_options
}

/// The YAML (`---`) or TOML (`+++`) front matter opening `text`, closing within its first
/// [`SEGMENT_BYTES`]; longer front matter is read as ordinary Markdown.
fn front_matter(text: &str, options: &MarkdownOptions) -> Option<Range<usize>> {
    if !(text.starts_with("---") || text.starts_with("+++")) {
        return None;
    }
    let prefix = &text[..line_end(text, SEGMENT_BYTES)];
    let metadata = parser_options(options)
        | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS;
    match Parser::new_ext(prefix, metadata).into_offset_iter().next() {
        Some((Event::Start(Tag::MetadataBlock(_)), range)) if range.start == 0 => Some(range),
        _ => None,
    }
}

/// Select the scannable ranges of `text` in one parse. For large documents the pipeline uses
/// [`segments`], which bounds memory; this is for callers that want the whole picture at once.
pub fn select(text: &str, options: &MarkdownOptions) -> Selection {
    select_in(text, options, &HashMap::new(), 0)
}

/// Parse the piece of a document that starts at `base`, skipping the document's front matter
/// when the piece is its start. `defs` resolves reference links whose definitions live in other
/// pieces, and every offset is shifted by `base` so the result indexes the whole document.
fn select_in(
    text: &str,
    options: &MarkdownOptions,
    defs: &HashMap<String, String>,
    base: usize,
) -> Selection {
    let front = if base == 0 {
        front_matter(text, options)
    } else {
        None
    };
    let Some(front) = front else {
        return parse_piece(text, options, defs, base);
    };
    let mut selection = parse_piece(&text[front.end..], options, defs, front.end);
    selection.skip.insert(0, (front.start, front.end));
    selection
}

/// [`select_in`] without the front matter check.
fn parse_piece(
    text: &str,
    options: &MarkdownOptions,
    defs: &HashMap<String, String>,
    base: usize,
) -> Selection {
    let mut builder = Builder::default();
    let callback = |link: BrokenLink<'_>| {
        defs.get(&normalize_label(&link.reference))
            .map(|dest| (CowStr::from(dest.clone()), CowStr::from("")))
    };
    let parser =
        Parser::new_with_broken_link_callback(text, parser_options(options), Some(callback));
    for (event, range) in parser.into_offset_iter() {
        builder.event(event, range, options);
    }
    let mut selection = builder.finish();
    if base > 0 {
        for r in selection.scan.iter_mut().chain(selection.skip.iter_mut()) {
            r.0 += base;
            r.1 += base;
        }
        for l in &mut selection.links {
            l.start += base;
            l.end += base;
            l.text_start += base;
            l.text_end += base;
        }
    }
    selection
}

/// Whitespace collapsed and lowercased: how link labels are matched across pieces.
fn normalize_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    for word in label.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.extend(word.chars().flat_map(char::to_lowercase));
    }
    out
}

/// Target size of one parsed piece. Measured on pulldown-cmark 0.13.4: a 64 KiB piece parses
/// in about 0.6 MB.
pub const SEGMENT_BYTES: usize = 64 * 1024;

/// How far past its target a planning window first reaches for a cut; a window that finds
/// none doubles its reach.
const PLAN_SLACK_BYTES: usize = SEGMENT_BYTES / 4;

/// Where a document is cut into pieces, and the reference definitions pieces resolve through.
#[derive(Default)]
struct Plan {
    cuts: Vec<usize>,
    defs: HashMap<String, String>,
}

/// Plans the cuts by parsing the document one bounded window at a time.
///
/// A cut is the start of a top-level block, as pulldown-cmark reports it, at the start of an
/// unindented line after a blank line, at least `target_bytes` after the previous cut. Whether
/// a block starts there is decided by what precedes it, so the window's verdict holds for the
/// whole document, and the blocks before the cut are complete. Definitions are the parser's
/// own, kept from a window only when they precede its cut, the first of a label winning. A
/// document with no cut has no definitions to share: its one piece resolves its own.
fn plan(text: &str, options: &MarkdownOptions, target_bytes: usize) -> Plan {
    let mut plan = Plan::default();
    let mut start = front_matter(text, options).map_or(0, |r| r.end);
    let mut reach = target_bytes.saturating_add(PLAN_SLACK_BYTES);
    while start.saturating_add(target_bytes) < text.len() {
        let end = line_end(text, start.saturating_add(reach));
        let parser = Parser::new_ext(&text[start..end], parser_options(options));
        let mut events = parser.into_offset_iter();
        let mut depth = 0usize;
        let mut cut = None;
        for (event, range) in events.by_ref() {
            match event {
                Event::Start(_) => {
                    let at = start + range.start;
                    if depth == 0 && at > start && at >= start + target_bytes && is_cut(text, at) {
                        cut = Some(at);
                        break;
                    }
                    depth += 1;
                }
                Event::End(_) => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        let limit = cut.unwrap_or(end);
        if cut.is_some() || end == text.len() {
            for (label, def) in events.reference_definitions().iter() {
                if start + def.span.start < limit {
                    plan.defs
                        .entry(normalize_label(label))
                        .or_insert_with(|| def.dest.to_string());
                }
            }
        }
        match cut {
            Some(at) => {
                plan.cuts.push(at);
                start = at;
                reach = target_bytes.saturating_add(PLAN_SLACK_BYTES);
            }
            None if end == text.len() => return plan,
            None => reach = reach.saturating_mul(2),
        }
    }
    if !plan.cuts.is_empty() {
        let parser = Parser::new_ext(&text[start..], parser_options(options));
        for (label, def) in parser.reference_definitions().iter() {
            plan.defs
                .entry(normalize_label(label))
                .or_insert_with(|| def.dest.to_string());
        }
    }
    plan
}

/// Whether `at` starts an unindented line whose previous line is blank.
fn is_cut(text: &str, at: usize) -> bool {
    let bytes = text.as_bytes();
    if at == 0 || bytes[at - 1] != b'\n' || matches!(bytes.get(at), Some(b' ' | b'\t') | None) {
        return false;
    }
    let before = &bytes[..at - 1];
    let line_start = before
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    before[line_start..]
        .iter()
        .all(|b| matches!(b, b' ' | b'\t' | b'\r'))
}

/// The end of the line holding byte `at`, past its newline; the text's end when `at` is
/// beyond it or no newline follows.
fn line_end(text: &str, at: usize) -> usize {
    if at >= text.len() {
        return text.len();
    }
    text.as_bytes()[at..]
        .iter()
        .position(|&b| b == b'\n')
        .map_or(text.len(), |i| at + i + 1)
}

/// One piece of a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// Byte offset of the piece's first byte in the document.
    pub start: usize,
    /// Byte offset after the piece's last byte, exclusive.
    pub end: usize,
}

/// Pieces of a document parsed one at a time, each with its selection in document offsets.
pub struct Segments<'a> {
    text: &'a str,
    options: &'a MarkdownOptions,
    defs: HashMap<String, String>,
    cuts: std::vec::IntoIter<usize>,
    start: usize,
}

impl Iterator for Segments<'_> {
    type Item = (Segment, Selection);

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.cuts.next()?;
        let segment = Segment {
            start: self.start,
            end,
        };
        let selection = select_in(
            &self.text[segment.start..end],
            self.options,
            &self.defs,
            segment.start,
        );
        self.start = end;
        Some((segment, selection))
    }
}

/// Pieces of about [`SEGMENT_BYTES`], parsed lazily after one planning pass over bounded
/// windows. A top-level block longer than a piece, such as a long table or a paragraph without
/// blank lines, is kept whole, so memory is bounded by the larger of the two. Joined in order
/// with `skip` re-sorted, the selections equal [`select`] on the whole document, except that a
/// label defined twice in different pieces resolves to the definition in its own piece.
pub fn segments<'a>(text: &'a str, options: &'a MarkdownOptions) -> Segments<'a> {
    segments_with(text, options, SEGMENT_BYTES)
}

fn segments_with<'a>(
    text: &'a str,
    options: &'a MarkdownOptions,
    target_bytes: usize,
) -> Segments<'a> {
    let Plan { mut cuts, defs } = plan(text, options, target_bytes);
    cuts.push(text.len());
    Segments {
        text,
        options,
        defs,
        cuts: cuts.into_iter(),
        start: 0,
    }
}

impl Builder {
    fn event(&mut self, event: Event<'_>, range: Range<usize>, options: &MarkdownOptions) {
        if self.opaque > 0 {
            match event {
                Event::Start(Tag::CodeBlock(_) | Tag::HtmlBlock) => self.opaque += 1,
                Event::End(TagEnd::CodeBlock | TagEnd::HtmlBlock) => self.opaque -= 1,
                _ => {}
            }
            return;
        }
        if self.image > 0 {
            match event {
                Event::Start(Tag::Image { .. }) => self.image += 1,
                Event::End(TagEnd::Image) => self.image -= 1,
                _ => {}
            }
            return;
        }
        match event {
            Event::Start(Tag::CodeBlock(_)) if !options.include_code => self.opaque_block(range),
            Event::Start(Tag::HtmlBlock) if !options.include_html => self.opaque_block(range),
            Event::Start(Tag::Image { .. }) => {
                self.extend(&range);
                self.selection.skip.push((range.start, range.end));
                self.image = 1;
            }
            Event::Start(Tag::Link {
                link_type,
                dest_url,
                ..
            }) => {
                self.extend(&range);
                let dest = match link_type {
                    LinkType::Email => format!("mailto:{dest_url}"),
                    _ => dest_url.into_string(),
                };
                self.links.push(LinkFrame {
                    dest,
                    start: range.start,
                    end: range.end,
                    text: None,
                });
            }
            Event::End(TagEnd::Link) => self.close_link(),
            Event::Start(
                Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Superscript
                | Tag::Subscript,
            )
            | Event::End(
                TagEnd::Emphasis
                | TagEnd::Strong
                | TagEnd::Strikethrough
                | TagEnd::Superscript
                | TagEnd::Subscript,
            ) => self.extend(&range),
            Event::Text(_)
            | Event::SoftBreak
            | Event::HardBreak
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_) => self.leaf(&range),
            Event::Code(_) if options.include_code => self.leaf(&range),
            Event::Code(_) => self.skipped_leaf(&range),
            Event::InlineHtml(_) | Event::Html(_) if options.include_html => self.leaf(&range),
            Event::InlineHtml(_) | Event::Html(_) => self.skipped_leaf(&range),
            Event::InlineMath(_) | Event::DisplayMath(_) => self.skipped_leaf(&range),
            Event::Start(_) | Event::End(_) | Event::Rule => self.close_run(),
        }
    }

    fn opaque_block(&mut self, range: Range<usize>) {
        self.close_run();
        self.selection.skip.push((range.start, range.end));
        self.opaque = 1;
    }

    fn extend(&mut self, range: &Range<usize>) {
        self.run = Some(match self.run {
            Some((start, end)) => (start.min(range.start), end.max(range.end)),
            None => (range.start, range.end),
        });
    }

    /// A leaf inline event: extends the run and, inside a link, the link text.
    fn leaf(&mut self, range: &Range<usize>) {
        self.extend(range);
        if let Some(frame) = self.links.last_mut() {
            frame.text = Some(match frame.text {
                Some((start, end)) => (start.min(range.start), end.max(range.end)),
                None => (range.start, range.end),
            });
        }
    }

    fn skipped_leaf(&mut self, range: &Range<usize>) {
        self.leaf(range);
        self.selection.skip.push((range.start, range.end));
    }

    fn close_link(&mut self) {
        let Some(frame) = self.links.pop() else {
            return;
        };
        match frame.text {
            Some((text_start, text_end)) => {
                if frame.start < text_start {
                    self.selection.skip.push((frame.start, text_start));
                }
                if text_end < frame.end {
                    self.selection.skip.push((text_end, frame.end));
                }
                self.selection.links.push(LinkTarget {
                    dest: frame.dest,
                    start: frame.start,
                    end: frame.end,
                    text_start,
                    text_end,
                });
            }
            None => self.selection.skip.push((frame.start, frame.end)),
        }
    }

    fn close_run(&mut self) {
        if let Some(run) = self.run.take() {
            self.selection.scan.push(run);
        }
    }

    fn finish(mut self) -> Selection {
        self.close_run();
        self.selection.skip.sort_unstable();
        self.selection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(text: &str) -> Selection {
        select(text, &MarkdownOptions::default())
    }

    fn slices<'a>(text: &'a str, ranges: &[(usize, usize)]) -> Vec<&'a str> {
        ranges.iter().map(|&(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn headings_and_paragraphs_are_runs_without_markers() {
        let text = "# Kavkaz Freight LLC\n\n## Contact\n\nNino Beridze  \nnino@kavkaz-freight.example\n\n### Phone\n\n+44 20 7946 0958\n";
        let s = sel(text);
        assert_eq!(
            s.scan,
            vec![(2, 20), (25, 32), (34, 76), (82, 87), (89, 105)]
        );
        assert_eq!(
            slices(text, &s.scan),
            [
                "Kavkaz Freight LLC",
                "Contact",
                "Nino Beridze  \nnino@kavkaz-freight.example",
                "Phone",
                "+44 20 7946 0958"
            ]
        );
        assert!(s.skip.is_empty());
        assert!(s.links.is_empty());
    }

    #[test]
    fn emphasis_markers_stay_inside_the_run() {
        let text = "Hello **Nino Beridze**,\n\nSend to *billing@kavkaz-freight.example* and\ncopy ops.  \nThanks\\, Ana\n";
        let s = sel(text);
        assert_eq!(
            slices(text, &s.scan),
            [
                "Hello **Nino Beridze**,",
                "Send to *billing@kavkaz-freight.example* and\ncopy ops.  \nThanks\\, Ana"
            ]
        );
    }

    #[test]
    fn list_items_exclude_bullets_and_keep_continuation_lines() {
        let text = "- Nino Beridze\n- nino@kavkaz-freight.example\n  continuation +44 20 7946 0958\n- Team\n  - lead@kavkaz-freight.example\n\n1. First\n\n   ana@kavkaz-freight.example\n\n- [ ] call +1 202 555 0143\n- [x] done\n";
        let s = sel(text);
        assert_eq!(
            s.scan,
            vec![
                (2, 14),
                (17, 76),
                (79, 83),
                (88, 115),
                (120, 125),
                (130, 156),
                (160, 184),
                (187, 195)
            ]
        );
        assert_eq!(
            &text[17..76],
            "nino@kavkaz-freight.example\n  continuation +44 20 7946 0958"
        );
        assert_eq!(&text[160..184], "[ ] call +1 202 555 0143");
    }

    #[test]
    fn blockquote_prefixes_between_lines_are_inside_the_run() {
        let text = "> Nino Beridze\n> nino@kavkaz-freight.example\n>\n> > +44 20 7946 0958\n\nOn Monday, Ana wrote:\n> ana@kavkaz-freight.example\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(2, 44), (51, 67), (69, 90), (93, 119)]);
        assert_eq!(&text[2..44], "Nino Beridze\n> nino@kavkaz-freight.example");
    }

    #[test]
    fn table_cells_are_separate_runs_and_pipes_are_not_scanned() {
        let text =
            "| Name | Email |\n| --- | --- |\n| Nino Beridze | nino@kavkaz-freight.example |\n";
        let s = sel(text);
        assert_eq!(
            slices(text, &s.scan),
            [
                "Name",
                "Email",
                "Nino Beridze",
                "nino@kavkaz-freight.example"
            ]
        );
        let no_tables = select(
            text,
            &MarkdownOptions {
                gfm_tables: false,
                ..MarkdownOptions::default()
            },
        );
        assert_eq!(no_tables.scan, vec![(0, text.len() - 1)]);
    }

    #[test]
    fn code_is_skipped_unless_included() {
        let text = "Contact nino@kavkaz-freight.example.\n\n```text\nhidden@kavkaz-freight.example\n```\n\n    indented@kavkaz-freight.example\n\nUse `inline@kavkaz-freight.example` for tests.\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(0, 36), (118, 164)]);
        assert_eq!(s.skip, vec![(38, 79), (85, 117), (122, 153)]);
        assert_eq!(&text[38..79], "```text\nhidden@kavkaz-freight.example\n```");
        assert_eq!(&text[122..153], "`inline@kavkaz-freight.example`");
        let included = select(
            text,
            &MarkdownOptions {
                include_code: true,
                ..MarkdownOptions::default()
            },
        );
        assert_eq!(
            included.scan,
            vec![(0, 36), (46, 76), (85, 117), (118, 164)]
        );
        assert!(included.skip.is_empty());
    }

    #[test]
    fn html_is_skipped_unless_included() {
        let text = "<div class=\"card\">\nNino Beridze<br>\nhtml@kavkaz-freight.example\n</div>\n\nLine<br>break with inline@kavkaz-freight.example.\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(72, 121)]);
        assert_eq!(s.skip, vec![(0, 71), (76, 80)]);
        let included = select(
            text,
            &MarkdownOptions {
                include_html: true,
                ..MarkdownOptions::default()
            },
        );
        assert_eq!(included.scan, vec![(0, 71), (72, 121)]);
        assert!(included.skip.is_empty());
    }

    #[test]
    fn dashes_after_the_start_are_not_front_matter() {
        let text = "Text\n\n---\nNino Beridze\nnino@x.example\n---\n\nMore\n";
        let s = sel(text);
        assert!(s.skip.is_empty());
        assert!(
            s.scan
                .iter()
                .any(|&(a, b)| text[a..b].contains("nino@x.example"))
        );
        let late = "\n---\ntitle: t\n---\n";
        assert!(sel(late).skip.is_empty());
    }

    #[test]
    fn front_matter_is_skipped() {
        let yaml = "---\ntitle: Contacts\nemail: fm@kavkaz-freight.example\n---\n\nBody nino@kavkaz-freight.example\n";
        let s = sel(yaml);
        assert_eq!(s.scan, vec![(58, 90)]);
        assert_eq!(s.skip, vec![(0, 56)]);
        let toml =
            "+++\nemail = \"fm@kavkaz-freight.example\"\n+++\n\nBody nino@kavkaz-freight.example\n";
        let s = sel(toml);
        assert_eq!(s.scan, vec![(45, 77)]);
        assert_eq!(s.skip, vec![(0, 43)]);
    }

    #[test]
    fn links_record_targets_and_skip_their_destinations() {
        let text = "Write to [Nino](mailto:nino@kavkaz-freight.example?subject=Hi) or call [the office](tel:+44-20-7946-0958).\nAutolink <ops@kavkaz-freight.example> and site [Kavkaz](https://kavkaz-freight.example/contact).\nReference [billing][b] here.\n\n[b]: mailto:billing@kavkaz-freight.example\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(0, 232)]);
        assert_eq!(
            s.skip,
            vec![
                (9, 10),
                (14, 62),
                (71, 72),
                (82, 105),
                (116, 117),
                (143, 144),
                (154, 155),
                (161, 202),
                (214, 215),
                (222, 226)
            ]
        );
        let targets: Vec<(&str, &str)> = s
            .links
            .iter()
            .map(|l| (l.dest.as_str(), &text[l.text_start..l.text_end]))
            .collect();
        assert_eq!(
            targets,
            [
                ("mailto:nino@kavkaz-freight.example?subject=Hi", "Nino"),
                ("tel:+44-20-7946-0958", "the office"),
                (
                    "mailto:ops@kavkaz-freight.example",
                    "ops@kavkaz-freight.example"
                ),
                ("https://kavkaz-freight.example/contact", "Kavkaz"),
                ("mailto:billing@kavkaz-freight.example", "billing"),
            ]
        );
    }

    #[test]
    fn images_are_skipped_whole_and_alt_text_is_not_link_text() {
        let text = "Text with ![alt nino@kavkaz-freight.example](img.png) image.\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(0, 60)]);
        assert_eq!(s.skip, vec![(10, 53)]);
        assert!(s.links.is_empty());
    }

    #[test]
    fn reference_definitions_and_rules_are_not_scanned() {
        let text = "Para.\n\n***\n\n[b]: mailto:billing@kavkaz-freight.example\n";
        let s = sel(text);
        assert_eq!(s.scan, vec![(0, 5)]);
        assert!(s.skip.is_empty());
    }

    #[test]
    fn invariants_hold_on_every_input() {
        for text in [
            "",
            "\n\n\n",
            "just text",
            "# h\n\n- a\n- b\n\n> q\n\n```\nc\n```\n\n| a | b |\n| - | - |\n| 1 | 2 |\n",
            "[unclosed](mailto:x@y.example",
            "![img](a.png) [l](mailto:a@b.example) `c` <b>x</b>",
        ] {
            let s = sel(text);
            for w in s.scan.windows(2) {
                assert!(w[0].1 <= w[1].0, "scan ranges overlap in {text:?}");
            }
            for &(start, end) in s.scan.iter().chain(&s.skip) {
                assert!(start <= end && end <= text.len());
                assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            }
            for l in &s.links {
                assert!(
                    l.start <= l.text_start && l.text_start <= l.text_end && l.text_end <= l.end
                );
            }
        }
    }

    fn join(segments: Segments<'_>) -> Selection {
        let mut joined = Selection::default();
        for (_, s) in segments {
            joined.scan.extend(s.scan);
            joined.skip.extend(s.skip);
            joined.links.extend(s.links);
        }
        joined.skip.sort_unstable();
        joined
    }

    fn first_lines(text: &str, points: &[usize]) -> Vec<String> {
        points
            .iter()
            .map(|&p| text[p..].lines().next().unwrap_or("").to_string())
            .collect()
    }

    fn cuts(text: &str) -> Vec<String> {
        first_lines(text, &plan(text, &MarkdownOptions::default(), 1).cuts)
    }

    #[test]
    fn cuts_fall_between_top_level_blocks_only() {
        let text = "---\ntitle: t\n---\n\nPara one [x][r].\n\n```\ncode\n\nmore code a@b.example\n```\n\n<script>\nvar a;\n\nvar b;\n</script>\n\n~~~~\ntilde\n\n~~~\nstill code\n~~~~\n\n<!--\n\ncomment@x.example\n\n-->\n\n<!DOCTYPE note\n\nstill html\n\n>\n\n<div>\n```\n</div>\n\nLast para y@z.example.\n\n[r]: mailto:ref@x.example\n";
        assert_eq!(
            cuts(text),
            [
                "Para one [x][r].",
                "```",
                "<script>",
                "~~~~",
                "<!--",
                "<!DOCTYPE note",
                "<div>",
                "Last para y@z.example."
            ]
        );
    }

    #[test]
    fn indented_and_tab_indented_lines_are_never_cuts() {
        let text = "- item\n\n    continued in the item\n\n\tstill the item\n\n  and this\n\nTop level.\n\nNext.\n";
        assert_eq!(cuts(text), ["Top level.", "Next."]);
        let options = MarkdownOptions::default();
        assert_eq!(
            first_lines(text, &plan(text, &options, 60).cuts),
            ["Top level."]
        );
        assert!(plan(text, &options, text.len()).cuts.is_empty());
    }

    #[test]
    fn definitions_are_the_parsers_own() {
        let text = "Intro.\n\n  [ Billing  Team ]: <mailto:billing@x.example> \"t\"\n\n[b]: mailto:b@x.example\n[B]: mailto:ignored@x.example\n\n[multi]:\n  mailto:multi@x.example\n\n> [quoted]: mailto:q@x.example\n\n```\n[fenced]: mailto:code@x.example\n```\n\nA paragraph\n[continued]: mailto:not@x.example\n\nEnd.\n";
        let defs = plan(text, &MarkdownOptions::default(), 1).defs;
        let get = |k: &str| defs.get(k).map(String::as_str);
        assert_eq!(get("billing team"), Some("mailto:billing@x.example"));
        assert_eq!(get("b"), Some("mailto:b@x.example"));
        assert_eq!(get("multi"), Some("mailto:multi@x.example"));
        assert_eq!(get("quoted"), Some("mailto:q@x.example"));
        assert_eq!(get("fenced"), None);
        assert_eq!(get("continued"), None);
    }

    #[test]
    fn a_document_without_cuts_shares_no_definitions() {
        let text = "See the note below.\n[b]: tel:+995322123456\n\n[b]\n";
        let p = plan(text, &MarkdownOptions::default(), SEGMENT_BYTES);
        assert!(p.cuts.is_empty() && p.defs.is_empty());
        assert_eq!(
            join(segments(text, &MarkdownOptions::default())),
            select(text, &MarkdownOptions::default())
        );
    }

    #[test]
    fn a_block_longer_than_the_window_is_kept_whole() {
        let row = "| a | b@x.example |\n";
        let text = format!(
            "Intro.\n\n| h | e |\n| --- | --- |\n{}\nAfter.\n",
            row.repeat(4000)
        );
        let p = plan(&text, &MarkdownOptions::default(), 1);
        assert_eq!(first_lines(&text, &p.cuts), ["| h | e |", "After."]);
    }

    #[test]
    fn cross_segment_reference_links_resolve() {
        let text = "Reference [billing][b] here.\n\n".to_string()
            + &"filler\n\n".repeat(4)
            + "[b]: mailto:billing@x.example\n";
        let options = MarkdownOptions::default();
        let segmented = segments_with(&text, &options, 1);
        assert_eq!(segmented.cuts.len(), 5);
        let joined = join(segmented);
        assert_eq!(joined, select(&text, &options));
        assert_eq!(joined.links.len(), 1);
        assert_eq!(joined.links[0].dest, "mailto:billing@x.example");
        assert_eq!(
            &text[joined.links[0].text_start..joined.links[0].text_end],
            "billing"
        );
    }

    /// Every fixture, cut at every blank line the cutter allows, selects exactly what one
    /// parse of the whole document selects.
    #[test]
    fn segmented_at_every_blank_line_equals_whole_selection() {
        let fixtures = [
            include_str!("../../fixtures/markdown/headings.json"),
            include_str!("../../fixtures/markdown/paragraphs.json"),
            include_str!("../../fixtures/markdown/lists.json"),
            include_str!("../../fixtures/markdown/blockquotes.json"),
            include_str!("../../fixtures/markdown/links.json"),
            include_str!("../../fixtures/markdown/tables.json"),
            include_str!("../../fixtures/markdown/code.json"),
            include_str!("../../fixtures/markdown/html.json"),
            include_str!("../../fixtures/markdown/front_matter.json"),
        ];
        let mut inputs: Vec<(String, MarkdownOptions)> = Vec::new();
        for json in fixtures {
            let file: serde_json::Value = serde_json::from_str(json).unwrap();
            for case in file["cases"].as_array().unwrap() {
                let flag = |k: &str| case["options"][k].as_bool().unwrap();
                let options = MarkdownOptions {
                    include_code: flag("include_code"),
                    include_html: flag("include_html"),
                    gfm_tables: flag("gfm_tables"),
                };
                inputs.push((case["input"].as_str().unwrap().to_string(), options));
            }
        }
        inputs.push((
            include_str!("../../fixtures/markdown/international.md").to_string(),
            MarkdownOptions::default(),
        ));
        for text in [
            "- item\n\n\tcontinued nino@kavkaz-freight.example\n\nTop.\n",
            "<!DOCTYPE note\n\nhidden nino@kavkaz-freight.example\n\n>\n\nAfter.\n",
            "<div>\n```\n</div>\n\nProse.\n\n```\ncode\n\nhidden@kavkaz-freight.example\n```\n\nEnd.\n",
            "Intro.\n\n---\ntitle: not front matter\n---\n\nEnd.\n",
            "Call [Nino][ops] today.\n\n```\n[ops]: mailto:ops@x.example\n```\n\nEnd.\n",
            "See the note below.\n[b]: tel:+995322123456\n\n[b] here.\n",
            "Write to [ops].\n\nFiller.\n\n[ops]:\n  mailto:ops@x.example\n\n> [q]: mailto:q@x.example\n\n[q] too.\n",
        ] {
            inputs.push((text.to_string(), MarkdownOptions::default()));
        }
        inputs.push((
            "- item\n\n    continued para a@x.example\n\n  ```\n  fenced\n\n  in item\n  ```\n\nTop.\n".into(),
            MarkdownOptions {
                include_code: true,
                ..MarkdownOptions::default()
            },
        ));
        for (input, options) in &inputs {
            assert_eq!(
                join(segments_with(input, options, 1)),
                select(input, options),
                "{input:?}"
            );
        }
    }
}
