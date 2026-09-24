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
//! [`SEGMENT_BYTES`] cut at blank lines before unindented lines, outside fenced
//! code, raw HTML blocks that may hold blank lines, and front matter. Reference
//! links whose definition lies in another piece resolve through a prescan of
//! every `[label]: destination` line. Labels there are matched lowercased rather
//! than Unicode case folded, so a label differing from its definition only in a
//! character such as `ẞ`, with the definition in another piece, stays unresolved.

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
    /// The whole link, markers included.
    pub start: usize,
    pub end: usize,
    /// The visible link text.
    pub text_start: usize,
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
    /// The bytes entities may lie in: scan minus skip.
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
    /// Depth inside a code block, HTML block, or metadata block being skipped whole.
    opaque: usize,
    /// Depth inside an image, whose alt text is never scanned.
    image: usize,
    links: Vec<LinkFrame>,
}

/// Parser options shared by every parse in this module.
pub(crate) fn parser_options(options: &MarkdownOptions) -> Options {
    let mut parser_options = Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS;
    if options.gfm_tables {
        parser_options |= Options::ENABLE_TABLES;
    }
    parser_options
}

/// Select the scannable ranges of `text` in one parse. For large documents the pipeline uses
/// [`segments`], which bounds memory; this is for callers that want the whole picture at once.
pub fn select(text: &str, options: &MarkdownOptions) -> Selection {
    select_in(text, options, &HashMap::new(), 0)
}

/// Parse one piece of a document. `defs` resolves reference links whose definitions live in
/// other pieces; `base` shifts every offset so the result indexes the whole document.
fn select_in(
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

/// Whitespace collapsed, trimmed, lowercased: how link labels are matched.
fn normalize_label(label: &str) -> String {
    label
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// `[label]: destination` lines anywhere in the document, the first definition of a label
/// winning as in CommonMark.
fn reference_definitions(text: &str) -> HashMap<String, String> {
    let mut defs = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim_start_matches(' ');
        if line.len() - trimmed.len() > 3 || !trimmed.starts_with('[') {
            continue;
        }
        let Some(close) = trimmed.find("]:") else {
            continue;
        };
        let label = &trimmed[1..close];
        if label.trim().is_empty() || label.contains('[') {
            continue;
        }
        let dest = trimmed[close + 2..].split_whitespace().next().unwrap_or("");
        let dest = dest.trim_start_matches('<').trim_end_matches('>');
        if dest.is_empty() {
            continue;
        }
        defs.entry(normalize_label(label))
            .or_insert_with(|| dest.to_string());
    }
    defs
}

/// Target size of one parsed piece. Measured on pulldown-cmark 0.13.4: a 64 KiB piece parses
/// in about 0.6 MB.
pub const SEGMENT_BYTES: usize = 64 * 1024;

enum CutState {
    Normal,
    Fence { marker: u8, len: usize },
    RawHtml(&'static str),
    FrontMatter(&'static str),
}

/// Raw HTML block kinds 1 to 5, which may contain blank lines: start condition, end condition.
const RAW_HTML: [(&str, &str); 7] = [
    ("<script", "</script>"),
    ("<pre", "</pre>"),
    ("<style", "</style>"),
    ("<textarea", "</textarea>"),
    ("<!--", "-->"),
    ("<?", "?>"),
    ("<![cdata[", "]]>"),
];

/// Byte offsets where the document may be cut without changing inline runs: the start of an
/// unindented non-blank line after a blank line, outside fenced code, raw HTML blocks, and
/// front matter, each at least `target_bytes` after the previous cut.
///
/// An indented line is never a cut: after a blank line it may continue a list item, and on its
/// own it would parse as an indented code block.
fn split_points(text: &str, target_bytes: usize) -> Vec<usize> {
    let mut state = CutState::Normal;
    let mut points = Vec::new();
    let mut last_cut = 0;
    let mut offset = 0;
    let mut previous_blank = false;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let start = offset;
        offset += line.len();
        let content = line.trim_end_matches(['\n', '\r']);
        let stripped = content.trim_start_matches(' ');
        let indent = content.len() - stripped.len();
        let blank = content.trim().is_empty();
        match &state {
            CutState::Normal => {}
            CutState::FrontMatter(close) => {
                if content == *close {
                    state = CutState::Normal;
                }
                continue;
            }
            CutState::Fence { marker, len } => {
                let run = stripped.bytes().take_while(|b| b == marker).count();
                if indent <= 3 && run >= *len && stripped[run..].trim().is_empty() {
                    state = CutState::Normal;
                }
                previous_blank = false;
                continue;
            }
            CutState::RawHtml(close) => {
                if content.to_ascii_lowercase().contains(close) {
                    state = CutState::Normal;
                }
                previous_blank = false;
                continue;
            }
        }
        if i == 0 && (content == "---" || content == "+++") {
            state = CutState::FrontMatter(if content == "---" { "---" } else { "+++" });
            continue;
        }
        if previous_blank && !blank && indent == 0 && start >= last_cut + target_bytes {
            points.push(start);
            last_cut = start;
        }
        previous_blank = blank;
        if indent > 3 {
            continue;
        }
        let backticks = stripped.bytes().take_while(|b| *b == b'`').count();
        if backticks >= 3 && !stripped[backticks..].contains('`') {
            state = CutState::Fence {
                marker: b'`',
                len: backticks,
            };
            continue;
        }
        let tildes = stripped.bytes().take_while(|b| *b == b'~').count();
        if tildes >= 3 {
            state = CutState::Fence {
                marker: b'~',
                len: tildes,
            };
            continue;
        }
        let lower = stripped.to_ascii_lowercase();
        for (open, close) in RAW_HTML {
            let bare = open.starts_with("<!") || open == "<?";
            let delimited = lower.len() == open.len()
                || matches!(lower.as_bytes().get(open.len()), Some(b' ' | b'>' | b'\t'));
            if lower.starts_with(open) && (bare || delimited) {
                if !lower[open.len()..].contains(close) {
                    state = CutState::RawHtml(close);
                }
                break;
            }
        }
    }
    points
}

/// One piece of a document, as byte offsets into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
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

/// Pieces of about [`SEGMENT_BYTES`], parsed lazily. Their selections, joined in order with
/// `skip` re-sorted, equal [`select`] on the whole document.
pub fn segments<'a>(text: &'a str, options: &'a MarkdownOptions) -> Segments<'a> {
    segments_with(text, options, SEGMENT_BYTES)
}

fn segments_with<'a>(
    text: &'a str,
    options: &'a MarkdownOptions,
    target_bytes: usize,
) -> Segments<'a> {
    let mut cuts = split_points(text, target_bytes);
    cuts.push(text.len());
    Segments {
        text,
        options,
        defs: reference_definitions(text),
        cuts: cuts.into_iter(),
        start: 0,
    }
}

impl Builder {
    fn event(&mut self, event: Event<'_>, range: Range<usize>, options: &MarkdownOptions) {
        if self.opaque > 0 {
            match event {
                Event::Start(Tag::CodeBlock(_) | Tag::HtmlBlock | Tag::MetadataBlock(_)) => {
                    self.opaque += 1
                }
                Event::End(TagEnd::CodeBlock | TagEnd::HtmlBlock | TagEnd::MetadataBlock(_)) => {
                    self.opaque -= 1
                }
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
            Event::Start(Tag::MetadataBlock(_)) => self.opaque_block(range),
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

    #[test]
    fn split_points_respect_fences_html_and_front_matter() {
        let text = "---\ntitle: t\n\n---\n\nPara one [x][r].\n\n```\ncode\n\nmore code a@b.example\n```\n\n<script>\nvar a;\n\nvar b;\n</script>\n\n~~~~\ntilde\n\n~~~\nstill code\n~~~~\n\n<!--\n\ncomment@x.example\n\n-->\n\nLast para y@z.example.\n\n[r]: mailto:ref@x.example\n";
        assert_eq!(
            first_lines(text, &split_points(text, 1)),
            [
                "Para one [x][r].",
                "```",
                "<script>",
                "~~~~",
                "<!--",
                "Last para y@z.example.",
                "[r]: mailto:ref@x.example"
            ]
        );
    }

    #[test]
    fn split_points_skip_indented_lines_and_respect_the_target() {
        let text =
            "- item\n\n    continued in the item\n\n  still the item\n\nTop level.\n\nNext.\n";
        assert_eq!(
            first_lines(text, &split_points(text, 1)),
            ["Top level.", "Next."]
        );
        assert_eq!(first_lines(text, &split_points(text, 50)), ["Top level."]);
        assert!(split_points(text, text.len()).is_empty());
    }

    #[test]
    fn reference_definitions_are_collected_and_normalized() {
        let defs = reference_definitions(
            "x\n\n  [ Billing  Team ]: <mailto:billing@x.example> \"t\"\n[b]: mailto:b@x.example\n[B]: mailto:ignored@x.example\n    [indented]: mailto:code@x.example\n",
        );
        assert_eq!(
            defs.get("billing team").map(String::as_str),
            Some("mailto:billing@x.example")
        );
        assert_eq!(
            defs.get("b").map(String::as_str),
            Some("mailto:b@x.example")
        );
        assert!(!defs.contains_key("indented"));
    }

    #[test]
    fn cross_segment_reference_links_resolve() {
        let text = "Reference [billing][b] here.\n\n".to_string()
            + &"filler\n\n".repeat(4)
            + "[b]: mailto:billing@x.example\n";
        let options = MarkdownOptions::default();
        let segmented = segments_with(&text, &options, 1);
        assert_eq!(segmented.cuts.len(), 6);
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
