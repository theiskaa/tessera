//! Feature `markdown`: which bytes of a Markdown document may be scanned.
//!
//! The source is never rewritten. `select` walks pulldown-cmark's offset
//! events once and returns byte ranges into the original string: runs of
//! inline content that the pipeline may scan, ranges inside them that must
//! be skipped (code, raw HTML, link and image destinations, front matter),
//! and link targets. Emphasis markers stay inside runs; the tokenizer treats
//! them as punctuation and the detector was trained with them present.

use std::ops::Range;

use pulldown_cmark::{Event, LinkType, Options, Parser, Tag, TagEnd};

use crate::MarkdownOptions;

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
    pub fn mask(&self) -> crate::chunk::Mask {
        crate::chunk::Mask::new(&self.scan, &self.skip)
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

/// Select the scannable ranges of `text`.
pub fn select(text: &str, options: &MarkdownOptions) -> Selection {
    let mut builder = Builder::default();
    for (event, range) in Parser::new_ext(text, parser_options(options)).into_offset_iter() {
        builder.event(event, range, options);
    }
    builder.finish()
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
}
