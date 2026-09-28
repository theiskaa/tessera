//! Text around the entities: localized sentences, reference lines, and reply headers, and the
//! wrap that pads a rendered document with them. Real documents are mostly text that is not a
//! contact, in the document's own language, full of numbers, dates, and capitalized words; the
//! templates alone are short, English, and dense with entities, so a detector trained on them
//! tags any unfamiliar line as something.

use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;

use crate::generate::{Doc, Gold, Link, localized};
use crate::negatives;
use crate::templates;

/// English sentences with times, amounts, counts, and reference numbers, and no names.
const EN_NUMERIC: &[&str] = &[
    "The three crates were due on 4 March, 09:00–11:00.",
    "The loading dock code is 2290 and it shuts at 17:45.",
    "The call is booked for Thursday at 10:30.",
    "We shipped 24 cartons this morning.",
    "The balance of 1,240.00 is due within 30 days.",
    "Delivery takes 3 to 5 working days.",
    "Your case number is 2025-1103-8826.",
    "Please allow 48 hours for the refund to show.",
    "The quote is valid for 30 days.",
    "Our hours are 9:00 to 17:30, Monday to Friday.",
    "The meeting moved from 14:00 to 15:30.",
    "Page 2 of the form still needs a signature.",
    "The unit price drops to 12.40 above 100 pieces.",
    "We received 3 of the 4 parcels.",
    "Section 4.2 of the contract covers returns.",
    "The update went out at 06:15 this morning.",
    "Invoice 2025-04418 is attached.",
    "The warehouse opens at 7 and closes at 4.",
    "The courier page has shown the same status since Tuesday.",
    "If the line is busy, try again after lunch.",
];

/// Lines that head or number a document: subjects, references, table headers, bank and tax
/// numbers, all fictitious.
const LINES: &[(&str, &[&str])] = &[(
    "en",
    &[
        "Invoice No. 2025-07742",
        "Customer ID 61930",
        "Order #77106-C",
        "Ticket 2025-0611-2093",
        "PO 81-4407-D",
        "Due: February 9, 2026",
        "Subject: Re: delivery for order 5162",
        "Description          Qty   Price     Amount",
        "Page 2 of 3",
        "Tax ID 00-0000000",
        "Our ref: KB/2026/118",
        "Sent: Monday 15 September 2026 09:12",
        "Confidential: for the addressee only.",
        "Subtotal 1,540.00 · Shipping 86.40 · Total 1,626.40",
        "Account no. 00000000 · Routing no. 000000000",
    ],
)];

/// Headers above a quoted reply that name its author where `{}` stands.
const NAMED_HEADERS: &[(&str, &[&str])] = &[(
    "en",
    &[
        "On 15 Sep 2026, at 09:12, {} wrote:",
        "-----Original Message-----\nFrom: {}\nSent: Monday 15 September 2026 09:12",
    ],
)];

/// Headers above a quoted reply that name no one.
const UNNAMED_HEADERS: &[(&str, &[&str])] = &[(
    "en",
    &[
        "-----Original Message-----",
        "On 15 Sep 2026, at 09:12, you wrote:",
    ],
)];

/// An English sentence for a US document.
pub fn sentence_local(_country: &str, rng: &mut ChaCha8Rng) -> &'static str {
    english(rng)
}

/// A neutral sentence for a US document.
pub fn sentence(_country: &str, rng: &mut ChaCha8Rng) -> &'static str {
    english(rng)
}

fn english(rng: &mut ChaCha8Rng) -> &'static str {
    if rng.random_bool(0.3) {
        pick(EN_NUMERIC, rng)
    } else {
        pick(templates::FILLER_SENTENCES, rng)
    }
}

fn line(country: &str, rng: &mut ChaCha8Rng) -> &'static str {
    localized(LINES, country, 0.6, rng)
}

fn pick(items: &[&'static str], rng: &mut ChaCha8Rng) -> &'static str {
    items.choose(rng).copied().unwrap_or("")
}

/// A block of filler: a paragraph of one to four sentences in one language, one to three
/// lines, one to three negative lines (see `negatives`), or now and then a few lines of code.
fn block(country: &str, rng: &mut ChaCha8Rng) -> String {
    let n = rng.random_range(1..=4);
    if rng.random_bool(0.03) {
        return negatives::code_block(rng);
    }
    if rng.random_bool(0.2) {
        return (0..n.min(3))
            .map(|_| negatives::line(country, rng))
            .collect::<Vec<_>>()
            .join("\n");
    }
    if rng.random_bool(0.4) {
        let n = n.min(3);
        return (0..n)
            .map(|_| line(country, rng))
            .collect::<Vec<_>>()
            .join("\n");
    }
    (0..n).map(|_| english(rng)).collect::<Vec<_>>().join(" ")
}

/// Filler to wrap one document in.
pub struct Wrap {
    pub before: Vec<String>,
    pub after: Vec<String>,
    /// A reply header to quote the document under, with the author's name when it has a `{}`.
    pub quote: Option<(&'static str, Option<String>)>,
}

impl Wrap {
    /// Draws the filler for a `country` document: 45% get blocks before, 35% after, and 12%
    /// become a quoted reply, named after `person` half the time.
    pub fn draw(country: &str, person: &str, rng: &mut ChaCha8Rng) -> Wrap {
        let before = if rng.random_bool(0.45) {
            (0..rng.random_range(1..=3))
                .map(|_| block(country, rng))
                .collect()
        } else {
            Vec::new()
        };
        let after = if rng.random_bool(0.35) {
            (0..rng.random_range(1..=2))
                .map(|_| block(country, rng))
                .collect()
        } else {
            Vec::new()
        };
        let quote = rng.random_bool(0.12).then(|| {
            let named = rng.random_bool(0.5);
            let lists = if named {
                NAMED_HEADERS
            } else {
                UNNAMED_HEADERS
            };
            (
                localized(lists, country, 0.5, rng),
                named.then(|| person.to_string()),
            )
        });
        Wrap {
            before,
            after,
            quote,
        }
    }

    /// `doc` inside the filler, with every span shifted to stay on its text. A quoted reply
    /// prefixes each line with `> `, so it is applied only when no span crosses a line break;
    /// its header's person, when named, becomes an unlinked person entity.
    pub fn apply(self, mut doc: Doc) -> Doc {
        let crosses = doc
            .entities
            .iter()
            .any(|e| doc.text[e.start..e.end].contains('\n'));
        let mut text = String::new();
        for b in &self.before {
            text.push_str(b);
            text.push_str("\n\n");
        }
        let mut header_person = None;
        let quoting = match self.quote {
            Some((header, name)) if !crosses => {
                match (header.split_once("{}"), name) {
                    (Some((head, tail)), Some(name)) => {
                        text.push_str(head);
                        let start = text.len();
                        text.push_str(&name);
                        header_person = Some((start, text.len()));
                        text.push_str(tail);
                    }
                    _ => text.push_str(header),
                }
                text.push('\n');
                true
            }
            _ => false,
        };
        let base = text.len();
        let mut shifted: Vec<Gold> = Vec::with_capacity(doc.entities.len());
        if quoting {
            // Each line gains two bytes before it; a span on line k moves by 2 × (k + 1).
            let mut line_starts = vec![0];
            line_starts.extend(doc.text.match_indices('\n').map(|(i, _)| i + 1));
            let shift = |at: usize| 2 * line_starts.partition_point(|&s| s <= at);
            for e in &doc.entities {
                shifted.push(Gold {
                    kind: e.kind,
                    start: base + e.start + shift(e.start),
                    end: base + e.end + shift(e.start),
                });
            }
            let quoted: Vec<String> = doc.text.split('\n').map(|l| format!("> {l}")).collect();
            text.push_str(&quoted.join("\n"));
        } else {
            for e in &doc.entities {
                shifted.push(Gold {
                    kind: e.kind,
                    start: base + e.start,
                    end: base + e.end,
                });
            }
            text.push_str(&doc.text);
        }
        for a in &self.after {
            text.push_str("\n\n");
            text.push_str(a);
        }
        if let Some((start, end)) = header_person {
            shifted.insert(
                0,
                Gold {
                    kind: "person",
                    start,
                    end,
                },
            );
            for link in &mut doc.links {
                link.index += 1;
            }
            doc.links.insert(
                0,
                Link {
                    kind: "person",
                    index: 0,
                    group: 0,
                },
            );
        }
        doc.text = text;
        doc.entities = shifted;
        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{Category, Family};
    use rand::SeedableRng;

    fn doc() -> Doc {
        let text = "Regards,\nAnna Schmidt\nanna@x.example".to_string();
        Doc {
            entities: vec![
                Gold {
                    kind: "person",
                    start: 9,
                    end: 21,
                },
                Gold {
                    kind: "email",
                    start: 22,
                    end: 36,
                },
            ],
            links: vec![
                Link {
                    kind: "person",
                    index: 0,
                    group: 1,
                },
                Link {
                    kind: "email",
                    index: 1,
                    group: 1,
                },
            ],
            text,
            family: Family::Signature,
            template_id: 202,
            category: Category::Both,
            country: "DE",
            phones_fixture_safe: true,
        }
    }

    fn texts(d: &Doc) -> Vec<&str> {
        d.entities.iter().map(|e| &d.text[e.start..e.end]).collect()
    }

    #[test]
    fn padding_keeps_every_span_on_its_text() {
        let wrap = Wrap {
            before: vec!["Seite 1 von 2".into()],
            after: vec!["Vielen Dank.".into()],
            quote: None,
        };
        let d = wrap.apply(doc());
        assert_eq!(texts(&d), ["Anna Schmidt", "anna@x.example"]);
        assert!(d.text.starts_with("Seite 1 von 2\n\nRegards,"));
        assert!(d.text.ends_with("anna@x.example\n\nVielen Dank."));
    }

    #[test]
    fn a_quoted_reply_shifts_spans_per_line_and_adds_the_named_author() {
        let wrap = Wrap {
            before: Vec::new(),
            after: Vec::new(),
            quote: Some((
                "Am 12.10.2026 um 09:12 schrieb {}:",
                Some("Jonas Weber".into()),
            )),
        };
        let d = wrap.apply(doc());
        assert_eq!(texts(&d), ["Jonas Weber", "Anna Schmidt", "anna@x.example"]);
        assert!(
            d.text
                .contains("\n> Regards,\n> Anna Schmidt\n> anna@x.example")
        );
        assert_eq!(
            d.links
                .iter()
                .map(|l| (l.index, l.group))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 1), (2, 1)]
        );
    }

    #[test]
    fn a_span_across_lines_is_never_quoted() {
        let mut d = doc();
        d.entities[0].start = 0;
        let wrap = Wrap {
            before: Vec::new(),
            after: Vec::new(),
            quote: Some(("-----Original Message-----", None)),
        };
        let d = wrap.apply(d);
        assert!(!d.text.contains("> "));
        assert_eq!(texts(&d)[1], "anna@x.example");
    }

    /// Every text the generator draws and that is specific enough to recognize, as opposed to
    /// greetings, closings, titles, and the hard-negative names the fixtures are asked to test.
    fn specific_pool_texts() -> Vec<&'static str> {
        let lists: [&[&str]; 9] = [
            EN_NUMERIC,
            templates::FILLER_SENTENCES,
            templates::NEG_DIGITS,
            templates::NEG_PRICES,
            templates::NEG_ORDERS,
            templates::NEG_ROAD_SENTENCES,
            templates::NEG_PARTIAL_LOCATIONS,
            templates::PRODUCTS,
            templates::DATES,
        ];
        let localized = LINES
            .iter()
            .chain(NAMED_HEADERS)
            .chain(UNNAMED_HEADERS)
            .chain(templates::NEG_HEADERS)
            .chain(templates::NEG_DEPARTMENTS)
            .chain(templates::NEG_PROMPTS)
            .chain(templates::NEG_HEADINGS)
            .flat_map(|(_, list)| list.iter());
        lists
            .into_iter()
            .flatten()
            .chain(localized)
            .copied()
            .filter(|t| t.chars().count() >= 8 && *t != "-----Original Message-----")
            .collect()
    }

    /// The demo documents and fixtures measure the model, so none may contain training text.
    #[test]
    fn no_training_text_appears_in_the_demo_documents_or_fixtures() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut held_out = vec![std::fs::read_to_string(root.join("site/src/samples.rs")).unwrap()];
        for entry in std::fs::read_dir(root.join("fixtures/detector")).unwrap() {
            held_out.push(std::fs::read_to_string(entry.unwrap().path()).unwrap());
        }
        let held_out = held_out.join("\n").to_lowercase();
        let found: Vec<&str> = specific_pool_texts()
            .into_iter()
            .filter(|t| {
                let t = t.to_lowercase();
                held_out.contains(t.trim_end_matches(['.', '。']))
            })
            .collect();
        assert!(
            found.is_empty(),
            "training text in held-out documents: {found:?}"
        );
    }

    #[test]
    fn drawn_filler_has_no_braces_and_every_language_list_is_used() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for country in ["US"] {
            for _ in 0..200 {
                let w = Wrap::draw(country, "Anna Schmidt", &mut rng);
                for b in w.before.iter().chain(&w.after) {
                    assert!(!b.contains('{') && !b.trim().is_empty(), "{b:?}");
                }
            }
        }
        for (country, list) in LINES.iter().chain(UNNAMED_HEADERS) {
            assert!(!list.is_empty(), "{country}");
        }
        for (country, list) in NAMED_HEADERS {
            assert!(list.iter().all(|h| h.contains("{}")), "{country}");
        }
        for (country, list) in UNNAMED_HEADERS {
            assert!(list.iter().all(|h| !h.contains("{}")), "{country}");
        }
    }
}
