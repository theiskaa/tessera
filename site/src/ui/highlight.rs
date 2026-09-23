//! Syntax colouring for the code on the page: JSON for the output view and a little JavaScript
//! for the snippet. Each token is a span around its own text, so copying still gives plain text.

use leptos::prelude::*;

/// What a token is, which picks its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    Plain,
    Key,
    Str,
    Num,
    Lit,
    Keyword,
    Punct,
    Comment,
}

impl Class {
    fn css(self) -> Option<&'static str> {
        match self {
            Class::Plain => None,
            Class::Key => Some("t-key"),
            Class::Str => Some("t-str"),
            Class::Num => Some("t-num"),
            Class::Lit => Some("t-lit"),
            Class::Keyword => Some("t-kw"),
            Class::Punct => Some("t-punct"),
            Class::Comment => Some("t-com"),
        }
    }
}

/// Collects token ranges, merging neighbours of one class so the page gets fewer spans.
#[derive(Default)]
struct Tokens {
    ranges: Vec<(Class, usize, usize)>,
}

impl Tokens {
    fn push(&mut self, class: Class, start: usize, end: usize) {
        match self.ranges.last_mut() {
            _ if start >= end => {}
            Some(last) if last.0 == class && last.2 == start => last.2 = end,
            _ => self.ranges.push((class, start, end)),
        }
    }

    fn finish(self, text: &str) -> Vec<(Class, &str)> {
        self.ranges
            .into_iter()
            .filter_map(|(class, start, end)| Some((class, text.get(start..end)?)))
            .collect()
    }
}

/// Byte index just past the end of `text[start..]`'s run of characters matching `keep`.
fn run(text: &str, start: usize, keep: impl Fn(char) -> bool) -> usize {
    text[start..]
        .char_indices()
        .find(|&(_, c)| !keep(c))
        .map_or(text.len(), |(i, _)| start + i)
}

/// Byte index just past the string literal opening at `start`, or the end of its line when it
/// is not closed there.
fn string_end(text: &str, start: usize, quote: char) -> usize {
    let mut escaped = false;
    for (i, c) in text[start + quote.len_utf8()..].char_indices() {
        let at = start + quote.len_utf8() + i;
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '\n' => return at,
            c if c == quote => return at + c.len_utf8(),
            _ => {}
        }
    }
    text.len()
}

fn is_number(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')
}

/// JSON tokens: keys, strings, numbers, `true`/`false`/`null`, and structural punctuation.
pub(crate) fn json(text: &str) -> Vec<(Class, &str)> {
    let mut tokens = Tokens::default();
    let mut at = 0;
    while let Some(c) = text[at..].chars().next() {
        let end = match c {
            '"' => {
                let end = string_end(text, at, '"');
                let next = text[end..].trim_start_matches([' ', '\t']).chars().next();
                let class = if next == Some(':') {
                    Class::Key
                } else {
                    Class::Str
                };
                tokens.push(class, at, end);
                end
            }
            '-' | '0'..='9' => {
                let end = run(text, at + 1, is_number);
                tokens.push(Class::Num, at, end);
                end
            }
            c if c.is_ascii_alphabetic() => {
                let end = run(text, at, |c| c.is_ascii_alphanumeric());
                let class = match &text[at..end] {
                    "true" | "false" | "null" => Class::Lit,
                    _ => Class::Plain,
                };
                tokens.push(class, at, end);
                end
            }
            '{' | '}' | '[' | ']' | ':' | ',' => {
                tokens.push(Class::Punct, at, at + 1);
                at + 1
            }
            c => {
                tokens.push(Class::Plain, at, at + c.len_utf8());
                at + c.len_utf8()
            }
        };
        at = end;
    }
    tokens.finish(text)
}

const KEYWORDS: &[&str] = &[
    "async", "await", "const", "default", "else", "export", "from", "function", "if", "import",
    "let", "new", "return", "var",
];

/// JavaScript tokens, enough for the snippet: comments, strings, numbers, keywords, literals,
/// and punctuation. Identifiers stay plain.
pub(crate) fn javascript(text: &str) -> Vec<(Class, &str)> {
    let mut tokens = Tokens::default();
    let mut at = 0;
    while let Some(c) = text[at..].chars().next() {
        let end = match c {
            // A comment keeps its line break, so hiding comments removes their lines.
            '/' if text[at..].starts_with("//") => {
                let end = run(text, at, |c| c != '\n');
                let end = end + usize::from(text[end..].starts_with('\n'));
                tokens.push(Class::Comment, at, end);
                end
            }
            '\'' | '"' | '`' => {
                let end = string_end(text, at, c);
                tokens.push(Class::Str, at, end);
                end
            }
            '0'..='9' => {
                let end = run(text, at, |c| c.is_ascii_alphanumeric() || c == '.');
                tokens.push(Class::Num, at, end);
                end
            }
            c if c.is_ascii_alphabetic() || c == '_' || c == '$' => {
                let end = run(text, at, |c| {
                    c.is_ascii_alphanumeric() || c == '_' || c == '$'
                });
                let word = &text[at..end];
                let class = if KEYWORDS.contains(&word) {
                    Class::Keyword
                } else if matches!(word, "true" | "false" | "null" | "undefined") {
                    Class::Lit
                } else {
                    Class::Plain
                };
                tokens.push(class, at, end);
                end
            }
            c if "{}()[].,;:=|?<>+-*/!&".contains(c) => {
                tokens.push(Class::Punct, at, at + 1);
                at + 1
            }
            c => {
                tokens.push(Class::Plain, at, at + c.len_utf8());
                at + c.len_utf8()
            }
        };
        at = end;
    }
    tokens.finish(text)
}

/// The tokens as text nodes and coloured spans.
pub(crate) fn render(tokens: Vec<(Class, &str)>) -> impl IntoView + use<> {
    tokens
        .into_iter()
        .map(|(class, text)| {
            let text = text.to_string();
            match class.css() {
                None => text.into_any(),
                Some(css) => view! { <span class=css>{text}</span> }.into_any(),
            }
        })
        .collect_view()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(tokens: &[(Class, &str)]) -> String {
        tokens.iter().map(|(_, t)| *t).collect()
    }

    fn classes<'a>(tokens: &[(Class, &'a str)], class: Class) -> Vec<&'a str> {
        tokens
            .iter()
            .filter(|(c, _)| *c == class)
            .map(|(_, t)| *t)
            .collect()
    }

    #[test]
    fn json_tokens_cover_the_text_and_tell_keys_from_strings() {
        let text = "[\n  {\n    \"kind\": \"phone\",\n    \"start\": 866,\n    \"x\": -1.5e3,\n    \"ok\": true, \"none\": null, \"q\": \"a\\\"b\"\n  }\n]";
        let tokens = json(text);
        assert_eq!(joined(&tokens), text);
        assert_eq!(
            classes(&tokens, Class::Key),
            [
                "\"kind\"",
                "\"start\"",
                "\"x\"",
                "\"ok\"",
                "\"none\"",
                "\"q\""
            ]
        );
        assert_eq!(classes(&tokens, Class::Str), ["\"phone\"", "\"a\\\"b\""]);
        assert_eq!(classes(&tokens, Class::Num), ["866", "-1.5e3"]);
        assert_eq!(classes(&tokens, Class::Lit), ["true", "null"]);
    }

    #[test]
    fn javascript_tokens_find_keywords_strings_and_comments() {
        let text =
            "import { createTessera } from 'tessera'\nconst found = await t.detect(text) // “ok”\n";
        let tokens = javascript(text);
        assert_eq!(joined(&tokens), text);
        assert_eq!(
            classes(&tokens, Class::Keyword),
            ["import", "from", "const", "await"]
        );
        assert_eq!(classes(&tokens, Class::Str), ["'tessera'"]);
        assert_eq!(classes(&tokens, Class::Comment), ["// “ok”\n"]);
    }

    #[test]
    fn unclosed_strings_stop_at_the_line_end() {
        let tokens = javascript("'open\nnext");
        assert_eq!(tokens[0], (Class::Str, "'open"));
        assert_eq!(joined(&tokens), "'open\nnext");
    }
}
