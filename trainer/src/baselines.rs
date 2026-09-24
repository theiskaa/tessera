//! Deterministic baselines: a rule-based address parser, entity detector, and contact
//! grouper built from the library's own token features. They are the floor the learned
//! models must beat, and the reference for what the deterministic signals can do alone.
//!
//! Nothing here ships. The grouper's rules move into the library in Milestone 5 once
//! these numbers say they are worth trusting.

use tessera::internal::{
    FeatureConfig, Script, Token, TokenClass, TokenFeatures, featurize, flag, is_content,
    line_ranges, scan_rules, tokenize,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub kind: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comp {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContactIdx {
    pub person: Option<usize>,
    pub org: Option<usize>,
    pub addresses: Vec<usize>,
    pub emails: Vec<usize>,
    pub phones: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Assignment {
    pub contacts: Vec<ContactIdx>,
    pub unassigned: Vec<usize>,
}

const ROAD_SUFFIXES: &[&str] = &[
    "straße", "strasse", "str.", "str", "weg", "gasse", "allee", "platz", "damm", "ufer", "straat",
    "laan", "plein", "gracht", "kade", "dijk", "singel",
];
const ROAD_PARTICLES: &[&str] = &[
    "den", "der", "de", "van", "am", "an", "auf", "im", "zum", "zur",
];
const NAME_PARTICLES: &[&str] = &["van", "der", "de", "von", "di", "da", "la", "le", "du"];
const ORG_PARTICLES: &[&str] = &["&", "and", "of", "de", "und"];
const JA_REGION_SUFFIX: &[char] = &['都', '道', '府', '県'];
const JA_CITY_SUFFIX: &[char] = &['市', '区'];
const JA_SUBURB_SUFFIX: &[char] = &['町', '村'];
const MAX_ROAD_PREFIX_TOKENS: usize = 3;
const MAX_ORG_TOKENS: usize = 5;

struct Doc<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    feats: Vec<TokenFeatures>,
}

impl<'a> Doc<'a> {
    /// Tokens only, with empty features, for callers that never read flags.
    fn plain(text: &'a str) -> Doc<'a> {
        let tokens = tokenize(text);
        let feats = vec![TokenFeatures::default(); tokens.len()];
        Doc {
            text,
            tokens,
            feats,
        }
    }

    fn new(text: &'a str, country: Option<&str>, rule_spans: &[(usize, usize)]) -> Doc<'a> {
        let tokens = tokenize(text);
        let feats = featurize(
            text,
            &tokens,
            rule_spans,
            country,
            &FeatureConfig::default(),
            None,
        );
        Doc {
            text,
            tokens,
            feats,
        }
    }

    fn s(&self, i: usize) -> &'a str {
        &self.text[self.tokens[i].start..self.tokens[i].end]
    }

    fn lower(&self, i: usize) -> String {
        self.s(i).chars().flat_map(char::to_lowercase).collect()
    }

    fn has(&self, i: usize, bit: u32) -> bool {
        self.feats[i].flags & bit != 0
    }

    fn is_content(&self, i: usize) -> bool {
        is_content(&self.tokens[i])
    }

    fn prev_content(&self, i: usize) -> Option<usize> {
        (0..i).rev().find(|&k| self.is_content(k))
    }

    fn next_content(&self, i: usize) -> Option<usize> {
        (i + 1..self.tokens.len()).find(|&k| self.is_content(k))
    }

    fn is_title(&self, i: usize) -> bool {
        self.tokens[i].class == TokenClass::Alpha
            && (self.has(i, flag::TITLE_CASE) || self.has(i, flag::ALL_UPPER))
    }

    fn is_break(&self, i: usize) -> bool {
        matches!(self.tokens[i].class, TokenClass::Newline) || matches!(self.s(i), "," | ";")
    }

    fn lines(&self) -> Vec<(usize, usize)> {
        line_ranges(&self.tokens)
    }

    fn content_of(&self, a: usize, b: usize) -> Vec<usize> {
        (a..b).filter(|&k| self.is_content(k)).collect()
    }
}

fn span_of(doc: &Doc<'_>, first: usize, last: usize) -> (usize, usize) {
    (doc.tokens[first].start, doc.tokens[last].end)
}

/// Component labels for one address string, by the precedence in the Milestone 1 plan.
pub fn parse_address(text: &str, country: Option<&str>) -> Vec<Comp> {
    let doc = Doc::new(text, country, &[]);
    let mut labels: Vec<Option<&'static str>> = vec![None; doc.tokens.len()];
    label_postcode(&doc, &mut labels);
    label_country(&doc, &mut labels);
    label_unit(&doc, &mut labels);
    label_road(&doc, &mut labels, country);
    label_house_number(&doc, &mut labels, country);
    label_region(&doc, &mut labels, country);
    label_city(&doc, &mut labels, country);
    merge_components(&doc, &labels)
}

fn label_postcode(doc: &Doc<'_>, labels: &mut [Option<&'static str>]) {
    for (i, label) in labels.iter_mut().enumerate() {
        if doc.has(i, flag::POSTCODE_LIKE) {
            *label = Some("postcode");
        }
    }
}

fn label_country(doc: &Doc<'_>, labels: &mut [Option<&'static str>]) {
    let n = doc.tokens.len();
    let mut i = 0;
    while i < n {
        if labels[i].is_some() || !doc.is_content(i) || !doc.has(i, flag::COUNTRY_TERM) {
            i += 1;
            continue;
        }
        // Multi-word countries ("United Kingdom") carry the flag on every word.
        let mut last = i;
        let mut k = i + 1;
        while k < n {
            if doc.tokens[k].class == TokenClass::Space {
                k += 1;
            } else if labels[k].is_none() && doc.is_content(k) && doc.has(k, flag::COUNTRY_TERM) {
                last = k;
                k += 1;
            } else {
                break;
            }
        }
        // Only at the end of its line, so "Georgia" in prose is not a country.
        if at_line_tail(doc, last) {
            for (k, slot) in labels.iter_mut().enumerate().take(last + 1).skip(i) {
                if doc.is_content(k) {
                    *slot = Some("country");
                }
            }
        }
        i = last + 1;
    }
}

/// Whether only spaces and trailing `,` `.` `;` follow token `i` before the line ends.
fn at_line_tail(doc: &Doc<'_>, i: usize) -> bool {
    for k in i + 1..doc.tokens.len() {
        match doc.tokens[k].class {
            TokenClass::Space => continue,
            TokenClass::Newline => return true,
            _ if matches!(doc.s(k), "," | "." | ";") => continue,
            _ => return false,
        }
    }
    true
}

fn label_unit(doc: &Doc<'_>, labels: &mut [Option<&'static str>]) {
    for i in 0..doc.tokens.len() {
        if labels[i].is_some() || !doc.has(i, flag::UNIT_TERM) {
            continue;
        }
        let Some(k) = doc.next_content(i) else {
            continue;
        };
        if labels[k].is_none()
            && matches!(doc.tokens[k].class, TokenClass::Digit | TokenClass::Alnum)
        {
            labels[i] = Some("unit");
            labels[k] = Some("unit");
        }
    }
}

fn label_road(doc: &Doc<'_>, labels: &mut [Option<&'static str>], country: Option<&str>) {
    for i in 0..doc.tokens.len() {
        if labels[i].is_some() || !doc.is_content(i) {
            continue;
        }
        let script = doc.tokens[i].script;
        let term = doc.has(i, flag::ROAD_TERM);
        let lower = doc.lower(i);
        let suffix = doc.tokens[i].class == TokenClass::Alpha
            && ROAD_SUFFIXES
                .iter()
                .any(|s| lower.ends_with(s) && lower.len() > s.len());
        if !term && !suffix {
            continue;
        }
        labels[i] = Some("road");
        let back = if suffix { 2 } else { MAX_ROAD_PREFIX_TOKENS };
        let back = if script == Script::Georgian { 1 } else { back };
        let mut cursor = i;
        for _ in 0..back {
            let Some(p) = doc.prev_content(cursor) else {
                break;
            };
            if labels[p].is_some() || doc.is_break(p) {
                break;
            }
            let joins = match script {
                Script::Georgian => {
                    doc.tokens[p].class == TokenClass::Alpha
                        && doc.tokens[p].script == Script::Georgian
                }
                _ => doc.is_title(p) || ROAD_PARTICLES.contains(&doc.lower(p).as_str()),
            };
            if !joins {
                break;
            }
            labels[p] = Some("road");
            cursor = p;
        }
        // A directional suffix stays inside the road: "Pennsylvania Avenue NW".
        if country != Some("JP")
            && let Some(k) = doc.next_content(i)
            && labels[k].is_none()
            && doc.tokens[k].class == TokenClass::Alpha
            && doc.has(k, flag::ALL_UPPER)
            && doc.s(k).chars().count() <= 2
        {
            labels[k] = Some("road");
        }
    }
}

fn label_house_number(doc: &Doc<'_>, labels: &mut [Option<&'static str>], country: Option<&str>) {
    if country == Some("JP") {
        label_ja_block_number(doc, labels);
        return;
    }
    for i in 0..doc.tokens.len() {
        if labels[i].is_some() || !doc.has(i, flag::HOUSE_NUMBER_LIKE) {
            continue;
        }
        let touches_road = [doc.prev_content(i), doc.next_content(i)]
            .into_iter()
            .flatten()
            .any(|k| labels[k] == Some("road"));
        if touches_road {
            labels[i] = Some("house_number");
        }
    }
}

/// A trailing `9-9` or `9-9-9` run, the Japanese block and building number.
fn label_ja_block_number(doc: &Doc<'_>, labels: &mut [Option<&'static str>]) {
    let content = doc.content_of(0, doc.tokens.len());
    let mut run: Vec<usize> = Vec::new();
    for &i in content.iter().rev() {
        let ok = doc.tokens[i].class == TokenClass::Digit || doc.s(i) == "-";
        if ok && labels[i].is_none() {
            run.push(i);
        } else {
            break;
        }
    }
    run.reverse();
    if run.len() >= 3 {
        for i in run {
            labels[i] = Some("house_number");
        }
    }
}

fn label_region(doc: &Doc<'_>, labels: &mut [Option<&'static str>], country: Option<&str>) {
    if country == Some("JP") {
        label_ja_run(doc, labels, JA_REGION_SUFFIX, "region");
        return;
    }
    for i in 0..doc.tokens.len() {
        if labels[i].is_some() || !doc.is_content(i) {
            continue;
        }
        if doc.has(i, flag::REGION_TERM) {
            labels[i] = Some("region");
            continue;
        }
        // A two-letter code right before a postcode is a state or province.
        let before_postcode = doc
            .next_content(i)
            .is_some_and(|k| labels[k] == Some("postcode"));
        if before_postcode
            && doc.tokens[i].class == TokenClass::Alpha
            && doc.has(i, flag::ALL_UPPER)
            && doc.s(i).chars().count() == 2
        {
            labels[i] = Some("region");
        }
    }
}

fn label_city(doc: &Doc<'_>, labels: &mut [Option<&'static str>], country: Option<&str>) {
    if country == Some("JP") {
        label_ja_run(doc, labels, JA_CITY_SUFFIX, "city");
        label_ja_run(doc, labels, JA_SUBURB_SUFFIX, "suburb");
        label_ja_remaining_run(doc, labels, "suburb");
        return;
    }
    let last_road =
        (0..doc.tokens.len()).rfind(|&i| matches!(labels[i], Some("road") | Some("house_number")));
    let start = last_road.map_or(0, |i| i + 1);
    let mut run: Vec<usize> = Vec::new();
    for i in (start..doc.tokens.len()).filter(|&i| doc.is_content(i)) {
        let candidate = labels[i].is_none()
            && (doc.is_title(i)
                || (doc.tokens[i].class == TokenClass::Alpha
                    && doc.tokens[i].script == Script::Georgian));
        if candidate {
            run.push(i);
        } else if !run.is_empty() {
            break;
        }
    }
    for i in run {
        labels[i] = Some("city");
    }
}

/// Label a run of Han tokens ending in one of `suffix` (for example `東京都`).
fn label_ja_run(
    doc: &Doc<'_>,
    labels: &mut [Option<&'static str>],
    suffix: &[char],
    label: &'static str,
) {
    let content = doc.content_of(0, doc.tokens.len());
    for (pos, &i) in content.iter().enumerate() {
        let ends = doc.s(i).chars().next().is_some_and(|c| suffix.contains(&c));
        if !ends || labels[i].is_some() {
            continue;
        }
        let mut first = pos;
        while first > 0 {
            let k = content[first - 1];
            if labels[k].is_some() || doc.tokens[k].class != TokenClass::Alpha {
                break;
            }
            first -= 1;
        }
        for &k in &content[first..=pos] {
            labels[k] = Some(label);
        }
    }
}

/// The remaining unlabelled alphabetic run before the block number.
fn label_ja_remaining_run(doc: &Doc<'_>, labels: &mut [Option<&'static str>], label: &'static str) {
    let content = doc.content_of(0, doc.tokens.len());
    let mut run = Vec::new();
    for &i in &content {
        if labels[i].is_none() && doc.tokens[i].class == TokenClass::Alpha {
            run.push(i);
        } else if !run.is_empty() {
            break;
        }
    }
    for i in run {
        labels[i] = Some(label);
    }
}

/// Adjacent tokens with the same label become one component; a comma or newline ends it.
fn merge_components(doc: &Doc<'_>, labels: &[Option<&'static str>]) -> Vec<Comp> {
    let mut out: Vec<Comp> = Vec::new();
    let mut open: Option<(&'static str, usize, usize)> = None;
    for i in 0..doc.tokens.len() {
        match labels[i] {
            Some(label) => match open {
                Some((prev, first, _)) if prev == label => open = Some((label, first, i)),
                _ => {
                    if let Some((label, first, last)) = open.take() {
                        let (s, e) = span_of(doc, first, last);
                        out.push(Comp {
                            label: label.to_string(),
                            start: s,
                            end: e,
                        });
                    }
                    open = Some((label, i, i));
                }
            },
            None => {
                let joins = doc.tokens[i].class == TokenClass::Space
                    && open.is_some()
                    && doc
                        .next_content(i)
                        .is_some_and(|k| labels[k] == open.map(|(l, _, _)| l));
                if !joins && let Some((label, first, last)) = open.take() {
                    let (s, e) = span_of(doc, first, last);
                    out.push(Comp {
                        label: label.to_string(),
                        start: s,
                        end: e,
                    });
                }
            }
        }
    }
    if let Some((label, first, last)) = open {
        let (s, e) = span_of(doc, first, last);
        out.push(Comp {
            label: label.to_string(),
            start: s,
            end: e,
        });
    }
    out.sort_by_key(|c| (c.start, c.end));
    out
}

/// Person, organization, and address spans, plus the rules layer's emails and phones.
pub fn detect(text: &str, country: Option<&str>) -> Vec<Span> {
    let hints: Vec<&str> = country.into_iter().collect();
    let rules = scan_rules(text, &hints);
    let rule_spans: Vec<(usize, usize)> = rules.iter().map(|e| (e.start, e.end)).collect();
    let doc = Doc::new(text, country, &rule_spans);

    let mut spans: Vec<Span> = Vec::new();
    spans.extend(detect_addresses(&doc));
    spans.extend(detect_orgs(&doc));
    spans.extend(detect_persons(&doc));
    // Address over org over person; a later span that overlaps an earlier one is dropped.
    let mut kept: Vec<Span> = Vec::new();
    for s in spans {
        let clashes = kept.iter().any(|k| s.start < k.end && k.start < s.end)
            || rule_spans.iter().any(|&(a, b)| s.start < b && a < s.end);
        if !clashes {
            kept.push(s);
        }
    }
    kept.extend(rules.iter().map(|e| Span {
        kind: e.kind.as_str().to_string(),
        start: e.start,
        end: e.end,
    }));
    kept.sort_by_key(|s| (s.start, s.end));
    kept
}

fn detect_addresses(doc: &Doc<'_>) -> Vec<Span> {
    let lines = doc.lines();
    let mut out = Vec::new();
    let mut line = 0;
    while line < lines.len() {
        let (a, b) = lines[line];
        let content = doc.content_of(a, b);
        if content.is_empty() || content.iter().any(|&i| doc.has(i, flag::IN_RULE_SPAN)) {
            line += 1;
            continue;
        }
        let has_postcode = content.iter().any(|&i| doc.has(i, flag::POSTCODE_LIKE));
        let house_then_road = content.iter().enumerate().any(|(pos, &i)| {
            doc.has(i, flag::HOUSE_NUMBER_LIKE)
                && content
                    .iter()
                    .skip(pos + 1)
                    .take(4)
                    .any(|&k| doc.has(k, flag::ROAD_TERM))
        });
        if !has_postcode && !house_then_road {
            line += 1;
            continue;
        }
        let (mut first, mut last) = (content[0], *content.last().unwrap());
        // A continuation line holding only a postcode or a country belongs to the same address.
        if let Some(&(na, nb)) = lines.get(line + 1) {
            let next = doc.content_of(na, nb);
            let continues = !next.is_empty()
                && next
                    .iter()
                    .any(|&i| doc.has(i, flag::POSTCODE_LIKE) || doc.has(i, flag::COUNTRY_TERM))
                && !next.iter().any(|&i| doc.has(i, flag::IN_RULE_SPAN));
            if continues {
                last = *next.last().unwrap();
                line += 1;
            }
        }
        if first > last {
            std::mem::swap(&mut first, &mut last);
        }
        let (s, e) = span_of(doc, first, last);
        out.push(Span {
            kind: "address".into(),
            start: s,
            end: e,
        });
        line += 1;
    }
    out
}

fn detect_orgs(doc: &Doc<'_>) -> Vec<Span> {
    let lines = doc.lines();
    let mut out = Vec::new();
    for &(a, b) in &lines {
        let content = doc.content_of(a, b);
        let Some(pos) = content.iter().position(|&i| doc.has(i, flag::LEGAL_FORM)) else {
            continue;
        };
        let i = content[pos];
        let (first, last) = if pos == 0 {
            // A leading legal form runs forwards: "შპს Kavkaz Freight".
            let mut last = i;
            for &k in content.iter().skip(1).take(MAX_ORG_TOKENS) {
                if doc.tokens[k].class != TokenClass::Alpha {
                    break;
                }
                last = k;
            }
            (i, last)
        } else {
            let mut first = i;
            for &k in content[..pos].iter().rev().take(MAX_ORG_TOKENS) {
                let lower = doc.lower(k);
                if !(doc.is_title(k) || ORG_PARTICLES.contains(&lower.as_str())) {
                    break;
                }
                first = k;
            }
            (first, i)
        };
        if first == last && doc.tokens[first].class != TokenClass::Alpha {
            continue;
        }
        let (s, e) = span_of(doc, first, last);
        out.push(Span {
            kind: "org".into(),
            start: s,
            end: e,
        });
    }
    out
}

fn detect_persons(doc: &Doc<'_>) -> Vec<Span> {
    let lines = doc.lines();
    let mut out = Vec::new();
    for (index, &(a, b)) in lines.iter().enumerate() {
        let content = doc.content_of(a, b);
        if content.is_empty() {
            continue;
        }
        let salutation_line = content
            .iter()
            .any(|&i| doc.has(i, flag::SALUTATION) || doc.has(i, flag::CLOSING));
        // A name after an honorific: "Dr Smith", "Dr. Jane Smith". The honorific may span
        // several tokens once the tokenizer splits off its dot.
        if let Some(pos) = content.iter().position(|&i| doc.has(i, flag::HONORIFIC)) {
            let names: Vec<usize> = content
                .iter()
                .skip(pos + 1)
                .copied()
                .skip_while(|&k| doc.s(k) == "." || doc.has(k, flag::HONORIFIC))
                .take(2)
                .take_while(|&k| doc.is_title(k))
                .collect();
            if let (Some(&f), Some(&l)) = (names.first(), names.last()) {
                let (s, e) = span_of(doc, f, l);
                out.push(Span {
                    kind: "person".into(),
                    start: s,
                    end: e,
                });
                continue;
            }
        }
        if salutation_line {
            // The line after a closing usually holds the sender's name.
            if let Some(&(na, nb)) = lines.get(index + 1)
                && let Some(span) = name_run(doc, &doc.content_of(na, nb))
            {
                out.push(span);
            }
            continue;
        }
        if content
            .iter()
            .all(|&i| doc.has(i, flag::IN_SHORT_LINE_BLOCK))
            && let Some(span) = name_run(doc, &content)
        {
            out.push(span);
        }
    }
    out
}

/// Two or three name tokens, allowing one lowercase particle between them.
fn name_run(doc: &Doc<'_>, content: &[usize]) -> Option<Span> {
    if content
        .iter()
        .any(|&i| doc.has(i, flag::LEGAL_FORM) || doc.has(i, flag::ROAD_TERM))
    {
        return None;
    }
    let georgian = content.iter().all(|&i| {
        doc.tokens[i].class == TokenClass::Alpha && doc.tokens[i].script == Script::Georgian
    });
    let mut run: Vec<usize> = Vec::new();
    for &i in content {
        let ok = if georgian {
            doc.tokens[i].class == TokenClass::Alpha
        } else {
            doc.is_title(i) || (!run.is_empty() && NAME_PARTICLES.contains(&doc.lower(i).as_str()))
        };
        if ok {
            run.push(i);
        } else if !run.is_empty() {
            break;
        }
    }
    while run.last().is_some_and(|&i| !doc.is_title(i) && !georgian) {
        run.pop();
    }
    if !(2..=4).contains(&run.len()) {
        return None;
    }
    let (s, e) = span_of(doc, run[0], *run.last().unwrap());
    Some(Span {
        kind: "person".into(),
        start: s,
        end: e,
    })
}

/// Group entities into contacts by block, anchoring on people and falling back to orgs.
pub fn group(text: &str, entities: &[Span]) -> Assignment {
    let doc = Doc::plain(text);
    let blocks = blocks(&doc);
    let block_of = |s: &Span| {
        blocks
            .iter()
            .position(|&(a, b)| s.start >= a && s.start < b)
            .unwrap_or(0)
    };

    let mut out = Assignment::default();
    let mut contact_of: Vec<Option<usize>> = vec![None; entities.len()];
    for (block, _) in blocks.iter().enumerate() {
        let here: Vec<usize> = (0..entities.len())
            .filter(|&i| block_of(&entities[i]) == block)
            .collect();
        let people: Vec<usize> = here
            .iter()
            .copied()
            .filter(|&i| entities[i].kind == "person")
            .collect();
        let orgs: Vec<usize> = here
            .iter()
            .copied()
            .filter(|&i| entities[i].kind == "org")
            .collect();
        let anchors: Vec<usize> = if people.is_empty() {
            orgs.clone()
        } else {
            people.clone()
        };

        for &anchor in &anchors {
            let mut c = ContactIdx::default();
            if entities[anchor].kind == "person" {
                c.person = Some(anchor);
            } else {
                c.org = Some(anchor);
            }
            contact_of[anchor] = Some(out.contacts.len());
            out.contacts.push(c);
        }
        // An org joins the only person in its block.
        if !people.is_empty() {
            for &org in &orgs {
                if people.len() == 1 {
                    let idx = contact_of[people[0]].expect("anchor has a contact");
                    if out.contacts[idx].org.is_none() {
                        out.contacts[idx].org = Some(org);
                        contact_of[org] = Some(idx);
                        continue;
                    }
                }
                out.unassigned.push(org);
            }
        }

        for &i in &here {
            if contact_of[i].is_some() || matches!(entities[i].kind.as_str(), "person" | "org") {
                continue;
            }
            let Some(anchor) = pick_anchor(text, entities, &anchors, i) else {
                out.unassigned.push(i);
                continue;
            };
            let idx = contact_of[anchor].expect("anchor has a contact");
            match entities[i].kind.as_str() {
                "address" => out.contacts[idx].addresses.push(i),
                "email" => out.contacts[idx].emails.push(i),
                "phone" => out.contacts[idx].phones.push(i),
                _ => {}
            }
            contact_of[i] = Some(idx);
        }
    }
    out.unassigned.sort_unstable();
    out
}

/// The nearest preceding anchor, unless an email's local part names another person.
fn pick_anchor(text: &str, entities: &[Span], anchors: &[usize], i: usize) -> Option<usize> {
    if entities[i].kind == "email" {
        let local = text[entities[i].start..entities[i].end]
            .split('@')
            .next()
            .unwrap_or("")
            .to_lowercase();
        let parts: Vec<&str> = local
            .split(['.', '_', '-'])
            .filter(|p| p.len() >= 3)
            .collect();
        let named = anchors.iter().copied().find(|&a| {
            entities[a].kind == "person"
                && text[entities[a].start..entities[a].end]
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|w| w.len() >= 3 && parts.iter().any(|p| p.eq_ignore_ascii_case(w)))
        });
        if named.is_some() {
            return named;
        }
    }
    anchors
        .iter()
        .copied()
        .rfind(|&a| entities[a].start <= entities[i].start)
        .or_else(|| anchors.first().copied())
}

/// Byte ranges of the blocks a document splits into: blank lines, separator lines, quoted
/// replies, and mail headers all start a new one.
fn blocks(doc: &Doc<'_>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for (a, b) in doc.lines() {
        let content = doc.content_of(a, b);
        let separator = content.len() >= 3
            && content
                .iter()
                .all(|&i| matches!(doc.s(i), "-" | "_" | "=" | "*"));
        let header = content.first().is_some_and(|&i| {
            matches!(doc.lower(i).as_str(), "from" | "sent" | "to" | "subject") || doc.s(i) == ">"
        });
        if content.is_empty() || separator || header {
            out.extend(current.take());
            continue;
        }
        let line_start = doc.tokens[content[0]].start;
        let line_end = doc.tokens[content[content.len() - 1]].end;
        current = Some(current.map_or((line_start, line_end), |(s, _)| (s, line_end)));
    }
    out.extend(current);
    if out.is_empty() {
        out.push((0, doc.text.len()));
    }
    // Extend each block to the start of the next so every entity falls inside one.
    let next_starts: Vec<usize> = out.iter().skip(1).map(|&(s, _)| s).collect();
    out.iter()
        .enumerate()
        .map(|(i, &(s, e))| (s, next_starts.get(i).map_or(doc.text.len(), |&n| n.max(e))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_spec_gb_address() {
        let comps = parse_address("Flat 4, 221B Baker Street, London NW1 6XE, UK", Some("GB"));
        let got: Vec<(&str, &str)> = comps
            .iter()
            .map(|c| {
                (
                    c.label.as_str(),
                    &"Flat 4, 221B Baker Street, London NW1 6XE, UK"[c.start..c.end],
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("unit", "Flat 4"),
                ("house_number", "221B"),
                ("road", "Baker Street"),
                ("city", "London"),
                ("postcode", "NW1 6XE"),
                ("country", "UK"),
            ]
        );
    }

    #[test]
    fn detects_the_spec_signature() {
        let text = "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example\n";
        let spans = detect(text, Some("GE"));
        let got: Vec<(&str, &str)> = spans
            .iter()
            .map(|s| (s.kind.as_str(), &text[s.start..s.end]))
            .collect();
        assert!(got.contains(&("person", "Nino Beridze")), "{got:?}");
        assert!(got.contains(&("org", "Kavkaz Freight LLC")), "{got:?}");
        assert!(
            got.contains(&("address", "14 Rustaveli Avenue, Tbilisi 0108, Georgia")),
            "{got:?}"
        );
        assert!(
            !got.iter().any(|&(k, t)| k == "person" && t == "Monday"),
            "{got:?}"
        );
    }

    #[test]
    fn groups_two_signatures_separately() {
        let text = "Jane O'Brien\njane@acme.example\n\nJohn Smith\njohn@acme.example\n";
        let entities = vec![
            Span {
                kind: "person".into(),
                start: 0,
                end: 12,
            },
            Span {
                kind: "email".into(),
                start: 13,
                end: 30,
            },
            Span {
                kind: "person".into(),
                start: 32,
                end: 42,
            },
            Span {
                kind: "email".into(),
                start: 43,
                end: 60,
            },
        ];
        let a = group(text, &entities);
        assert_eq!(a.contacts.len(), 2, "{a:?}");
        assert_eq!(a.contacts[0].emails, vec![1]);
        assert_eq!(a.contacts[1].emails, vec![3]);
        assert!(a.unassigned.is_empty());
    }

    #[test]
    fn two_word_country_is_labelled() {
        let text = "10 Downing Street, London SW1A 2AA, United Kingdom";
        let got: Vec<(String, &str)> = parse_address(text, Some("GB"))
            .into_iter()
            .map(|c| (c.label.clone(), &text[c.start..c.end]))
            .collect();
        assert!(
            got.contains(&("country".into(), "United Kingdom")),
            "{got:?}"
        );
        let text = "Georgia and Jordan visited";
        assert!(
            parse_address(text, Some("US"))
                .iter()
                .all(|c| c.label != "country")
        );
    }

    #[test]
    fn honorific_with_a_dot_precedes_a_name() {
        let text = "Dr. Jane Smith\nAcme Ltd\n";
        let spans = detect(text, Some("GB"));
        let got: Vec<(&str, &str)> = spans
            .iter()
            .map(|s| (s.kind.as_str(), &text[s.start..s.end]))
            .collect();
        assert!(got.contains(&("person", "Jane Smith")), "{got:?}");
    }

    #[test]
    fn blocks_are_not_duplicated() {
        let doc = Doc::plain("a\nb\n\nc\n---\nd\n");
        let b = blocks(&doc);
        assert_eq!(b.len(), 3, "{b:?}");
        assert!(
            b.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].0),
            "{b:?}"
        );
    }

    #[test]
    fn email_local_part_breaks_the_tie() {
        let text = "Jane O'Brien and John Smith\njohn@acme.example\n";
        let entities = vec![
            Span {
                kind: "person".into(),
                start: 0,
                end: 12,
            },
            Span {
                kind: "person".into(),
                start: 17,
                end: 27,
            },
            Span {
                kind: "email".into(),
                start: 28,
                end: 45,
            },
        ];
        let a = group(text, &entities);
        let with_email = a
            .contacts
            .iter()
            .find(|c| !c.emails.is_empty())
            .expect("an email was assigned");
        assert_eq!(with_email.person, Some(1), "{a:?}");
    }
}
