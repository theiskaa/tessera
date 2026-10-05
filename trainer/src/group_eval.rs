//! Scores the contact grouper on the grouper fixtures: on the fixtures' gold entities, which
//! isolates grouping from detection, and end to end through `Tessera::extract_contacts` with a
//! bundle. Wrong and missed assignments are counted apart, because a wrong assignment is the
//! failure the grouper exists to prevent and an unassigned entity is the safe one.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tessera::{Config, Entity, Extraction, Kind, Query, Source, Tessera};

use crate::EvalArgs;
use crate::fixtures::{self, GrouperCase, GrouperFixture};

/// Counts of one fixture family, or of all of them.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct GrouperCounts {
    pub cases: usize,
    /// Gold entities, each owned by the contact it belongs to (an anchor by its own) or by none.
    pub assignments: usize,
    pub correct: usize,
    /// Given to a contact other than the gold owner, or to any contact when gold leaves it out.
    pub wrong: usize,
    /// Left unassigned although gold gives it an owner.
    pub missed: usize,
    pub gold_contacts: usize,
    /// Gold contacts whose predicted contact has exactly the same members.
    pub exact_contacts: usize,
    /// Predicted contacts whose anchor anchors no gold contact.
    pub false_contacts: usize,
    pub predicted_contacts: usize,
    pub exact_card_matches: usize,
    pub unmatched_predicted_contacts: usize,
    pub unmatched_gold_contacts: usize,
    pub false_card_documents: usize,
    pub false_assigned_fields: usize,
    pub false_assigned_by_kind: BTreeMap<String, usize>,
    pub wrong_ownership_fields: usize,
    pub high_confidence_predicted_cards: usize,
    pub high_confidence_exact_cards: usize,
    /// Exact entity matches, including fields left unassigned by the grouper.
    pub detection_by_kind: BTreeMap<String, crate::exact_metrics::Counts>,
    /// Exact `(contact anchor, kind, start, end)` matches for assigned fields.
    pub contact_fields_by_kind: BTreeMap<String, crate::exact_metrics::Counts>,
}

impl GrouperCounts {
    fn add(&mut self, other: &GrouperCounts) {
        self.cases += other.cases;
        self.assignments += other.assignments;
        self.correct += other.correct;
        self.wrong += other.wrong;
        self.missed += other.missed;
        self.gold_contacts += other.gold_contacts;
        self.exact_contacts += other.exact_contacts;
        self.false_contacts += other.false_contacts;
        self.predicted_contacts += other.predicted_contacts;
        self.exact_card_matches += other.exact_card_matches;
        self.unmatched_predicted_contacts += other.unmatched_predicted_contacts;
        self.unmatched_gold_contacts += other.unmatched_gold_contacts;
        self.false_card_documents += other.false_card_documents;
        self.false_assigned_fields += other.false_assigned_fields;
        self.wrong_ownership_fields += other.wrong_ownership_fields;
        self.high_confidence_predicted_cards += other.high_confidence_predicted_cards;
        self.high_confidence_exact_cards += other.high_confidence_exact_cards;
        for (kind, count) in &other.false_assigned_by_kind {
            *self.false_assigned_by_kind.entry(kind.clone()).or_default() += count;
        }
        for (kind, count) in &other.detection_by_kind {
            self.detection_by_kind
                .entry(kind.clone())
                .or_default()
                .add(count);
        }
        for (kind, count) in &other.contact_fields_by_kind {
            self.contact_fields_by_kind
                .entry(kind.clone())
                .or_default()
                .add(count);
        }
    }
}

/// Per fixture family: gold-span counts and end-to-end counts.
#[derive(Debug, Default, Serialize)]
pub struct GrouperReport {
    pub families: Vec<(String, GrouperCounts, GrouperCounts)>,
    pub total_gold: GrouperCounts,
    pub total_end_to_end: GrouperCounts,
    pub by_country: BTreeMap<String, (GrouperCounts, GrouperCounts)>,
    pub by_source: BTreeMap<String, (GrouperCounts, GrouperCounts)>,
}

type Span = (&'static str, usize, usize);

/// A fixture case with its gold entities and, per gold contact, its anchor (the person, else the
/// org) and every member span including the anchor.
#[derive(Debug)]
struct GoldCase {
    text: String,
    entities: Vec<Entity>,
    contacts: Vec<(Span, BTreeSet<Span>)>,
}

fn validate_case(case: &GrouperCase) -> anyhow::Result<()> {
    let mut ownership = vec![0usize; case.entities.len()];
    let mut spans = BTreeSet::new();
    for entity in &case.entities {
        ensure!(
            case.input.get(entity.start..entity.end) == Some(entity.text.as_str()),
            "{}: entity {}..{} does not slice to its text",
            case.name,
            entity.start,
            entity.end
        );
        ensure!(
            spans.insert((&entity.kind, entity.start, entity.end)),
            "{}: duplicate entity span {}..{}",
            case.name,
            entity.start,
            entity.end
        );
    }
    let mut anchors = BTreeSet::new();
    for contact in &case.contacts {
        let anchor = contact
            .person
            .or(contact.org)
            .with_context(|| format!("{}: a contact has no anchor", case.name))?;
        ensure!(
            anchors.insert(anchor),
            "{}: duplicate contact anchor {anchor}",
            case.name
        );
        let fields = contact
            .person
            .into_iter()
            .map(|i| (i, "person"))
            .chain(contact.org.into_iter().map(|i| (i, "org")))
            .chain(contact.addresses.iter().copied().map(|i| (i, "address")))
            .chain(contact.emails.iter().copied().map(|i| (i, "email")))
            .chain(contact.phones.iter().copied().map(|i| (i, "phone")));
        for (index, kind) in fields {
            let entity = case
                .entities
                .get(index)
                .with_context(|| format!("{}: no entity {index}", case.name))?;
            ensure!(
                entity.kind == kind,
                "{}: entity {index} is not {kind}",
                case.name
            );
            ownership[index] += 1;
        }
    }
    for &index in &case.unassigned {
        if index >= ownership.len() {
            bail!("{}: no entity {index}", case.name);
        }
        ownership[index] += 1;
    }
    for (index, count) in ownership.into_iter().enumerate() {
        ensure!(
            count == 1,
            "{}: entity {index} has {count} owners",
            case.name
        );
    }
    Ok(())
}

fn span(e: &Entity) -> Span {
    (e.kind.as_str(), e.start, e.end)
}

/// Gold entities as `detect` would return them: confidence 0.95, emails and phones from the
/// rules with a normalized form, which the email name match reads.
fn gold_case(case: &GrouperCase) -> anyhow::Result<GoldCase> {
    validate_case(case)?;
    let entities = case
        .entities
        .iter()
        .map(|e| {
            let kind = Kind::from_str_label(&e.kind)
                .with_context(|| format!("{}: unknown kind {}", case.name, e.kind))?;
            Ok(Entity {
                kind,
                start: e.start,
                end: e.end,
                confidence: 0.95,
                review_recommended: false,
                source: if matches!(kind, Kind::Email | Kind::Phone) {
                    Source::Rules
                } else {
                    Source::Model
                },
                components: Vec::new(),
                normalized: match kind {
                    Kind::Email => Some(e.text.to_ascii_lowercase()),
                    Kind::Phone => Some(e.text.replace(' ', "")),
                    _ => None,
                },
                region: None,
            })
        })
        .collect::<anyhow::Result<Vec<Entity>>>()?;
    let at = |i: usize| -> anyhow::Result<Span> {
        entities
            .get(i)
            .map(span)
            .with_context(|| format!("{}: no entity {i}", case.name))
    };
    let mut contacts = Vec::new();
    for c in &case.contacts {
        let anchor = at(c
            .person
            .or(c.org)
            .with_context(|| format!("{}: a contact has no anchor", case.name))?)?;
        let members = c
            .person
            .iter()
            .chain(&c.org)
            .chain(&c.addresses)
            .chain(&c.emails)
            .chain(&c.phones)
            .map(|&i| at(i))
            .collect::<anyhow::Result<BTreeSet<Span>>>()?;
        contacts.push((anchor, members));
    }
    Ok(GoldCase {
        text: case.input.clone(),
        entities,
        contacts,
    })
}

/// Adds one case's counts: every gold entity is correct, wrong, or missed (an anchor attached to
/// another contact is wrong, one dropped is missed), and every gold contact is exact or not.
fn score(case: &GoldCase, predicted: &Extraction, counts: &mut GrouperCounts) {
    counts.cases += 1;
    score_fields(case, predicted, counts);
    let mut predicted_by_anchor: HashMap<Span, BTreeSet<Span>> = HashMap::new();
    let mut owner_of: HashMap<Span, Span> = HashMap::new();
    let mut predicted_cards = Vec::with_capacity(predicted.contacts.len());
    for c in &predicted.contacts {
        let members: BTreeSet<Span> = c.entities().map(span).collect();
        if let Some(anchor) = c.person.as_ref().or(c.org.as_ref()).map(span) {
            for m in &members {
                owner_of.insert(*m, anchor);
            }
            predicted_by_anchor.insert(anchor, members.clone());
        }
        predicted_cards.push((members, c.confidence));
    }
    let gold_owner: HashMap<Span, Span> = case
        .contacts
        .iter()
        .flat_map(|(a, ms)| ms.iter().map(move |m| (*m, *a)))
        .collect();
    let gold_anchors: BTreeSet<Span> = case.contacts.iter().map(|(a, _)| *a).collect();
    for s in case.entities.iter().map(span) {
        counts.assignments += 1;
        match (gold_owner.get(&s), owner_of.get(&s)) {
            (None, None) => counts.correct += 1,
            (Some(g), Some(p)) if g == p => counts.correct += 1,
            (Some(_), None) => counts.missed += 1,
            _ => counts.wrong += 1,
        }
    }
    counts.gold_contacts += case.contacts.len();
    counts.exact_contacts += case
        .contacts
        .iter()
        .filter(|(anchor, members)| predicted_by_anchor.get(anchor) == Some(members))
        .count();
    counts.false_contacts += predicted_by_anchor
        .keys()
        .filter(|a| !gold_anchors.contains(*a))
        .count();

    counts.predicted_contacts += predicted_cards.len();
    let mut matched_gold = vec![false; case.contacts.len()];
    for (predicted_members, confidence) in &predicted_cards {
        if *confidence >= 0.90 {
            counts.high_confidence_predicted_cards += 1;
        }
        if let Some(index) = case
            .contacts
            .iter()
            .enumerate()
            .find(|(i, (_, gold_members))| !matched_gold[*i] && *gold_members == *predicted_members)
            .map(|(i, _)| i)
        {
            matched_gold[index] = true;
            counts.exact_card_matches += 1;
            if *confidence >= 0.90 {
                counts.high_confidence_exact_cards += 1;
            }
        } else {
            counts.unmatched_predicted_contacts += 1;
        }
    }
    counts.unmatched_gold_contacts += matched_gold.iter().filter(|&&matched| !matched).count();
    counts.false_card_documents += usize::from(
        predicted_cards.len() > matched_gold.iter().filter(|&&matched| matched).count(),
    );

    for c in &predicted.contacts {
        let gold_members = c
            .person
            .as_ref()
            .or(c.org.as_ref())
            .map(span)
            .and_then(|anchor| case.contacts.iter().find(|(a, _)| *a == anchor))
            .map(|(_, members)| members);
        for member in c.entities().map(|e| (span(e), e.kind.as_str())) {
            if gold_members.is_some_and(|members| members.contains(&member.0)) {
                continue;
            }
            if gold_owner.contains_key(&member.0) {
                counts.wrong_ownership_fields += 1;
            } else {
                counts.false_assigned_fields += 1;
                *counts
                    .false_assigned_by_kind
                    .entry(member.1.to_string())
                    .or_default() += 1;
            }
        }
    }
}

fn score_fields(case: &GoldCase, predicted: &Extraction, counts: &mut GrouperCounts) {
    let gold_entities: Vec<_> = case.entities.iter().map(span).collect();
    let predicted_entities: Vec<_> = predicted
        .contacts
        .iter()
        .flat_map(|contact| contact.entities())
        .chain(&predicted.unassigned)
        .map(span)
        .collect();
    let gold_fields: Vec<_> = case
        .contacts
        .iter()
        .flat_map(|(anchor, members)| members.iter().map(move |member| (Some(*anchor), *member)))
        .collect();
    let predicted_fields: Vec<_> = predicted
        .contacts
        .iter()
        .flat_map(|contact| {
            let anchor = contact.person.as_ref().or(contact.org.as_ref()).map(span);
            contact.entities().map(move |entity| (anchor, span(entity)))
        })
        .collect();
    let detection_matches = crate::exact_metrics::matches(&gold_entities, &predicted_entities);
    let field_matches = crate::exact_metrics::matches(&gold_fields, &predicted_fields);
    for kind in Kind::ALL {
        let kind = kind.as_str();
        counts
            .detection_by_kind
            .entry(kind.to_string())
            .or_default()
            .add(&crate::exact_metrics::Counts {
                tp: predicted_entities
                    .iter()
                    .zip(&detection_matches)
                    .filter(|(s, matched)| s.0 == kind && **matched)
                    .count(),
                predicted: predicted_entities.iter().filter(|s| s.0 == kind).count(),
                gold: gold_entities.iter().filter(|s| s.0 == kind).count(),
            });
        counts
            .contact_fields_by_kind
            .entry(kind.to_string())
            .or_default()
            .add(&crate::exact_metrics::Counts {
                tp: predicted_fields
                    .iter()
                    .zip(&field_matches)
                    .filter(|((_, s), matched)| s.0 == kind && **matched)
                    .count(),
                predicted: predicted_fields.iter().filter(|(_, s)| s.0 == kind).count(),
                gold: gold_fields.iter().filter(|(_, s)| s.0 == kind).count(),
            });
    }
}

/// Scores every fixture file in `dir`, on gold entities and end to end through `tessera`.
pub fn eval_grouper(dir: &Path, tessera: &Tessera) -> anyhow::Result<GrouperReport> {
    let mut report = GrouperReport::default();
    for (family, fixture) in fixtures::load_dir::<GrouperFixture>(dir)? {
        let mut gold_counts = GrouperCounts::default();
        let mut e2e_counts = GrouperCounts::default();
        for case in &fixture.cases {
            let gold = gold_case(case)?;
            let predicted = tessera::internal::group(&gold.text, gold.entities.clone());
            let mut gold_one = GrouperCounts::default();
            score(&gold, &predicted, &mut gold_one);
            let predicted = tessera
                .extract_contacts(&gold.text, &Query::default())
                .map_err(|e| anyhow::anyhow!("{family}/{}: {e}", case.name))?;
            let mut e2e_one = GrouperCounts::default();
            score(&gold, &predicted, &mut e2e_one);
            gold_counts.add(&gold_one);
            e2e_counts.add(&e2e_one);
            if let Some(country) = &case.country {
                let (gold_total, e2e_total) = report.by_country.entry(country.clone()).or_default();
                gold_total.add(&gold_one);
                e2e_total.add(&e2e_one);
            }
            if let Some(source) = &case.source {
                let (gold_total, e2e_total) = report.by_source.entry(source.clone()).or_default();
                gold_total.add(&gold_one);
                e2e_total.add(&e2e_one);
            }
        }
        report.total_gold.add(&gold_counts);
        report.total_end_to_end.add(&e2e_counts);
        report.families.push((family, gold_counts, e2e_counts));
    }
    Ok(report)
}

fn pct(n: usize, d: usize) -> String {
    if d == 0 {
        "–".into()
    } else {
        format!("{:.1}%", 100.0 * n as f64 / d as f64)
    }
}

fn card_f1(c: &GrouperCounts) -> String {
    if c.predicted_contacts == 0 || c.gold_contacts == 0 {
        return "0.0%".into();
    }
    let p = c.exact_card_matches as f64 / c.predicted_contacts as f64;
    let r = c.exact_card_matches as f64 / c.gold_contacts as f64;
    if p + r == 0.0 {
        "0.0%".into()
    } else {
        format!("{:.1}%", 200.0 * p * r / (p + r))
    }
}

fn count_row(label: &str, mode: &str, c: &GrouperCounts) -> String {
    format!(
        "| {label} | {mode} | {} | {} | {} | {} | {} / {} / {} | {} | {} | {} | {} | {} / {} | {} | {} | {} |\n",
        c.cases,
        pct(c.correct, c.assignments),
        c.wrong,
        c.missed,
        c.exact_card_matches,
        c.predicted_contacts,
        c.gold_contacts,
        pct(c.exact_card_matches, c.predicted_contacts),
        pct(c.exact_card_matches, c.gold_contacts),
        card_f1(c),
        pct(c.false_card_documents, c.cases),
        c.high_confidence_exact_cards,
        c.high_confidence_predicted_cards,
        c.false_assigned_fields,
        c.wrong_ownership_fields,
        pct(c.exact_contacts, c.gold_contacts),
    )
}

fn table(rows: impl Iterator<Item = (String, GrouperCounts, GrouperCounts)>) -> String {
    let mut out = String::from(
        "| Slice | Mode | Cases | Assignment accuracy | Wrong | Missed | Exact TP / predicted / gold | Card P | Card R | Card F1 | False-card documents | High-confidence exact / predicted | Extra assigned fields | Wrong ownership fields | Legacy exact recall |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for (label, gold, e2e) in rows {
        out.push_str(&count_row(&label, "gold entities", &gold));
        out.push_str(&count_row(&label, "end to end", &e2e));
    }
    out
}

fn family_table(report: &GrouperReport) -> String {
    let rows = report.families.iter().cloned().chain([(
        "total".to_string(),
        report.total_gold.clone(),
        report.total_end_to_end.clone(),
    )]);
    table(rows)
}

fn breakdown_table(rows: &BTreeMap<String, (GrouperCounts, GrouperCounts)>) -> String {
    table(
        rows.iter()
            .map(|(name, (g, e))| (name.clone(), g.clone(), e.clone())),
    )
}

fn false_fields_table(c: &GrouperCounts) -> String {
    let mut out = String::from("| Kind | Extra assigned fields |\n| --- | ---: |\n");
    for (kind, n) in &c.false_assigned_by_kind {
        out.push_str(&format!("| {kind} | {n} |\n"));
    }
    out
}

fn field_table(counts: &GrouperCounts) -> String {
    let mut out = String::from(
        "| Kind | Mode | TP | FP | FN | Precision | Recall | F1 |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for kind in Kind::ALL {
        for (mode, fields) in [
            ("entities", &counts.detection_by_kind),
            ("contact fields", &counts.contact_fields_by_kind),
        ] {
            let c = fields.get(kind.as_str()).cloned().unwrap_or_default();
            let (p, r, f) = c.prf();
            out.push_str(&format!(
                "| {} | {mode} | {} | {} | {} | {:.2}% | {:.2}% | {:.2}% |\n",
                kind.as_str(),
                c.tp,
                c.predicted - c.tp,
                c.gold - c.tp,
                p * 100.0,
                r * 100.0,
                f * 100.0
            ));
        }
    }
    out
}

/// The grouping report: the fixture table, the known-hard table when given, and what the
/// end-to-end numbers include.
pub fn render_markdown(
    report: &GrouperReport,
    hard: Option<&GrouperReport>,
    bundle: &str,
    bundle_sha256: &str,
    manifest_sha256: Option<&str>,
) -> String {
    let mut out = format!(
        "# Contact grouping\n\nMetric version 4. Bundle `{bundle}`; SHA-256 `{bundle_sha256}`. Corpus manifest SHA-256 `{}`. End-to-end query: `Query::default()` (no country hint). Exact cards match `(kind, start, end)` member sets one-to-one. Extra assigned fields include members of false-anchor cards. High confidence means contact confidence ≥0.90.\n\n{}",
        manifest_sha256.unwrap_or("not provided"),
        family_table(report)
    );
    out.push_str(&format!("\n## Exact fields, end to end\n\n{}\nEntity scores include unassigned fields. Contact-field scores require the correct exact anchor and field span; a wrong owner counts as both a false positive and a miss. Empty kinds score zero and cannot establish accuracy.\n", field_table(&report.total_end_to_end)));
    if !report.by_country.is_empty() {
        out.push_str(&format!(
            "\n## By country\n\n{}",
            breakdown_table(&report.by_country)
        ));
    }
    if !report.by_source.is_empty() {
        out.push_str(&format!(
            "\n## By source\n\n{}",
            breakdown_table(&report.by_source)
        ));
    }
    out.push_str(&format!(
        "\n## Extra assigned fields by kind, end to end\n\n{}\nHigh-confidence exact-card precision: {} ({} / {} predicted cards). Unmatched predicted cards: {}; unmatched gold cards: {}; contacts with an unexpected anchor: {}.\n",
        false_fields_table(&report.total_end_to_end),
        pct(report.total_end_to_end.high_confidence_exact_cards, report.total_end_to_end.high_confidence_predicted_cards),
        report.total_end_to_end.high_confidence_exact_cards,
        report.total_end_to_end.high_confidence_predicted_cards,
        report.total_end_to_end.unmatched_predicted_contacts,
        report.total_end_to_end.unmatched_gold_contacts,
        report.total_end_to_end.false_contacts,
    ));
    if let Some(hard) = hard {
        out.push_str(&format!(
            "\n## Known-hard cases (not gating)\n\n{}",
            family_table(hard)
        ));
    }
    out.push_str(
        "\nEnd-to-end numbers include detector boundary errors; a span that differs by one character from gold counts as a grouping miss. Legacy exact recall is the old anchor-based count over gold contacts and is retained only for comparison. The fixture corpus is small; its country breakdowns are descriptive, not release estimates.\n",
    );
    out
}

/// `trainer eval --grouper <dir>`: scores the grouper fixtures and, with `--grouper-hard`, the
/// known-hard cases, end to end with `--bundle`; prints the report and writes it to `--report`.
pub fn run(args: &EvalArgs, dir: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(&args.bundle)
        .with_context(|| format!("reading {}", args.bundle.display()))?;
    let tessera = Tessera::load(
        &bytes,
        Config {
            kinds: Kind::all(),
            expected_checksum: None,
        },
    )
    .map_err(|e| anyhow::anyhow!("loading {}: {e}", args.bundle.display()))?;
    let report = eval_grouper(dir, &tessera)?;
    let hard = args
        .grouper_hard
        .as_deref()
        .map(|d| eval_grouper(d, &tessera))
        .transpose()?;
    let bundle_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let manifest = dir.join("manifest.jsonl");
    let manifest_sha256 = manifest
        .exists()
        .then(|| std::fs::read(&manifest).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
        .transpose()?;
    let markdown = render_markdown(
        &report,
        hard.as_ref(),
        &args.bundle.display().to_string(),
        &bundle_sha256,
        manifest_sha256.as_deref(),
    );
    print!("{markdown}");
    if let Some(path) = &args.report {
        std::fs::write(path, &markdown).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{ExpectedContact, ExpectedSpan};

    fn entity(kind: Kind, start: usize, end: usize) -> Entity {
        Entity {
            kind,
            start,
            end,
            confidence: 0.95,
            review_recommended: false,
            source: Source::Model,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    fn contact(person: Entity, phones: Vec<Entity>) -> tessera::Contact {
        tessera::Contact {
            start: person.start,
            end: phones.last().map_or(person.end, |phone| phone.end),
            confidence: 0.95,
            review_recommended: false,
            person: Some(person),
            org: None,
            addresses: Vec::new(),
            emails: Vec::new(),
            phones,
        }
    }

    fn gold(people: &[Entity]) -> GoldCase {
        GoldCase {
            text: String::new(),
            entities: people.to_vec(),
            contacts: people
                .iter()
                .map(|person| (span(person), BTreeSet::from([span(person)])))
                .collect(),
        }
    }

    #[test]
    fn extra_card_lowers_precision_but_not_recall() {
        let person = entity(Kind::Person, 0, 4);
        let extra = entity(Kind::Person, 10, 14);
        let predicted = Extraction {
            contacts: vec![contact(person.clone(), vec![]), contact(extra, vec![])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold(&[person]), &predicted, &mut counts);
        assert_eq!(
            (
                counts.exact_card_matches,
                counts.predicted_contacts,
                counts.gold_contacts
            ),
            (1, 2, 1)
        );
        assert_eq!(
            (
                counts.unmatched_predicted_contacts,
                counts.false_card_documents
            ),
            (1, 1)
        );
    }

    #[test]
    fn extra_phone_on_valid_card_is_a_false_assigned_field() {
        let person = entity(Kind::Person, 0, 4);
        let phone = entity(Kind::Phone, 5, 15);
        let predicted = Extraction {
            contacts: vec![contact(person.clone(), vec![phone])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold(&[person]), &predicted, &mut counts);
        assert_eq!(counts.false_assigned_fields, 1);
        assert_eq!(counts.false_assigned_by_kind["phone"], 1);
        assert_eq!(counts.exact_card_matches, 0);
    }

    #[test]
    fn field_taken_from_another_card_is_wrong_ownership() {
        let first = entity(Kind::Person, 0, 4);
        let second = entity(Kind::Person, 10, 14);
        let phone = entity(Kind::Phone, 15, 25);
        let gold = GoldCase {
            text: String::new(),
            entities: vec![first.clone(), second.clone(), phone.clone()],
            contacts: vec![
                (span(&first), BTreeSet::from([span(&first)])),
                (span(&second), BTreeSet::from([span(&second), span(&phone)])),
            ],
        };
        let predicted = Extraction {
            contacts: vec![contact(first, vec![phone]), contact(second, vec![])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold, &predicted, &mut counts);
        assert_eq!(counts.wrong_ownership_fields, 1);
        assert_eq!(counts.false_assigned_fields, 0);
        assert_eq!(counts.detection_by_kind["phone"].tp, 1);
        assert_eq!(
            counts.contact_fields_by_kind["phone"],
            crate::exact_metrics::Counts {
                tp: 0,
                predicted: 1,
                gold: 1
            }
        );
    }

    #[test]
    fn missing_card_lowers_recall() {
        let first = entity(Kind::Person, 0, 4);
        let second = entity(Kind::Person, 10, 14);
        let predicted = Extraction {
            contacts: vec![contact(first.clone(), vec![])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold(&[first, second]), &predicted, &mut counts);
        assert_eq!(
            (
                counts.exact_card_matches,
                counts.gold_contacts,
                counts.unmatched_gold_contacts
            ),
            (1, 2, 1)
        );
    }

    #[test]
    fn duplicate_predicted_card_only_matches_once() {
        let person = entity(Kind::Person, 0, 4);
        let predicted = Extraction {
            contacts: vec![
                contact(person.clone(), vec![]),
                contact(person.clone(), vec![]),
            ],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold(&[person]), &predicted, &mut counts);
        assert_eq!(
            (
                counts.exact_card_matches,
                counts.unmatched_predicted_contacts
            ),
            (1, 1)
        );
        assert_eq!(
            counts.detection_by_kind["person"].prf(),
            (0.5, 1.0, 2.0 / 3.0)
        );
        assert_eq!(counts.contact_fields_by_kind["person"].tp, 1);
        assert_eq!(counts.contact_fields_by_kind["person"].predicted, 2);
    }

    #[test]
    fn missing_unassigned_entity_is_a_detection_miss() {
        let phone = entity(Kind::Phone, 0, 10);
        let case = GoldCase {
            text: String::new(),
            entities: vec![phone],
            contacts: vec![],
        };
        let predicted = Extraction {
            contacts: vec![],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&case, &predicted, &mut counts);
        assert_eq!(counts.correct, 1);
        assert_eq!(counts.detection_by_kind["phone"].gold, 1);
        assert_eq!(counts.detection_by_kind["phone"].tp, 0);
        assert_eq!(counts.contact_fields_by_kind["phone"].gold, 0);
    }

    #[test]
    fn one_character_boundary_error_costs_detection_and_contact_recall() {
        let person = entity(Kind::Person, 0, 4);
        let predicted = Extraction {
            contacts: vec![contact(entity(Kind::Person, 0, 3), vec![])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold(&[person]), &predicted, &mut counts);
        for fields in [&counts.detection_by_kind, &counts.contact_fields_by_kind] {
            assert_eq!(
                fields["person"],
                crate::exact_metrics::Counts {
                    tp: 0,
                    predicted: 1,
                    gold: 1
                }
            );
        }
    }

    #[test]
    fn zero_gold_cards_still_count_invented_cards() {
        let person = entity(Kind::Person, 0, 4);
        let gold = GoldCase {
            text: String::new(),
            entities: vec![],
            contacts: vec![],
        };
        let predicted = Extraction {
            contacts: vec![contact(person, vec![])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold, &predicted, &mut counts);
        assert_eq!(
            (
                counts.predicted_contacts,
                counts.unmatched_predicted_contacts,
                counts.false_card_documents
            ),
            (1, 1, 1)
        );
        assert_eq!(card_f1(&counts), "0.0%");
    }

    #[test]
    fn false_card_counts_its_anchor_and_extra_phone() {
        let person = entity(Kind::Person, 0, 4);
        let phone = entity(Kind::Phone, 5, 15);
        let gold = GoldCase {
            text: String::new(),
            entities: vec![],
            contacts: vec![],
        };
        let predicted = Extraction {
            contacts: vec![contact(person, vec![phone])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold, &predicted, &mut counts);
        assert_eq!(
            (
                counts.unmatched_predicted_contacts,
                counts.false_card_documents
            ),
            (1, 1)
        );
        assert_eq!(counts.false_assigned_fields, 2);
        assert_eq!(counts.false_assigned_by_kind["person"], 1);
        assert_eq!(counts.false_assigned_by_kind["phone"], 1);
    }

    #[test]
    fn false_card_stealing_gold_phone_counts_wrong_ownership() {
        let true_person = entity(Kind::Person, 0, 4);
        let invented_person = entity(Kind::Person, 10, 14);
        let phone = entity(Kind::Phone, 15, 25);
        let gold = GoldCase {
            text: String::new(),
            entities: vec![true_person.clone(), phone.clone()],
            contacts: vec![(
                span(&true_person),
                BTreeSet::from([span(&true_person), span(&phone)]),
            )],
        };
        let predicted = Extraction {
            contacts: vec![contact(invented_person, vec![phone])],
            unassigned: vec![],
        };
        let mut counts = GrouperCounts::default();
        score(&gold, &predicted, &mut counts);
        assert_eq!(counts.wrong_ownership_fields, 1);
        assert_eq!(counts.false_assigned_by_kind["person"], 1);
        assert_eq!(
            (
                counts.exact_card_matches,
                counts.predicted_contacts,
                counts.gold_contacts
            ),
            (0, 1, 1)
        );
    }

    #[test]
    fn fixture_rejects_duplicate_ownership() {
        let case = GrouperCase {
            name: "duplicate".into(),
            country: Some("GB".into()),
            source: Some("fixture".into()),
            input: "Jane".into(),
            entities: vec![ExpectedSpan {
                kind: "person".into(),
                text: "Jane".into(),
                start: 0,
                end: 4,
            }],
            contacts: vec![ExpectedContact {
                person: Some(0),
                org: None,
                addresses: vec![],
                emails: vec![],
                phones: vec![],
            }],
            unassigned: vec![0],
        };
        assert!(
            gold_case(&case)
                .unwrap_err()
                .to_string()
                .contains("2 owners")
        );
    }

    #[test]
    fn score_counts_wrong_and_missed_separately() {
        let person = entity(Kind::Person, 0, 4);
        let phone = entity(Kind::Phone, 5, 10);
        let email = entity(Kind::Email, 11, 20);
        let gold = GoldCase {
            text: String::new(),
            entities: vec![person.clone(), phone.clone(), email.clone()],
            contacts: vec![(span(&person), BTreeSet::from([span(&person), span(&phone)]))],
        };
        let predicted = Extraction {
            contacts: vec![tessera::Contact {
                start: 0,
                end: 20,
                confidence: 0.95,
                review_recommended: false,
                person: Some(person),
                org: None,
                addresses: Vec::new(),
                emails: vec![email],
                phones: Vec::new(),
            }],
            unassigned: vec![phone],
        };
        let mut counts = GrouperCounts::default();
        score(&gold, &predicted, &mut counts);
        assert_eq!(
            (
                counts.correct,
                counts.wrong,
                counts.missed,
                counts.exact_contacts
            ),
            (1, 1, 1, 0)
        );
    }
}
