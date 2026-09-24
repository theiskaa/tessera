//! Scores the contact grouper on the grouper fixtures: on the fixtures' gold entities, which
//! isolates grouping from detection, and end to end through `Tessera::extract_contacts` with a
//! bundle. Wrong and missed assignments are counted apart, because a wrong assignment is the
//! failure the grouper exists to prevent and an unassigned entity is the safe one.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use tessera::{Config, Entity, Extraction, Kind, Query, Source, Tessera};

use crate::EvalArgs;
use crate::fixtures::{self, GrouperCase, GrouperFixture};

/// Counts of one fixture family, or of all of them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GrouperCounts {
    pub cases: usize,
    /// Gold entities that are not a gold contact's anchor, each owned by a contact or by none.
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
    }
}

/// Per fixture family: gold-span counts and end-to-end counts.
#[derive(Debug, Default, Serialize)]
pub struct GrouperReport {
    pub families: Vec<(String, GrouperCounts, GrouperCounts)>,
    pub total_gold: GrouperCounts,
    pub total_end_to_end: GrouperCounts,
}

type Span = (&'static str, usize, usize);

/// A fixture case with its gold entities and, per gold contact, its anchor (the person, else the
/// org) and every member span including the anchor.
struct GoldCase {
    text: String,
    entities: Vec<Entity>,
    contacts: Vec<(Span, BTreeSet<Span>)>,
}

fn span(e: &Entity) -> Span {
    (e.kind.as_str(), e.start, e.end)
}

/// Gold entities as `detect` would return them: confidence 0.95, emails and phones from the
/// rules with a normalized form, which the email name match reads.
fn gold_case(case: &GrouperCase) -> anyhow::Result<GoldCase> {
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

/// Adds one case's counts: every non-anchor gold entity is correct, wrong, or missed, and every
/// gold contact is exact or not.
fn score(case: &GoldCase, predicted: &Extraction, counts: &mut GrouperCounts) {
    counts.cases += 1;
    let mut predicted_by_anchor: HashMap<Span, BTreeSet<Span>> = HashMap::new();
    let mut owner_of: HashMap<Span, Span> = HashMap::new();
    for c in &predicted.contacts {
        let Some(anchor) = c.person.as_ref().or(c.org.as_ref()).map(span) else {
            continue;
        };
        let members: BTreeSet<Span> = c
            .person
            .iter()
            .chain(&c.org)
            .chain(&c.addresses)
            .chain(&c.emails)
            .chain(&c.phones)
            .map(span)
            .collect();
        for m in &members {
            owner_of.insert(*m, anchor);
        }
        predicted_by_anchor.insert(anchor, members);
    }
    let gold_owner: HashMap<Span, Span> = case
        .contacts
        .iter()
        .flat_map(|(a, ms)| ms.iter().map(move |m| (*m, *a)))
        .collect();
    let gold_anchors: BTreeSet<Span> = case.contacts.iter().map(|(a, _)| *a).collect();
    for s in case.entities.iter().map(span) {
        if gold_anchors.contains(&s) {
            continue;
        }
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
}

/// Scores every fixture file in `dir`, on gold entities and, with `tessera`, end to end.
pub fn eval_grouper(dir: &Path, tessera: Option<&Tessera>) -> anyhow::Result<GrouperReport> {
    let mut report = GrouperReport::default();
    for (family, fixture) in fixtures::load_dir::<GrouperFixture>(dir)? {
        let mut gold_counts = GrouperCounts::default();
        let mut e2e_counts = GrouperCounts::default();
        for case in &fixture.cases {
            let gold = gold_case(case)?;
            let tokens = tessera::internal::tokenize(&gold.text);
            let predicted = tessera::internal::group(&gold.text, &tokens, gold.entities.clone());
            score(&gold, &predicted, &mut gold_counts);
            if let Some(t) = tessera {
                let predicted = t
                    .extract_contacts(&gold.text, &Query::default())
                    .map_err(|e| anyhow::anyhow!("{family}/{}: {e}", case.name))?;
                score(&gold, &predicted, &mut e2e_counts);
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

fn table(report: &GrouperReport) -> String {
    let mut out = String::from(
        "| family | cases | assignment acc | wrong | missed | exact contacts (gold) | exact contacts (e2e) | false contacts (e2e) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    let totals = (
        "total".to_string(),
        report.total_gold,
        report.total_end_to_end,
    );
    for (family, g, e) in report.families.iter().chain([&totals]) {
        out.push_str(&format!(
            "| {family} | {} | {} | {} | {} | {} | {} | {} |\n",
            g.cases,
            pct(g.correct, g.assignments),
            g.wrong,
            g.missed,
            pct(g.exact_contacts, g.gold_contacts),
            pct(e.exact_contacts, e.gold_contacts),
            e.false_contacts,
        ));
    }
    out
}

/// The grouping report: the fixture table, the known-hard table when given, and what the
/// end-to-end numbers include.
pub fn render_markdown(
    report: &GrouperReport,
    hard: Option<&GrouperReport>,
    bundle: &str,
) -> String {
    let mut out = format!(
        "# Contact grouping\n\nBundle `{bundle}`. Assignment accuracy, wrong, and missed are on the fixtures' gold entities; exact contacts are on gold entities and end to end through `extract_contacts`.\n\n{}",
        table(report)
    );
    if let Some(hard) = hard {
        out.push_str(&format!(
            "\n## Known-hard cases (not gating)\n\n{}",
            table(hard)
        ));
    }
    out.push_str(
        "\nEnd-to-end numbers include detector boundary errors; a span that differs by one character from gold counts as a grouping miss. The fixture corpus is small and per-country breakdowns are not meaningful at this size.\n",
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
    let report = eval_grouper(dir, Some(&tessera))?;
    let hard = args
        .grouper_hard
        .as_deref()
        .map(|d| eval_grouper(d, Some(&tessera)))
        .transpose()?;
    let markdown = render_markdown(&report, hard.as_ref(), &args.bundle.display().to_string());
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
            (0, 1, 1, 0)
        );
    }
}
