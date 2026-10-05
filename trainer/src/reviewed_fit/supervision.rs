//! Reviewed source-ambiguity loss exclusions: a hash-pinned sidecar naming label-O TRAIN token
//! ranges whose source annotation two reviewers could not resolve.
//!
//! Selection comes only from the sidecar, resolved against canonical native TRAIN encodings;
//! model scores and DEV rows never choose a token. Excluded tokens keep their O label and every
//! forward input; only their loss activity is removed.

use crate::dataset::Encoded;
use crate::detector::DetectorDoc;
use crate::reviewed_data::{Cohort, DeclaredExclusion, Receipt, digest, valid_sha};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use tessera::internal::flag;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
enum Version {
    #[serde(rename = "reviewed-source-ambiguity-exclusions-v1")]
    V1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// One unresolved source mention, in UTF-8 bytes of the exact native input.
struct Exclusion {
    name: String,
    input_sha256: String,
    start: u32,
    end: u32,
    quote: String,
    reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Versioned sidecar binding two independent reviewer outputs and their adjudication evidence.
pub(super) struct Sidecar {
    version: Version,
    reviewers: [Receipt; 2],
    adjudication: Receipt,
    exclusions: Vec<Exclusion>,
}

/// A canonical TRAIN row as resolution sees it: name, encoding and every expected span.
pub(super) struct Case<'a> {
    pub(super) name: &'a str,
    pub(super) doc: &'a DetectorDoc,
    /// Byte ranges of all five expected kinds, including rule-owned email and phone.
    pub(super) positives: Vec<(u32, u32)>,
}

/// View native TRAIN rows with their complete reviewed expected spans.
pub(super) fn cases(train: &Cohort) -> anyhow::Result<Vec<Case<'_>>> {
    ensure!(
        train.names.len() == train.docs.len() && train.expected.len() == train.docs.len(),
        "native TRAIN rows are not aligned"
    );
    Ok(train
        .names
        .iter()
        .zip(&train.docs)
        .zip(&train.expected)
        .map(|((name, doc), expected)| Case {
            name,
            doc,
            positives: expected
                .iter()
                .map(|s| (s.start, s.end))
                .chain(doc.gold.iter().map(|g| (g.start, g.end)))
                .collect(),
        })
        .collect())
}

/// Read a pinned sidecar and verify its evidence receipts and entry structure.
pub(super) fn load(receipt: &Receipt) -> anyhow::Result<Sidecar> {
    let sidecar: Sidecar = serde_json::from_slice(&receipt.bytes()?)
        .context("reviewed ambiguity sidecar is not the strict versioned schema")?;
    sidecar.validate()?;
    Ok(sidecar)
}

impl Sidecar {
    fn validate(&self) -> anyhow::Result<()> {
        let mut paths = BTreeSet::new();
        let mut hashes = BTreeSet::new();
        for receipt in self.reviewers.iter().chain([&self.adjudication]) {
            receipt.bytes()?;
            ensure!(
                paths.insert(receipt.path.canonicalize()?) && hashes.insert(&receipt.sha256),
                "reviewer outputs and adjudication evidence must be three distinct files"
            );
        }
        ensure!(
            !self.exclusions.is_empty(),
            "reviewed ambiguity sidecar declares no exclusions"
        );
        let mut cases = BTreeMap::<&str, (&str, Vec<(u32, u32)>)>::new();
        for entry in &self.exclusions {
            ensure!(
                !entry.name.trim().is_empty()
                    && valid_sha(&entry.input_sha256)
                    && entry.start < entry.end
                    && usize::try_from(entry.end - entry.start)? == entry.quote.len()
                    && !entry.reason.trim().is_empty(),
                "reviewed ambiguity entry {} needs a name, input hash, nonempty byte range, exact quote and reason",
                entry.name
            );
            let (hash, spans) = cases
                .entry(&entry.name)
                .or_insert((&entry.input_sha256, Vec::new()));
            ensure!(
                *hash == entry.input_sha256,
                "reviewed ambiguity entries for {} bind different inputs",
                entry.name
            );
            spans.push((entry.start, entry.end));
        }
        for (name, (_, spans)) in &mut cases {
            spans.sort_unstable();
            ensure!(
                spans.windows(2).all(|p| p[0].1 <= p[1].0),
                "reviewed ambiguity entries for {name} repeat or overlap"
            );
        }
        Ok(())
    }

    /// Map every entry onto whole label-O tokens of exactly one canonical TRAIN row.
    pub(super) fn resolve(
        &self,
        sidecar: &Receipt,
        cases: &[Case<'_>],
    ) -> anyhow::Result<Resolved> {
        let mut by_name = BTreeMap::new();
        for (row, case) in cases.iter().enumerate() {
            ensure!(
                by_name.insert(case.name, row).is_none(),
                "TRAIN case name {} repeats",
                case.name
            );
        }
        let mut rows = BTreeMap::<usize, Vec<bool>>::new();
        let mut entries = Vec::new();
        let mut keys = BTreeSet::new();
        for entry in &self.exclusions {
            let row = *by_name.get(entry.name.as_str()).with_context(|| {
                format!("reviewed ambiguity names unknown TRAIN case {}", entry.name)
            })?;
            let tokens = covered_tokens(entry, &cases[row])?;
            let excluded = rows
                .entry(row)
                .or_insert_with(|| vec![false; cases[row].doc.enc.token_spans.len()]);
            for token in tokens.clone() {
                excluded[token] = true;
            }
            keys.insert(Key::of(entry));
            entries.push(
                json!({"name":entry.name,"native_train_row":row,"input_sha256":entry.input_sha256,
                "start":entry.start,"end":entry.end,"quote":entry.quote,"reason":entry.reason,
                "tokens":[tokens.start,tokens.end]}),
            );
        }
        let mut excluded_tokens = 0usize;
        for (&row, excluded) in &rows {
            let enc = &cases[row].doc.enc;
            ensure!(
                excluded
                    .iter()
                    .zip(&enc.flags)
                    .any(|(&out, &flags)| !out && flags & flag::IN_RULE_SPAN == 0),
                "reviewed ambiguity would remove all loss from TRAIN case {}",
                cases[row].name
            );
            excluded_tokens += excluded.iter().filter(|&&out| out).count();
        }
        let mut dependencies = vec![sidecar.clone()];
        dependencies.extend(self.reviewers.iter().cloned());
        dependencies.push(self.adjudication.clone());
        let identity = json!({"version":self.version,"sidecar":sidecar,"reviewers":self.reviewers,
            "adjudication":self.adjudication,"entries":entries,"excluded_unique_tokens":excluded_tokens,
            "selection":"reviewed sidecar only, resolved against canonical native TRAIN; never model scores or DEV",
            "excluded_labels":"canonical O placeholder kept; forward inputs unchanged"});
        Ok(Resolved {
            rows,
            keys,
            dependencies,
            identity,
        })
    }
}

/// Whole tokens exactly spanning the entry, rejecting rule, positive or supervised overlap.
fn covered_tokens(entry: &Exclusion, case: &Case<'_>) -> anyhow::Result<std::ops::Range<usize>> {
    let (doc, name) = (case.doc, case.name);
    ensure!(
        digest(doc.text.as_bytes()) == entry.input_sha256,
        "reviewed ambiguity input hash differs for {name}"
    );
    let start = usize::try_from(entry.start)?;
    let end = usize::try_from(entry.end)?;
    ensure!(
        doc.text.get(start..end) == Some(entry.quote.as_str()),
        "reviewed ambiguity quote differs from {name} bytes [{start},{end})"
    );
    let spans = &doc.enc.token_spans;
    let first = spans.iter().position(|t| t.0 == entry.start);
    let last = spans.iter().position(|t| t.1 == entry.end);
    let (Some(first), Some(last)) = (first, last) else {
        anyhow::bail!("reviewed ambiguity in {name} [{start},{end}) splits a token");
    };
    ensure!(
        first <= last && doc.enc.flags.len() == spans.len() && doc.enc.labels.len() == spans.len(),
        "reviewed ambiguity in {name} does not map to whole canonical tokens"
    );
    ensure!(
        !case
            .positives
            .iter()
            .any(|&(s, e)| s < entry.end && entry.start < e),
        "reviewed ambiguity in {name} [{start},{end}) overlaps a known expected positive span"
    );
    for token in first..=last {
        ensure!(
            doc.enc.flags[token] & (flag::IN_RULE_SPAN | flag::MASKED) == 0,
            "reviewed ambiguity in {name} covers a rule-owned token"
        );
        ensure!(
            doc.enc.labels[token] == 0,
            "reviewed ambiguity in {name} covers a positively supervised token"
        );
    }
    Ok(first..last + 1)
}

/// The span identity both the sidecar and the TRAIN adjudication must declare.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    name: String,
    input_sha256: String,
    start: u32,
    end: u32,
    quote: String,
}

impl Key {
    fn of(entry: &Exclusion) -> Self {
        Self {
            name: entry.name.clone(),
            input_sha256: entry.input_sha256.clone(),
            start: entry.start,
            end: entry.end,
            quote: entry.quote.clone(),
        }
    }
}

/// Resolved per-row token exclusions keyed by native TRAIN row, with their binding receipts.
pub(super) struct Resolved {
    rows: BTreeMap<usize, Vec<bool>>,
    keys: BTreeSet<Key>,
    pub(super) dependencies: Vec<Receipt>,
    pub(super) identity: Value,
}

impl Resolved {
    /// Per-token exclusions for drawn rows; `owners` maps each drawn row to its native TRAIN row.
    pub(super) fn batch(
        &self,
        owners: &[Option<usize>],
        items: &[Encoded],
    ) -> anyhow::Result<Vec<Vec<bool>>> {
        ensure!(
            owners.len() == items.len(),
            "reviewed ambiguity owners differ from batch rows"
        );
        owners
            .iter()
            .zip(items)
            .map(|(owner, item)| self.row(*owner, item.token_spans.len()))
            .collect()
    }

    /// Exclusions for one row, all false for rows without reviewed ambiguity.
    pub(super) fn row(&self, owner: Option<usize>, tokens: usize) -> anyhow::Result<Vec<bool>> {
        match owner.and_then(|row| self.rows.get(&row)) {
            Some(excluded) => {
                ensure!(
                    excluded.len() == tokens,
                    "reviewed ambiguity row geometry differs from canonical encoding"
                );
                Ok(excluded.clone())
            }
            None => Ok(vec![false; tokens]),
        }
    }

    /// The trust boundary: the sidecar must name exactly the TRAIN adjudication's declared spans.
    pub(super) fn require_declared(&self, declared: &[DeclaredExclusion]) -> anyhow::Result<()> {
        let declared: BTreeSet<_> = declared
            .iter()
            .map(|d| Key {
                name: d.name.clone(),
                input_sha256: d.input_sha256.clone(),
                start: d.start,
                end: d.end,
                quote: d.quote.clone(),
            })
            .collect();
        let missing = declared.difference(&self.keys).count();
        let undeclared = self.keys.difference(&declared).count();
        ensure!(
            missing == 0 && undeclared == 0,
            "reviewed ambiguity sidecar differs from TRAIN adjudication: {missing} declared spans missing, {undeclared} undeclared"
        );
        Ok(())
    }

    /// Whether any exclusion belongs to the named native TRAIN case.
    pub(super) fn excludes_case(&self, name: &str) -> bool {
        self.keys.iter().any(|k| k.name == name)
    }

    /// Recheck every bound sidecar and evidence file.
    pub(super) fn verify(&self) -> anyhow::Result<()> {
        for receipt in &self.dependencies {
            receipt.bytes()?;
        }
        Ok(())
    }
}

#[cfg(test)]
impl Resolved {
    /// Exclusions keyed by native row, without sidecar evidence, for batch-geometry tests.
    pub(super) fn from_rows(rows: BTreeMap<usize, Vec<bool>>) -> Self {
        Self {
            rows,
            keys: BTreeSet::new(),
            dependencies: Vec::new(),
            identity: Value::Null,
        }
    }
}

#[cfg(test)]
#[path = "supervision_tests.rs"]
mod tests;
