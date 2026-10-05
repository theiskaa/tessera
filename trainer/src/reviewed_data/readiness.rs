//! Hash-bound adjudication coverage and independently reviewed split evidence.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::data::Row;
use super::{Inputs, Receipt};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Review {
    split: String,
    dataset_sha256: String,
    policy_sha256: String,
    adjudication_complete: bool,
    unresolved_uncertainties: usize,
    reviewers: Vec<String>,
    review_a: Receipt,
    review_b: Receipt,
    adjudication: Receipt,
    documents: Vec<String>,
    reviewed_cases: Vec<ReviewedCase>,
    native_row_coverage_complete: bool,
    clean_scope_limits: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewedCase {
    name: String,
    review_a_row_sha256: String,
    review_b_row_sha256: String,
    adjudication_row_sha256: String,
    all_proposals_and_uncertainties_reviewed: bool,
    resolution_basis: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    name: String,
    input: String,
    country: String,
    expected: Vec<super::data::Span>,
    uncertainties: Vec<Value>,
    annotation_policy_sha256: String,
}

fn validate_proposal(value: &Value, row: &Row, policy: &str) -> anyhow::Result<()> {
    let proposal: Proposal = serde_json::from_value(value.clone())?;
    ensure!(
        proposal.name == row.name
            && proposal.input == row.input
            && proposal.country == "US"
            && proposal.annotation_policy_sha256 == policy,
        "independent proposal text/policy differs"
    );
    let mut spans: Vec<_> = proposal.expected.iter().collect();
    spans.sort_by_key(|s| (s.start, s.end));
    ensure!(
        spans
            .iter()
            .all(
                |s| ["person", "org", "address", "email", "phone"].contains(&s.kind.as_str())
                    && s.start < s.end
                    && row.input.get(s.start as usize..s.end as usize) == Some(s.text.as_str())
            )
            && spans.windows(2).all(|p| p[0].end <= p[1].start),
        "independent proposal spans are invalid or incomplete"
    );
    ensure!(
        proposal
            .uncertainties
            .iter()
            .all(|u| u.is_object() || u.is_string()),
        "independent proposal uncertainties must be explicit review subjects"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    ReviewedLossExclusion,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
/// The only accepted adjudicated unresolved form: a disclosed TRAIN loss-only exclusion.
struct Unresolved {
    start: u32,
    end: u32,
    quote: String,
    #[serde(rename = "disposition")]
    _disposition: Disposition,
    reason: String,
}

const RESOLVED: &str = "resolved";
const RESOLVED_WITH_EXCLUSIONS: &str = "resolved_with_reviewed_loss_exclusions";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
/// A source span the TRAIN adjudication left unresolved and declared loss-excluded, in UTF-8
/// bytes of the exact input. Only a fit objective that excludes it may consume such a row.
pub(crate) struct DeclaredExclusion {
    pub(crate) name: String,
    pub(crate) input_sha256: String,
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) quote: String,
    pub(crate) reason: String,
}

/// Parse an adjudication row's status and its typed unresolved spans for one cohort row.
fn declared(resolved: &Value, row: &Row, split: &str) -> anyhow::Result<Vec<DeclaredExclusion>> {
    let unresolved: Vec<Unresolved> = serde_json::from_value(resolved["unresolved"].clone())
        .with_context(|| format!("adjudication unresolved spans are not typed: {}", row.name))?;
    let status = resolved["status"].as_str().unwrap_or_default();
    ensure!(
        resolved["uncertainties"] == serde_json::json!([])
            && ((status == RESOLVED && unresolved.is_empty())
                || (status == RESOLVED_WITH_EXCLUSIONS && !unresolved.is_empty())),
        "adjudication status differs from its declared unresolved spans: {}",
        row.name
    );
    ensure!(
        unresolved.is_empty() || split == "train",
        "{split} cannot declare reviewed loss exclusions: {}",
        row.name
    );
    let mut spans: Vec<_> = unresolved
        .into_iter()
        .map(|u| {
            ensure!(
                u.start < u.end
                    && row.input.get(u.start as usize..u.end as usize) == Some(u.quote.as_str())
                    && !u.reason.trim().is_empty(),
                "declared loss exclusion quote/reason differs from input bytes: {}",
                row.name
            );
            ensure!(
                !row.expected
                    .iter()
                    .any(|s| s.start < u.end && u.start < s.end),
                "declared loss exclusion overlaps an expected span: {}",
                row.name
            );
            Ok(DeclaredExclusion {
                name: row.name.clone(),
                input_sha256: super::digest(row.input.as_bytes()),
                start: u.start,
                end: u.end,
                quote: u.quote,
                reason: u.reason,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    spans.sort();
    ensure!(
        spans.windows(2).all(|p| p[0].end <= p[1].start),
        "declared loss exclusions repeat or overlap: {}",
        row.name
    );
    Ok(spans)
}

fn evidence(receipt: &Receipt) -> anyhow::Result<BTreeMap<String, Value>> {
    let bytes = receipt.bytes()?;
    let mut result = BTreeMap::new();
    for line in std::str::from_utf8(&bytes)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        let name = row["name"]
            .as_str()
            .context("review evidence requires named cases")?
            .to_owned();
        ensure!(
            result.insert(name, row).is_none(),
            "review evidence repeats a case"
        );
    }
    Ok(result)
}

/// Verify a split's independent reviews and return its declared loss exclusions; DEV has none.
pub(super) fn verify(
    receipt: &Receipt,
    dataset: &Receipt,
    policy: &Receipt,
    split: &str,
    rows: &[Row],
) -> anyhow::Result<Vec<DeclaredExclusion>> {
    let review: Review = serde_json::from_slice(&receipt.bytes()?)?;
    let names: Vec<_> = rows.iter().map(|r| r.name.clone()).collect();
    ensure!(
        review.split == split
            && review.dataset_sha256 == dataset.sha256
            && review.policy_sha256 == policy.sha256
            && review.adjudication_complete
            && review.native_row_coverage_complete
            && review.documents == names
            && !review.clean_scope_limits.is_empty()
            && review
                .clean_scope_limits
                .iter()
                .all(|s| !s.trim().is_empty())
            && review.reviewers.len() >= 2
            && review.reviewers.iter().all(|s| !s.trim().is_empty())
            && review.reviewers.iter().collect::<BTreeSet<_>>().len() == review.reviewers.len()
            && review.review_a.sha256 != review.review_b.sha256,
        "{split} requires hash-bound independent reviews, exact resolved membership and disclosed native-row scope"
    );
    let a = evidence(&review.review_a)?;
    let b = evidence(&review.review_b)?;
    let adjudication = evidence(&review.adjudication)?;
    ensure!(
        review.reviewed_cases.len() == rows.len(),
        "adjudication coverage count differs"
    );
    let mut exclusions = Vec::new();
    for (row, coverage) in rows.iter().zip(&review.reviewed_cases) {
        for review in [&a, &b, &adjudication] {
            let reviewed = review
                .get(&row.name)
                .context("review evidence lacks a cohort case")?;
            ensure!(
                reviewed["input"] == row.input
                    && reviewed["country"] == "US"
                    && reviewed["annotation_policy_sha256"] == policy.sha256,
                "review evidence differs from complete cohort text/policy: {}",
                row.name
            );
        }
        let resolved = adjudication
            .get(&row.name)
            .context("adjudication case missing")?;
        let proposal_a = a
            .get(&row.name)
            .context("independent review A case missing")?;
        let proposal_b = b
            .get(&row.name)
            .context("independent review B case missing")?;
        validate_proposal(proposal_a, row, &policy.sha256)?;
        validate_proposal(proposal_b, row, &policy.sha256)?;
        ensure!(
            coverage.name == row.name
                && coverage.all_proposals_and_uncertainties_reviewed
                && !coverage.resolution_basis.trim().is_empty()
                && coverage.review_a_row_sha256 == super::digest(&serde_json::to_vec(proposal_a)?)
                && coverage.review_b_row_sha256 == super::digest(&serde_json::to_vec(proposal_b)?)
                && coverage.adjudication_row_sha256
                    == super::digest(&serde_json::to_vec(resolved)?),
            "adjudication lacks pinned per-case proposal/uncertainty resolution coverage"
        );
        ensure!(
            resolved["expected"] == serde_json::to_value(&row.expected)?,
            "cohort labels differ from resolved adjudication: {}",
            row.name
        );
        exclusions.extend(declared(resolved, row, split)?);
    }
    ensure!(
        review.unresolved_uncertainties == exclusions.len(),
        "{split} review discloses {} unresolved spans but adjudication declares {}",
        review.unresolved_uncertainties,
        exclusions.len()
    );
    Ok(exclusions)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    name: String,
    origin: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Checks {
    source_document_separation: bool,
    parent_and_family_separation: bool,
    alias_separation: bool,
    contact_separation: bool,
    historical_and_synthetic_exposure_reviewed: bool,
    domain_and_agency_exposure_reviewed: bool,
    complete_native_rows_verified: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Audit {
    scope: String,
    train_sha256: String,
    dev_sha256: String,
    policy_sha256: String,
    source_checkpoint_sha256: String,
    historical_closure: Vec<Receipt>,
    synthetic_closure: Vec<Receipt>,
    audit_outputs: Vec<Receipt>,
    declared_closure_complete: bool,
    unreviewed_weak_signals: usize,
    clean_scope_limits: Vec<String>,
    checks: Checks,
    train_documents: Vec<Member>,
    dev_documents: Vec<Member>,
}

fn membership(rows: &[Row], members: &[Member]) -> anyhow::Result<()> {
    ensure!(
        rows.len() == members.len(),
        "split audit document count differs"
    );
    for (row, member) in rows.iter().zip(members) {
        ensure!(
            row.name == member.name && serde_json::to_value(&row.origin)? == member.origin,
            "split audit native source membership differs: {}",
            row.name
        );
    }
    Ok(())
}

fn normalized(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn lineage(rows: &[Row]) -> BTreeSet<String> {
    rows.iter()
        .flat_map(|r| {
            let o = &r.origin;
            let mut ids = vec![format!("native-row:{}:{}", o.dataset.sha256, o.row_index)];
            for (_role, value) in [
                ("document", &o.source_document_id),
                ("parent", &o.parent_source_id),
                ("family", &o.source_family_id),
            ] {
                if let Some(value) = value {
                    ids.push(format!("lineage:{value}"));
                }
            }
            ids
        })
        .collect()
}

fn signals(rows: &[Row], contacts: bool) -> BTreeSet<String> {
    rows.iter()
        .flat_map(|row| {
            let declared = if contacts {
                &row.origin.contacts
            } else {
                &row.origin.aliases
            };
            declared
                .iter()
                .map(String::as_str)
                .chain(row.expected.iter().filter_map(|s| {
                    let contact = s.kind == "email" || s.kind == "phone";
                    let alias = s.kind == "person" || s.kind == "org";
                    ((contacts && contact) || (!contacts && alias)).then_some(s.text.as_str())
                }))
                .map(normalized)
                .collect::<Vec<_>>()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn person_identities(rows: &[Row]) -> BTreeSet<(String, String)> {
    rows.iter()
        .flat_map(|row| &row.expected)
        .filter(|span| span.kind == "person")
        .filter_map(|span| {
            let text = span
                .text
                .rsplit_once(',')
                .map_or(span.text.as_str(), |(name, tail)| {
                    let suffix = tail.trim().trim_end_matches('.').to_ascii_lowercase();
                    if matches!(suffix.as_str(), "jr" | "sr" | "ii" | "iii" | "iv") {
                        name.trim_end()
                    } else {
                        span.text.as_str()
                    }
                });
            crate::generate::person_key(text)
        })
        .collect()
}

pub(super) fn verify_separation(
    manifest: &Inputs<'_>,
    train: &[Row],
    dev: &[Row],
) -> anyhow::Result<()> {
    let audit: Audit = serde_json::from_slice(&manifest.separation_audit.bytes()?)?;
    ensure!(
        audit.scope == "reviewed-native-cohort-separation-v1"
            && audit.train_sha256 == manifest.train.sha256
            && audit.dev_sha256 == manifest.dev.sha256
            && audit.policy_sha256 == manifest.policy.sha256
            && audit.source_checkpoint_sha256 == manifest.checkpoint.sha256
            && audit.declared_closure_complete
            && audit.unreviewed_weak_signals == 0
            && !audit.historical_closure.is_empty()
            && !audit.synthetic_closure.is_empty()
            && !audit.audit_outputs.is_empty()
            && !audit.clean_scope_limits.is_empty()
            && audit
                .clean_scope_limits
                .iter()
                .all(|s| !s.trim().is_empty())
            && audit.checks.source_document_separation
            && audit.checks.parent_and_family_separation
            && audit.checks.alias_separation
            && audit.checks.contact_separation
            && audit.checks.historical_and_synthetic_exposure_reviewed
            && audit.checks.domain_and_agency_exposure_reviewed
            && audit.checks.complete_native_rows_verified,
        "fit requires a successful independently reviewed separation audit with pinned exposure closure and no unreviewed signals"
    );
    for receipt in audit
        .historical_closure
        .iter()
        .chain(&audit.synthetic_closure)
        .chain(&audit.audit_outputs)
    {
        receipt.bytes()?;
    }
    membership(train, &audit.train_documents)?;
    membership(dev, &audit.dev_documents)?;
    ensure!(
        lineage(train).is_disjoint(&lineage(dev)),
        "TRAIN/DEV share a native row, source document, parent or family"
    );
    ensure!(
        signals(train, true).is_disjoint(&signals(dev, true)),
        "TRAIN/DEV share contact exposure"
    );
    ensure!(
        signals(train, false).is_disjoint(&signals(dev, false)),
        "TRAIN/DEV share exact or normalized name/alias exposure"
    );
    ensure!(
        person_identities(train).is_disjoint(&person_identities(dev)),
        "TRAIN/DEV share a given-name/surname identity despite name order or middle-name differences"
    );
    Ok(())
}

pub(super) fn verify_nested_receipts(manifest: &Inputs) -> anyhow::Result<()> {
    for receipt in [&manifest.train_review, &manifest.dev_review] {
        let review: Review = serde_json::from_slice(&receipt.bytes()?)?;
        for receipt in [&review.review_a, &review.review_b, &review.adjudication] {
            receipt.bytes()?;
        }
    }
    let audit: Audit = serde_json::from_slice(&manifest.separation_audit.bytes()?)?;
    for receipt in audit
        .historical_closure
        .iter()
        .chain(&audit.synthetic_closure)
        .chain(&audit.audit_outputs)
    {
        receipt.bytes()?;
    }
    for receipt in [&manifest.train, &manifest.dev] {
        let bytes = receipt.bytes()?;
        let mut verified = BTreeSet::new();
        for line in std::str::from_utf8(&bytes)?
            .lines()
            .filter(|l| !l.trim().is_empty())
        {
            let row: Row = serde_json::from_str(line)?;
            if verified.insert((
                row.origin.dataset.path.clone(),
                row.origin.dataset.sha256.clone(),
            )) {
                row.origin.dataset.bytes()?;
            }
        }
    }
    Ok(())
}

/// Collect exactly the native/review/separation receipts checked by this validator.
pub(super) fn dependency_receipts(manifest: &Inputs<'_>) -> anyhow::Result<Vec<Receipt>> {
    verify_nested_receipts(manifest)?;
    let mut receipts = Vec::new();
    for receipt in [
        manifest.policy,
        manifest.checkpoint,
        manifest.train,
        manifest.dev,
        manifest.train_review,
        manifest.dev_review,
        manifest.separation_audit,
    ] {
        receipts.push(receipt.clone());
    }
    for receipt in [manifest.train_review, manifest.dev_review] {
        let review: Review = serde_json::from_slice(&receipt.bytes()?)?;
        receipts.extend([review.review_a, review.review_b, review.adjudication]);
    }
    let audit: Audit = serde_json::from_slice(&manifest.separation_audit.bytes()?)?;
    receipts.extend(audit.historical_closure);
    receipts.extend(audit.synthetic_closure);
    receipts.extend(audit.audit_outputs);
    for receipt in [manifest.train, manifest.dev] {
        for line in std::str::from_utf8(&receipt.bytes()?)?
            .lines()
            .filter(|l| !l.trim().is_empty())
        {
            let row: Row = serde_json::from_str(line)?;
            receipts.push(row.origin.dataset);
        }
    }
    Ok(receipts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, text: &str, index: usize) -> Row {
        serde_json::from_value(serde_json::json!({"name":name,"country":"US","input":text,
                "status":"resolved","expected":[],"uncertainties":[],"unresolved":[],"annotation_policy_sha256":"policy",
                "origin":{"dataset":{"path":"/unused/source","sha256":"source"},"row_index":index,"row_id":name,
                "text_sha256":"text","source_document_id":null,"parent_source_id":"parent-123",
                "source_family_id":"same-template","source_group":null,"aliases":[],"contacts":[]}})).unwrap()
    }

    #[test]
    fn differently_named_cropped_rows_cannot_hide_shared_parent_or_family() {
        let train = vec![row("original", "Longer original paragraph.", 0)];
        let dev = vec![row("renamed", "original paragraph.", 1)];
        assert!(!lineage(&train).is_disjoint(&lineage(&dev)));
        let mut train = train;
        let mut dev = dev;
        train[0].origin.source_family_id = None;
        dev[0].origin.source_family_id = None;
        train[0].origin.source_document_id = Some("shared-doc".into());
        train[0].origin.parent_source_id = None;
        dev[0].origin.source_document_id = None;
        dev[0].origin.parent_source_id = Some("shared-doc".into());
        assert!(!lineage(&train).is_disjoint(&lineage(&dev)));
    }

    #[test]
    fn surname_first_and_middle_name_variants_are_shared_person_exposure() {
        fn person(text: &str, index: usize) -> Row {
            let mut row = row("case", text, index);
            row.expected.push(super::super::data::Span {
                kind: "person".into(),
                text: text.into(),
                start: 0,
                end: u32::try_from(text.len()).unwrap(),
            });
            row
        }
        for (first, second, shared) in [
            ("Jacobsen, Kelly", "Kelly Jacobsen", true),
            ("Anthony T. Lee Jr.", "Anthony Lee", true),
            ("Anthony Lee, Jr.", "Anthony Lee", true),
            ("Lee, Anthony, Jr.", "Anthony Lee", true),
            ("Jane Doe", "John Doe", false),
        ] {
            let train = vec![person(first, 0)];
            let dev = vec![person(second, 1)];
            assert!(signals(&train, false).is_disjoint(&signals(&dev, false)));
            assert_eq!(
                !person_identities(&train).is_disjoint(&person_identities(&dev)),
                shared
            );
        }
    }

    const INPUT: &str = "the Respondent met Jane";

    /// Run full review verification for one case with the given adjudication outcome.
    fn verify_case(
        split: &str,
        status: &str,
        unresolved: serde_json::Value,
        disclosed: usize,
    ) -> anyhow::Result<Vec<DeclaredExclusion>> {
        use serde_json::json;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            Receipt {
                path,
                sha256: super::super::digest(bytes),
            }
        };
        let policy = write("policy.json", b"{}");
        let dataset = write("dataset.jsonl", b"rows");
        let proposal = json!({"name":"case","input":INPUT,"country":"US","expected":[],
            "uncertainties":[],"annotation_policy_sha256":policy.sha256});
        let adjudication = json!({"name":"case","input":INPUT,"country":"US","expected":[],
            "annotation_policy_sha256":policy.sha256,"status":status,"uncertainties":[],
            "unresolved":unresolved});
        let line = |v: &serde_json::Value| serde_json::to_vec(v).unwrap();
        let review_a = write("a.jsonl", &line(&proposal));
        let review_b = write("b.jsonl", &[line(&proposal), b"\n".to_vec()].concat());
        let adjudicated = write("adjudication.jsonl", &line(&adjudication));
        let hash = |v: &serde_json::Value| super::super::digest(&line(v));
        let review = json!({"split":split,"dataset_sha256":dataset.sha256,"policy_sha256":policy.sha256,
            "adjudication_complete":true,"unresolved_uncertainties":disclosed,"reviewers":["a","b"],
            "review_a":review_a,"review_b":review_b,"adjudication":adjudicated,"documents":["case"],
            "reviewed_cases":[{"name":"case","review_a_row_sha256":hash(&proposal),
                "review_b_row_sha256":hash(&proposal),"adjudication_row_sha256":hash(&adjudication),
                "all_proposals_and_uncertainties_reviewed":true,"resolution_basis":"two reviews"}],
            "native_row_coverage_complete":true,"clean_scope_limits":["model-agent review"]});
        let receipt = write("review.json", &line(&review));
        verify(&receipt, &dataset, &policy, split, &[row("case", INPUT, 0)])
    }

    fn respondent() -> serde_json::Value {
        serde_json::json!([{"start":4,"end":14,"quote":"Respondent",
            "disposition":"reviewed_loss_exclusion","reason":"reviewers split on ORG"}])
    }

    #[test]
    fn train_review_exposes_typed_declared_loss_exclusions() {
        assert!(
            verify_case("train", RESOLVED, serde_json::json!([]), 0)
                .unwrap()
                .is_empty()
        );
        let declared = verify_case("train", RESOLVED_WITH_EXCLUSIONS, respondent(), 1).unwrap();
        assert_eq!(
            declared,
            vec![DeclaredExclusion {
                name: "case".into(),
                input_sha256: super::super::digest(INPUT.as_bytes()),
                start: 4,
                end: 14,
                quote: "Respondent".into(),
                reason: "reviewers split on ORG".into(),
            }]
        );
    }

    #[test]
    fn declared_loss_exclusions_reject_dev_counts_status_and_untyped_forms() {
        let typed = RESOLVED_WITH_EXCLUSIONS;
        assert!(verify_case("dev", typed, respondent(), 1).is_err());
        assert!(verify_case("dev", typed, respondent(), 0).is_err());
        assert!(verify_case("train", typed, respondent(), 0).is_err());
        assert!(verify_case("train", typed, respondent(), 2).is_err());
        assert!(verify_case("train", RESOLVED, serde_json::json!([]), 1).is_err());
        assert!(verify_case("train", RESOLVED, respondent(), 1).is_err());
        assert!(verify_case("train", typed, serde_json::json!([]), 0).is_err());
        assert!(verify_case("train", "unresolved", respondent(), 1).is_err());
        let mut variants = Vec::new();
        for (field, value) in [
            ("quote", serde_json::json!("respondent")),
            ("end", serde_json::json!(13)),
            ("disposition", serde_json::json!("label_o")),
            ("reason", serde_json::json!(" ")),
            ("label", serde_json::json!("org")),
        ] {
            let mut changed = respondent();
            changed[0][field] = value;
            variants.push(changed);
        }
        let mut repeated = respondent();
        repeated
            .as_array_mut()
            .unwrap()
            .push(respondent()[0].clone());
        variants.push(repeated);
        for changed in variants {
            assert!(
                verify_case(
                    "train",
                    typed,
                    changed.clone(),
                    changed.as_array().unwrap().len()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn declared_loss_exclusion_cannot_overlap_an_expected_span() {
        let mut case = row("case", "The Respondent agrees.", 0);
        let adjudication = serde_json::json!({"status":RESOLVED_WITH_EXCLUSIONS,"uncertainties":[],
            "unresolved":[{"start":4,"end":14,"quote":"Respondent",
            "disposition":"reviewed_loss_exclusion","reason":"reviewers split on ORG"}]});
        assert_eq!(declared(&adjudication, &case, "train").unwrap().len(), 1);
        case.expected.push(super::super::data::Span {
            kind: "person".into(),
            text: "Respondent agrees".into(),
            start: 4,
            end: 21,
        });
        assert!(declared(&adjudication, &case, "train").is_err());
    }

    #[test]
    fn old_boolean_only_review_receipt_is_not_readiness_evidence() {
        assert!(
            serde_json::from_value::<Review>(
                serde_json::json!({"split":"train","dataset_sha256":"data",
            "policy_sha256":"policy","adjudication_complete":true,"unresolved_uncertainties":0,
            "reviewers":["a","b"],"whole_documents":true,"provenance":"claimed review"})
            )
            .is_err()
        );
    }

    #[test]
    fn text_identity_without_annotation_proposals_is_not_an_independent_review() {
        assert!(
            serde_json::from_value::<Proposal>(serde_json::json!({"name":"case","input":"Text.",
            "country":"US","annotation_policy_sha256":"policy"}))
            .is_err()
        );
    }
}
