//! Whole-document TRAIN supervision and gold-independent DEV feature loading.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::readiness::DeclaredExclusion;
use super::{Inputs, Receipt, digest};
use crate::config::Config;
use crate::detector::{self, DetectorDoc};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// One literal entity occurrence, with offsets in UTF-8 bytes.
pub(crate) struct Span {
    pub(crate) kind: String,
    pub(crate) text: String,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Audited native row identity and recorded source relationships; absent metadata stays absent.
pub(super) struct Origin {
    pub(super) dataset: Receipt,
    pub(super) row_index: usize,
    pub(super) row_id: String,
    pub(super) text_sha256: String,
    pub(super) source_document_id: Option<String>,
    pub(super) parent_source_id: Option<String>,
    pub(super) source_family_id: Option<String>,
    pub(super) source_group: Option<String>,
    pub(super) domain: Option<String>,
    pub(super) agency: Option<String>,
    pub(super) aliases: Vec<String>,
    pub(super) contacts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Row {
    pub(super) name: String,
    country: String,
    pub(super) input: String,
    pub(crate) expected: Vec<Span>,
    status: String,
    uncertainties: Vec<Value>,
    unresolved: Vec<Value>,
    annotation_policy_sha256: String,
    pub(super) origin: Origin,
}

/// Complete native documents and their aligned reviewed supervision.
pub(crate) struct Cohort {
    pub(crate) names: Vec<String>,
    pub(crate) docs: Vec<DetectorDoc>,
    pub(crate) expected: Vec<Vec<Span>>,
    origins: Vec<Origin>,
    exclusions: Vec<DeclaredExclusion>,
}

#[cfg(test)]
impl Cohort {
    /// Unsupervised rows without native origins, for encoder comparisons in other modules.
    pub(crate) fn unreviewed_for_test(names: Vec<String>, docs: Vec<DetectorDoc>) -> Self {
        Self {
            expected: vec![Vec::new(); docs.len()],
            names,
            docs,
            origins: Vec::new(),
            exclusions: Vec::new(),
        }
    }
}

impl Cohort {
    /// Adjudicated unresolved spans declared loss-excluded; empty unless TRAIN review says so.
    pub(crate) fn declared_exclusions(&self) -> &[DeclaredExclusion] {
        &self.exclusions
    }

    /// Refuse rows whose supervision is incomplete for consumers without a reviewed exclusion objective.
    pub(crate) fn refuse_declared_exclusions(&self, consumer: &str) -> anyhow::Result<()> {
        ensure!(
            self.exclusions.is_empty(),
            "{consumer} cannot consume {} reviewed loss exclusions; their rows are not fully supervised",
            self.exclusions.len()
        );
        Ok(())
    }

    /// Retain complete native rows for a TRAIN-only observational learning probe.
    pub(crate) fn subset(&self, indices: &[usize]) -> anyhow::Result<Self> {
        ensure!(!indices.is_empty(), "native probe cannot be empty");
        let mut seen = BTreeSet::new();
        let mut subset = Self {
            names: Vec::new(),
            docs: Vec::new(),
            expected: Vec::new(),
            origins: Vec::new(),
            exclusions: Vec::new(),
        };
        for &index in indices {
            ensure!(
                index < self.docs.len() && seen.insert(index),
                "probe index missing or repeated"
            );
            subset
                .names
                .push(self.names.get(index).context("probe name missing")?.clone());
            subset.docs.push(self.docs[index].clone());
            subset.expected.push(
                self.expected
                    .get(index)
                    .context("probe gold missing")?
                    .clone(),
            );
            subset.origins.push(
                self.origins
                    .get(index)
                    .context("probe origin missing")?
                    .clone(),
            );
        }
        subset.exclusions = self
            .exclusions
            .iter()
            .filter(|e| subset.names.contains(&e.name))
            .cloned()
            .collect();
        Ok(subset)
    }

    /// Append complete rows without reconstructing or dropping their native origins.
    pub(super) fn append(&mut self, other: Self) {
        self.names.extend(other.names);
        self.docs.extend(other.docs);
        self.expected.extend(other.expected);
        self.origins.extend(other.origins);
        self.exclusions.extend(other.exclusions);
    }

    /// Rows without declared exclusions keep their historical identity record unchanged.
    fn identity(&self) -> anyhow::Result<Value> {
        let records = self.names.iter().zip(&self.docs).zip(&self.expected).zip(&self.origins).map(|(((name, doc), gold), origin)| {
            let mut record = json!({
                "name":name, "text_sha256":digest(doc.text.as_bytes()), "retained_tokens":doc.enc.labels.len(),
                "labels_sha256":digest(&serde_json::to_vec(&doc.enc.labels)?),
                "expected_sha256":digest(&serde_json::to_vec(gold)?),
                "complete_native_source_row":true, "origin":origin,
            });
            let declared: Vec<_> = self.exclusions.iter().filter(|e| &e.name == name).collect();
            if !declared.is_empty() {
                record["declared_reviewed_loss_exclusions"] = serde_json::to_value(declared)?;
            }
            Ok(record)
        }).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(json!(records))
    }
}

/// Reviewed training documents paired with gold-independent development features.
pub(crate) struct Cohorts {
    pub(crate) train: Cohort,
    pub(crate) dev: Cohort,
}

impl Cohorts {
    pub(crate) fn identity(&self) -> anyhow::Result<Value> {
        Ok(json!({"train":self.train.identity()?, "dev":self.dev.identity()?}))
    }
}

fn rows(bytes: &[u8], policy: &str) -> anyhow::Result<Vec<Row>> {
    let mut names = BTreeSet::new();
    let mut texts = BTreeSet::new();
    let mut source_bytes = BTreeMap::new();
    std::str::from_utf8(bytes)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            let row: Row =
                serde_json::from_str(line).with_context(|| format!("cohort row{}", index + 1))?;
            ensure!(
                row.input.len() <= u32::MAX as usize
                    && !row.name.trim().is_empty()
                    && names.insert(row.name.clone())
                    && texts.insert(normalized_text(&row.input))
                    && row.country == "US"
                    && row.annotation_policy_sha256 == policy
                    && row.status == "resolved"
                    && row.uncertainties.is_empty()
                    && row.unresolved.is_empty(),
                "cohort row{} duplicates an identity or lacks resolved US policy annotations",
                index + 1
            );
            let mut ordered: Vec<_> = row.expected.iter().collect();
            ordered.sort_by_key(|span| (span.start, span.end));
            for span in &ordered {
                ensure!(
                    ["person", "org", "address", "email", "phone"].contains(&span.kind.as_str())
                        && span.start < span.end
                        && row.input.get(span.start as usize..span.end as usize)
                            == Some(span.text.as_str()),
                    "cohort row{} has an unknown kind or invalid UTF-8 exact quote",
                    index + 1
                );
            }
            ensure!(
                ordered.windows(2).all(|pair| pair[0].end <= pair[1].start),
                "cohort row{} has overlapping or duplicate spans",
                index + 1
            );
            verify_origin(&row, &mut source_bytes)?;
            Ok(row)
        })
        .collect()
}

fn normalized_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn validate_docs(docs: &[DetectorDoc]) -> anyhow::Result<()> {
    ensure!(
        docs.iter()
            .all(|doc| !doc.enc.labels.is_empty() && doc.enc.labels.len() <= 900),
        "cohort pilot accepts only nonempty whole documents with at most900 retained tokens; no truncation"
    );
    Ok(())
}

fn training(rows: &[Row], cfg: &Config) -> anyhow::Result<Vec<DetectorDoc>> {
    rows.iter()
        .map(|row| {
            detector::annotated_document(
                &row.input,
                &row.country,
                &serde_json::to_string(&row.expected)?,
                &cfg.features.to_tessera(),
            )
            .with_context(|| format!("unreachable TRAIN gold in {}", row.name))
        })
        .collect()
}

fn verify_origin(
    row: &Row,
    cache: &mut BTreeMap<(std::path::PathBuf, String), Vec<u8>>,
) -> anyhow::Result<()> {
    let origin = &row.origin;
    ensure!(
        !origin.row_id.trim().is_empty() && digest(row.input.as_bytes()) == origin.text_sha256,
        "cohort native source row text identity is missing or changed"
    );
    let key = (origin.dataset.path.clone(), origin.dataset.sha256.clone());
    if !cache.contains_key(&key) {
        cache.insert(key.clone(), origin.dataset.bytes()?);
    }
    let bytes = cache.get(&key).context("native source cache missing")?;
    let original: Value = serde_json::from_str(
        std::str::from_utf8(bytes)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .nth(origin.row_index)
            .context("original native row index missing")?,
    )?;
    let text = original
        .get("input")
        .or_else(|| original.get("text"))
        .and_then(Value::as_str)
        .context("original native row text missing")?;
    ensure!(
        text == row.input && original["country"] == "US",
        "cohort is cropped or differs from the complete original native source row"
    );
    if let Some(id) = original
        .get("name")
        .or_else(|| original.get("id"))
        .and_then(Value::as_str)
    {
        ensure!(id == origin.row_id, "native source row id differs");
    }
    if let Some(parent) = original.get("source_document_key").and_then(Value::as_str) {
        ensure!(
            origin.parent_source_id.as_deref() == Some(parent)
                || origin.source_document_id.as_deref() == Some(parent),
            "native source document key must be bound as document/parent lineage, separately from row id"
        );
    }
    for (key, value) in [
        ("source_document_id", &origin.source_document_id),
        ("parent_source_id", &origin.parent_source_id),
        ("source_family_id", &origin.source_family_id),
        ("source_group", &origin.source_group),
        ("domain", &origin.domain),
        ("agency", &origin.agency),
    ] {
        if let Some(original) = original.get(key).and_then(Value::as_str) {
            ensure!(
                value.as_deref() == Some(original),
                "native source {key} differs from its audited origin metadata"
            );
        }
    }
    for id in [
        &origin.source_document_id,
        &origin.parent_source_id,
        &origin.source_family_id,
    ] {
        ensure!(
            id.as_ref().is_none_or(|s| !s.trim().is_empty()),
            "empty source lineage identity"
        );
    }
    Ok(())
}

fn cohort(rows: Vec<Row>, docs: Vec<DetectorDoc>, exclusions: Vec<DeclaredExclusion>) -> Cohort {
    let names = rows.iter().map(|r| r.name.clone()).collect();
    let (expected, origins) = rows.into_iter().map(|r| (r.expected, r.origin)).unzip();
    Cohort {
        names,
        docs,
        expected,
        origins,
        exclusions,
    }
}

pub(super) fn load(manifest: &Inputs<'_>, cfg: &Config) -> anyhow::Result<Cohorts> {
    load_checked(manifest, cfg, true)
}

/// Validate a complete segment; its caller must enforce TRAIN support on the composed union.
pub(super) fn load_composed(manifest: &Inputs<'_>, cfg: &Config) -> anyhow::Result<Cohorts> {
    load_checked(manifest, cfg, false)
}

/// Require actual supervised support for each neural kind in a complete split.
pub(super) fn require_neural_support(docs: &[DetectorDoc], split: &str) -> anyhow::Result<()> {
    let mut support = BTreeMap::new();
    for gold in docs.iter().flat_map(|doc| &doc.gold) {
        *support.entry(gold.kind).or_insert(0usize) += 1;
    }
    ensure!(
        (0..3).all(|kind| support.get(&kind).copied().unwrap_or(0) > 0),
        "{split} must support PERSON, ORG and ADDRESS"
    );
    Ok(())
}

fn load_checked(
    manifest: &Inputs<'_>,
    cfg: &Config,
    require_train_support: bool,
) -> anyhow::Result<Cohorts> {
    let train_rows = rows(&manifest.train.bytes()?, &manifest.policy.sha256)?;
    let dev_rows = rows(&manifest.dev.bytes()?, &manifest.policy.sha256)?;
    let train_exclusions = super::readiness::verify(
        manifest.train_review,
        manifest.train,
        manifest.policy,
        "train",
        &train_rows,
    )?;
    ensure!(
        super::readiness::verify(
            manifest.dev_review,
            manifest.dev,
            manifest.policy,
            "dev",
            &dev_rows,
        )?
        .is_empty(),
        "DEV cannot declare reviewed loss exclusions"
    );
    super::readiness::verify_separation(manifest, &train_rows, &dev_rows)?;
    ensure!(
        train_rows.len() == manifest.train_documents && dev_rows.len() == manifest.dev_documents,
        "cohort resolved counts differ from their preregistered manifest"
    );
    let train_names: BTreeSet<_> = train_rows.iter().map(|r| r.name.as_str()).collect();
    let train_texts: BTreeSet<_> = train_rows
        .iter()
        .map(|r| normalized_text(&r.input))
        .collect();
    ensure!(
        dev_rows
            .iter()
            .all(|r| !train_names.contains(r.name.as_str())
                && !train_texts.contains(&normalized_text(&r.input))),
        "DEV overlaps TRAIN by document name or normalized whole text"
    );
    let train_docs = training(&train_rows, cfg)?;
    let dev_docs = detector::load_development_gold(&manifest.dev.path, &cfg.features.to_tessera())?;
    if require_train_support {
        require_neural_support(&train_docs, "TRAIN")?;
    }
    require_neural_support(&dev_docs, "DEV")?;
    manifest.dev.bytes()?;
    ensure!(
        dev_docs.len() == dev_rows.len()
            && dev_docs
                .iter()
                .zip(&dev_rows)
                .all(|(d, r)| d.text == r.input)
            && dev_docs
                .iter()
                .all(|d| d.enc.labels.iter().all(|&label| label == 0)),
        "DEV features must be gold-independent and aligned with its reviewed rows"
    );
    validate_docs(&train_docs)?;
    validate_docs(&dev_docs)?;
    Ok(Cohorts {
        train: cohort(train_rows, train_docs, train_exclusions),
        dev: cohort(dev_rows, dev_docs, Vec::new()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRow {
        _dir: tempfile::TempDir,
        bytes: Vec<u8>,
    }
    impl std::ops::Deref for TestRow {
        type Target = [u8];
        fn deref(&self) -> &[u8] {
            &self.bytes
        }
    }
    fn row(input: &str, spans: Value) -> TestRow {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.jsonl");
        let original =
            serde_json::to_vec(&json!({"name":"case","country":"US","input":input})).unwrap();
        std::fs::write(&path, &original).unwrap();
        let bytes = serde_json::to_vec(&json!({"name":"case", "country":"US", "input":input,
            "expected":spans, "status":"resolved", "uncertainties":[], "unresolved":[],
            "annotation_policy_sha256":"policy", "origin":{"dataset":{"path":path,"sha256":digest(&original)},
            "row_index":0,"row_id":"case","text_sha256":digest(input.as_bytes()),
            "source_document_id":null,"parent_source_id":null,"source_family_id":null,"aliases":[],"contacts":[]}})).unwrap();
        TestRow { _dir: dir, bytes }
    }

    #[test]
    fn declared_exclusions_are_refused_kept_by_subsets_and_absent_from_plain_identity() {
        let cfg = crate::config::learning_test_config();
        let input = "The Respondent agrees.";
        let bytes = row(input, json!([]));
        let plain_rows = rows(&bytes, "policy").unwrap();
        let docs = training(&plain_rows, &cfg).unwrap();
        let plain = cohort(plain_rows, docs.clone(), Vec::new());
        assert!(plain.refuse_declared_exclusions("consumer").is_ok());
        assert!(
            plain.identity().unwrap()[0]
                .get("declared_reviewed_loss_exclusions")
                .is_none()
        );
        let exclusion = DeclaredExclusion {
            name: "case".into(),
            input_sha256: digest(input.as_bytes()),
            start: 4,
            end: 14,
            quote: "Respondent".into(),
            reason: "reviewers split on ORG".into(),
        };
        let declared = cohort(
            rows(&bytes, "policy").unwrap(),
            docs,
            vec![exclusion.clone()],
        );
        assert!(declared.refuse_declared_exclusions("consumer").is_err());
        assert_eq!(
            declared.identity().unwrap()[0]["declared_reviewed_loss_exclusions"],
            json!([exclusion])
        );
        assert_eq!(
            declared.subset(&[0]).unwrap().declared_exclusions(),
            &[exclusion]
        );
    }

    #[test]
    fn train_has_bio_targets_and_dev_features_do_not_consume_gold() {
        let cfg = crate::config::learning_test_config();
        let input = "Jane Doe works here.";
        let bytes = row(
            input,
            json!([{"kind":"person", "text":"Jane Doe", "start":0, "end":8}]),
        );
        let rows = rows(&bytes, "policy").unwrap();
        let train = training(&rows, &cfg).unwrap();
        assert_eq!(&train[0].enc.labels[..2], &[1, 2]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dev.jsonl");
        std::fs::write(&path, &bytes.bytes).unwrap();
        let dev = detector::load_development_gold(&path, &cfg.features.to_tessera()).unwrap();
        assert!(dev[0].enc.labels.iter().all(|&label| label == 0));
        assert_eq!(train[0].enc.ngram_ids, dev[0].enc.ngram_ids);
        assert_eq!(train[0].enc.flags, dev[0].enc.flags);
        assert_eq!(train[0].gold, dev[0].gold);
    }

    #[test]
    fn byte_boundaries_overlap_and_rule_unreachability_are_guarded() {
        let bad = row(
            "Zoë Smith",
            json!([{"kind":"person", "text":"Zo", "start":0, "end":3}]),
        );
        assert!(rows(&bad, "policy").is_err());
        let duplicate = row(
            "Jane Doe",
            json!([
            {"kind":"person", "text":"Jane Doe", "start":0, "end":8},
            {"kind":"person", "text":"Jane Doe", "start":0, "end":8}]),
        );
        assert!(rows(&duplicate, "policy").is_err());
        let cfg = crate::config::learning_test_config();
        let unreachable = row(
            "jane@example.org",
            json!([{"kind":"org", "text":"jane@example.org", "start":0, "end":16}]),
        );
        assert!(training(&rows(&unreachable, "policy").unwrap(), &cfg).is_err());
    }

    #[test]
    fn whole_document_limit_rejects_instead_of_truncating() {
        let cfg = crate::config::learning_test_config();
        let input = vec!["word"; 901].join(" ");
        let docs = training(&rows(&row(&input, json!([])), "policy").unwrap(), &cfg).unwrap();
        assert_eq!(docs[0].enc.labels.len(), 901);
        assert!(validate_docs(&docs).is_err());
    }

    #[test]
    fn quarantine_unresolved_missing_origin_and_cropped_native_rows_are_rejected() {
        let original = row("Full native paragraph.", json!([]));
        let value: Value = serde_json::from_slice(&original).unwrap();
        for (key, bad) in [
            ("status", json!("quarantine")),
            ("unresolved", json!(["ambiguous"])),
            ("input", json!("native paragraph.")),
        ] {
            let mut changed = value.clone();
            changed[key] = bad;
            assert!(rows(&serde_json::to_vec(&changed).unwrap(), "policy").is_err());
        }
        for key in ["status", "uncertainties", "unresolved", "origin"] {
            let mut changed = value.clone();
            changed.as_object_mut().unwrap().remove(key);
            assert!(rows(&serde_json::to_vec(&changed).unwrap(), "policy").is_err());
        }
    }

    #[test]
    fn declared_native_parent_and_family_cannot_be_replaced() {
        let original = row("Full native paragraph.", json!([]));
        let mut value: Value = serde_json::from_slice(&original).unwrap();
        let path = std::path::PathBuf::from(value["origin"]["dataset"]["path"].as_str().unwrap());
        let mut native: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for key in ["source_document_id", "parent_source_id", "source_family_id"] {
            native[key] = json!(format!("native-{key}"));
            value["origin"][key] = native[key].clone();
        }
        let bytes = serde_json::to_vec(&native).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        value["origin"]["dataset"]["sha256"] = json!(digest(&bytes));
        assert!(rows(&serde_json::to_vec(&value).unwrap(), "policy").is_ok());
        for key in ["source_document_id", "parent_source_id", "source_family_id"] {
            let mut changed = value.clone();
            changed["origin"][key] = json!("different-source");
            assert!(rows(&serde_json::to_vec(&changed).unwrap(), "policy").is_err());
        }
    }
}

#[cfg(test)]
mod composed_support_tests {
    use super::*;
    use crate::detector::KindSpan;

    fn doc(text: &str, kind: Option<usize>) -> DetectorDoc {
        let gold: Vec<_> = kind
            .into_iter()
            .map(|kind| KindSpan {
                kind,
                start: 0,
                end: text.len() as u32,
            })
            .collect();
        let cfg = crate::config::learning_test_config();
        DetectorDoc {
            text: text.to_owned(),
            enc: detector::encode_document(text, &gold, &cfg.features.to_tessera()).unwrap(),
            gold,
            breaks: detector::breaks_of(text),
        }
    }

    #[test]
    fn negative_and_partial_segments_need_union_support_and_dev_stays_supported() {
        let negative = vec![doc("County: Adams", None)];
        assert!(require_neural_support(&negative, "legacy TRAIN").is_err());
        let mut union = negative;
        union.push(doc("Jane Doe", Some(0)));
        assert!(require_neural_support(&union, "composed TRAIN").is_err());
        union.push(doc("River Office", Some(1)));
        assert!(require_neural_support(&union, "DEV").is_err());
        union.push(doc("1 Main Street", Some(2)));
        require_neural_support(&union, "composed TRAIN").unwrap();
        require_neural_support(&union, "DEV").unwrap();
    }
}
