//! Explicit whole-original PERSON cohort for a single bounded learnability diagnostic.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{OriginalRow, Prepared, form, original_gold, support};
use crate::config::Config;
use crate::detector;

pub(crate) const SCOPE: &str = "whole-original-person-learnability-v1";
use super::exposure::INITIAL_SHA;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cohort {
    scope: String,
    source_config: String,
    config_sha256: String,
    expected_initial_sha256: String,
    documents: Vec<Row>,
    separation_receipt: Option<Receipt>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    name: String,
    role: String,
    source_path: String,
    source_line: usize,
    source_sha256: String,
    row_sha256: String,
    text_sha256: String,
    model_gold_sha256: String,
    content_tokens: usize,
}

const ORIGINALS: [(&str, usize, &str); 15] = [
    (
        "data/interim/silver/us-person-long-reviewed-v2.jsonl",
        3,
        "biography",
    ),
    (
        "data/interim/silver/us-person-long-reviewed-v2.jsonl",
        4,
        "biography",
    ),
    (
        "data/interim/silver/us-person-long-reviewed-v3.jsonl",
        4,
        "biography",
    ),
    (
        "data/interim/silver/us-person-long-reviewed-v3.jsonl",
        1,
        "biography",
    ),
    (
        "data/interim/silver/us-person-keltner-policy-reviewed-v1.jsonl",
        1,
        "biography",
    ),
    (
        "data/interim/silver/us-federal-org-reviewed-safe-v2.jsonl",
        102,
        "short-singleton",
    ),
    (
        "data/interim/silver/us-address-prefix-reviewed-v1.jsonl",
        3,
        "short-singleton",
    ),
    (
        "data/interim/silver/us-address-prefix-reviewed-v1.jsonl",
        13,
        "short-singleton",
    ),
    (
        "data/interim/silver/us-dol-whd-reviewed-offices-v1.jsonl",
        8,
        "short-multiword",
    ),
    (
        "data/interim/silver/us-reviewed-contact-snippets-safe-v1.jsonl",
        30,
        "short-multiword",
    ),
    (
        "data/interim/silver/us-v4-reviewed-additions-strict-v2.jsonl",
        17,
        "short-multiword",
    ),
    (
        "data/interim/silver/us-v4-reviewed-additions-strict-v2.jsonl",
        59,
        "model-negative",
    ),
    (
        "data/interim/silver/us-federal-org-reviewed-safe-v2.jsonl",
        100,
        "model-negative",
    ),
    (
        "data/interim/silver/us-data-corrections-v1/us-2025-year-contact-expansion-safe-v1.jsonl",
        17,
        "model-negative",
    ),
    (
        "data/interim/silver/us-environmental-hard-negatives-v1.jsonl",
        4,
        "model-negative",
    ),
];

fn validate_cohort(cohort: &Cohort) -> anyhow::Result<()> {
    ensure!(
        cohort.scope == SCOPE
            && cohort.documents.len() == ORIGINALS.len()
            && cohort.expected_initial_sha256 == INITIAL_SHA,
        "unsupported frozen memorization cohort"
    );
    for (row, (path, line, role)) in cohort.documents.iter().zip(ORIGINALS) {
        ensure!(
            row.source_path == path && row.source_line == line && row.role == role,
            "frozen PERSON originals, roles or order changed"
        );
    }
    Ok(())
}

fn verify_singleton_control(singletons: usize) -> anyhow::Result<()> {
    ensure!(
        singletons == 1,
        "short singleton control changed; retain all original additional gold"
    );
    Ok(())
}

fn validate_row(row: &Row, raw: &str) -> anyhow::Result<OriginalRow> {
    ensure!(
        crate::export::sha256_hex(raw.as_bytes()) == row.row_sha256,
        "frozen row changed"
    );
    let original: OriginalRow = serde_json::from_str(raw)?;
    ensure!(
        original.provenance["country"] == "US"
            && crate::export::sha256_hex(original.text.as_bytes()) == row.text_sha256,
        "frozen text is changed or non-US"
    );
    let identity: Vec<_> = original_gold(&original)
        .iter()
        .map(|g| json!([detector::KINDS[g.kind].as_str(), g.start, g.end]))
        .collect();
    ensure!(
        crate::export::sha256_hex(&serde_json::to_vec(&identity)?) == row.model_gold_sha256,
        "frozen native model gold changed"
    );
    Ok(original)
}

fn validate_receipt(cohort: &Cohort, approved: &Value) -> anyhow::Result<()> {
    ensure!(
        approved["scope"] == SCOPE
            && approved["approved_for_training_only_person15"] == true
            && approved["no_held_expansion"] == true
            && approved["whole_native_text_and_gold_verified"] == true
            && approved["cohort_sha256_without_receipt"] == content_identity(cohort)?,
        "frozen cohort separation/context review did not pass"
    );
    Ok(())
}

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&std::fs::read(path)?))
}

fn contract(cfg: &Config) -> Value {
    json!({
        "task": cfg.task, "seed": cfg.seed, "features": cfg.features, "net": cfg.net,
        "batch_size": cfg.train.batch_size, "learning_rate": cfg.train.learning_rate,
        "warmup_steps": cfg.train.warmup_steps,
        "gradient_clip_norm": cfg.train.gradient_clip_norm,
        "horizon": cfg.train.diagnostic_schedule_steps,
        "class_weights": cfg.detector.as_ref().map(|d| &d.class_weights),
    })
}

/// Reconstruct only the fifteen frozen original rows through the native silver loader.
pub(crate) fn prepare_frozen(cfg: &Config, path: &Path) -> anyhow::Result<Prepared> {
    let bytes = std::fs::read(path)?;
    let cohort: Cohort = serde_json::from_slice(&bytes)?;
    validate_cohort(&cohort)?;
    ensure!(
        cfg.seed == 42
            && cfg.train.batch_size == 32
            && cfg.net.dropout == 0.1
            && cfg.train.learning_rate == 0.0001
            && cfg.train.warmup_steps == 500
            && cfg.train.gradient_clip_norm == Some(1.0)
            && cfg.train.diagnostic_schedule_steps == Some(7000)
            && cfg
                .detector
                .as_ref()
                .is_some_and(|d| d.class_weights == [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5]),
        "frozen native objective or optimizer contract changed"
    );
    ensure!(
        hash(Path::new(&cohort.source_config))? == cohort.config_sha256,
        "frozen source config changed"
    );
    let original = crate::config::load(Path::new(&cohort.source_config))?;
    ensure!(
        contract(cfg) == contract(&original),
        "frozen training contract changed"
    );
    let receipt = cohort
        .separation_receipt
        .as_ref()
        .context("frozen cohort requires a reviewed held-separation receipt")?;
    ensure!(
        hash(Path::new(&receipt.path))? == receipt.sha256,
        "separation receipt changed"
    );
    let approved: Value = serde_json::from_slice(&std::fs::read(&receipt.path)?)?;
    validate_receipt(&cohort, &approved)?;
    let configured = &cfg
        .detector
        .as_ref()
        .context("missing detector settings")?
        .silver;
    let mut names = BTreeSet::new();
    let mut texts = BTreeSet::new();
    let mut selected = Vec::new();
    let mut documents = Vec::new();
    let mut sources = Vec::new();
    let mut loaded_sources = std::collections::BTreeMap::new();
    let mut biography_person = 0;
    let mut biography_singletons = 0;
    for row in &cohort.documents {
        ensure!(names.insert(&row.name), "duplicate frozen document name");
        let source_index = configured
            .iter()
            .position(|p| p == &row.source_path)
            .context("frozen source is absent from the approved training config")?;
        ensure!(
            row.source_line > 0 && hash(Path::new(&row.source_path))? == row.source_sha256,
            "frozen source changed"
        );
        let source = std::fs::read_to_string(&row.source_path)?;
        let raw = source
            .lines()
            .nth(row.source_line - 1)
            .context("frozen source row missing")?;
        let original = validate_row(row, raw)?;
        ensure!(
            texts.insert(original.text.clone()),
            "frozen text is duplicated"
        );
        let gold = original_gold(&original);
        if !loaded_sources.contains_key(&row.source_path) {
            let (native, counts) = detector::load_silver(
                std::slice::from_ref(&row.source_path),
                &cfg.features.to_tessera(),
                900,
            )?;
            sources.push(json!({"path": row.source_path, "sha256": row.source_sha256,
                               "loaded_pieces": counts.pieces}));
            loaded_sources.insert(row.source_path.clone(), native);
        }
        let native = loaded_sources
            .get(&row.source_path)
            .context("missing native source")?;
        let doc = native
            .iter()
            .find(|d| {
                let mut actual = d.gold.clone();
                actual.sort_by_key(|g| (g.start, g.end, g.kind));
                d.text == original.text && actual == gold
            })
            .context("frozen whole row is clipped, split or native gold changed")?;
        ensure!(
            doc.enc.labels.len() == row.content_tokens && (1..=900).contains(&row.content_tokens),
            "frozen native token count changed or exceeds 900"
        );
        let mut encoded = detector::encode_document(&doc.text, &gold, &cfg.features.to_tessera())?;
        encoded.country = "US".into();
        ensure!(encoded == doc.enc, "frozen whole native features changed");
        selected.push(json!({
            "name": row.name, "role": row.role,
            "source_index": source_index, "source_path": row.source_path,
            "source_line": row.source_line, "source_sha256": row.source_sha256,
            "row_sha256": row.row_sha256, "model_gold_sha256": row.model_gold_sha256,
            "source_provenance": original.provenance, "original_source_row": true,
            "derived_piece": false, "text_sha256": row.text_sha256,
            "content_tokens": row.content_tokens, "form": form(doc, support(doc)?)?,
            "gold": gold.iter().map(|g| json!({"kind": detector::KINDS[g.kind].as_str(),
                                                "start": g.start, "end": g.end})).collect::<Vec<_>>(),
        }));
        let person: Vec<_> = gold.iter().filter(|g| g.kind == 0).collect();
        let singleton = person
            .iter()
            .filter(|g| {
                doc.text[g.start as usize..g.end as usize]
                    .split_whitespace()
                    .count()
                    == 1
            })
            .count();
        match row.role.as_str() {
            "biography" => {
                biography_person += person.len();
                biography_singletons += singleton;
            }
            "short-singleton" => verify_singleton_control(singleton)?,
            "short-multiword" => ensure!(
                !person.is_empty() && singleton == 0,
                "short multiword control changed"
            ),
            "model-negative" => ensure!(
                gold.is_empty(),
                "model-negative control acquired native gold"
            ),
            _ => anyhow::bail!("unsupported frozen role"),
        }
        documents.push(doc.clone());
    }
    ensure!(
        biography_person == 38 && biography_singletons == 27,
        "frozen biography PERSON support changed"
    );
    ensure!(
        hash(path)? == crate::export::sha256_hex(&bytes),
        "cohort changed during preparation"
    );
    Ok(Prepared {
        documents,
        manifest: json!({
            "scope": SCOPE, "frozen_cohort_sha256": crate::export::sha256_hex(&bytes),
            "expected_initial_sha256": INITIAL_SHA, "selected_documents": 15,
            "max_content_tokens": 900, "biography_person_gold": 38, "biography_singleton_gold": 27, "original_text_and_gold_preserved": true,
            "text_scope": "complete existing silver row; no new clipping or windows",
            "separation_receipt_sha256": receipt.sha256,
            "sources": sources, "documents": selected,
        }),
    })
}

fn content_identity(cohort: &Cohort) -> anyhow::Result<String> {
    let rows: Vec<_> = cohort
        .documents
        .iter()
        .map(|r| {
            json!([
                r.name,
                r.role,
                r.source_path,
                r.source_line,
                r.source_sha256,
                r.row_sha256,
                r.text_sha256,
                r.model_gold_sha256,
                r.content_tokens
            ])
        })
        .collect();
    Ok(crate::export::sha256_hex(&serde_json::to_vec(&json!([
        cohort.scope,
        cohort.source_config,
        cohort.config_sha256,
        cohort.expected_initial_sha256,
        rows,
    ]))?))
}

/// Keep the wider inspector policy tied to an exact reviewed cohort and its native originals.
pub(crate) fn verify_frozen_selection(
    cfg: &Config,
    marker: &Value,
    selection: &Value,
) -> anyhow::Result<()> {
    ensure!(
        marker["scope"] == "fixed training-only memorization diagnostic; never export"
            && marker["release_quality_claim"] == false
            && marker["frozen_cohort_scope"] == SCOPE
            && marker["max_steps"] == 2000
            && marker["observation_steps"] == json!(super::exposure::OBSERVATIONS)
            && marker["logit_penalty"] == 0.0
            && marker["expected_initial_sha256"] == INITIAL_SHA
            && marker["actual_initial_sha256"] == INITIAL_SHA,
        "frozen PERSON diagnostic marker differs"
    );
    let path = Path::new(
        marker["frozen_cohort_path"]
            .as_str()
            .context("missing frozen cohort path")?,
    );
    ensure!(
        marker["frozen_cohort_sha256"] == hash(path)?,
        "frozen cohort hash changed"
    );
    let mut expected = prepare_frozen(cfg, path)?.manifest;
    if selection.get("forward_operator").is_some() {
        ensure!(
            selection["forward_operator"] == crate::diagnostic_operator::NAME
                && marker["forward_operator"]["name"] == crate::diagnostic_operator::NAME,
            "frozen selection operator differs"
        );
        expected["forward_operator"] = json!(crate::diagnostic_operator::NAME);
    }
    ensure!(expected == *selection, "frozen selection changed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort() -> Cohort {
        Cohort {
            scope: SCOPE.into(),
            source_config: "original.toml".into(),
            config_sha256: "config".into(),
            expected_initial_sha256: INITIAL_SHA.into(),
            documents: ORIGINALS
                .iter()
                .enumerate()
                .map(|(i, (path, line, role))| Row {
                    name: format!("original-{i}"),
                    role: role.to_string(),
                    source_path: path.to_string(),
                    source_line: *line,
                    source_sha256: "source".into(),
                    row_sha256: "row".into(),
                    text_sha256: "text".into(),
                    model_gold_sha256: "gold".into(),
                    content_tokens: 2,
                })
                .collect(),
            separation_receipt: None,
        }
    }

    #[test]
    fn exact_original_coordinates_roles_count_and_initial_identity_are_required() {
        let mut c = cohort();
        validate_cohort(&c).unwrap();
        c.documents[0].source_line += 1;
        assert!(validate_cohort(&c).is_err());
        c = cohort();
        c.documents[0].source_path = "other.jsonl".into();
        assert!(validate_cohort(&c).is_err());
        c = cohort();
        c.documents[0].role = "model-negative".into();
        assert!(validate_cohort(&c).is_err());
        c = cohort();
        c.documents.pop();
        assert!(validate_cohort(&c).is_err());
        c = cohort();
        c.expected_initial_sha256 = "different".into();
        assert!(validate_cohort(&c).is_err());
    }

    #[test]
    fn row_text_and_native_gold_identity_each_fail_closed() {
        let raw = r#"{"text":"Ada Lane","country":"US","entities":[{"kind":"person","start":0,"end":8}]}"#;
        let mut row = cohort().documents.remove(0);
        row.row_sha256 = crate::export::sha256_hex(raw.as_bytes());
        row.text_sha256 = crate::export::sha256_hex(b"Ada Lane");
        row.model_gold_sha256 = crate::export::sha256_hex(br#"[["person",0,8]]"#);
        assert!(validate_row(&row, raw).is_ok());
        assert!(validate_row(&row, &raw.replace("Ada", "Ava")).is_err());
        row.text_sha256 = "changed".into();
        assert!(validate_row(&row, raw).is_err());
        row.text_sha256 = crate::export::sha256_hex(b"Ada Lane");
        row.model_gold_sha256 = "changed".into();
        assert!(validate_row(&row, raw).is_err());
    }

    #[test]
    fn reviewed_receipt_is_bound_to_the_entire_cohort_identity() {
        let mut c = cohort();
        let mut review = json!({"scope": SCOPE, "approved_for_training_only_person15": true,
            "no_held_expansion": true, "whole_native_text_and_gold_verified": true,
            "cohort_sha256_without_receipt": content_identity(&c).unwrap()});
        validate_receipt(&c, &review).unwrap();
        c.documents[0].source_sha256 = "changed".into();
        assert!(validate_receipt(&c, &review).is_err());
        c = cohort();
        review["no_held_expansion"] = json!(false);
        assert!(validate_receipt(&c, &review).is_err());
    }

    #[test]
    fn cohort_paths_are_hashed_and_file_changes_are_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source");
        std::fs::write(&path, "unchanged").unwrap();
        let original = hash(&path).unwrap();
        std::fs::write(&path, "changed").unwrap();
        assert_ne!(hash(&path).unwrap(), original);
    }

    #[test]
    fn one_singleton_control_can_retain_an_additional_multiword_person() {
        let text = "Ada Lane spoke to Kent";
        let gold = [
            detector::KindSpan {
                kind: 0,
                start: 0,
                end: 8,
            },
            detector::KindSpan {
                kind: 0,
                start: 18,
                end: 22,
            },
        ];
        let singleton = gold
            .iter()
            .filter(|g| {
                text[g.start as usize..g.end as usize]
                    .split_whitespace()
                    .count()
                    == 1
            })
            .count();
        verify_singleton_control(singleton).unwrap();
        assert_eq!(gold.len(), 2);
        assert_eq!(singleton, 1);
        assert!(verify_singleton_control(0).is_err());
        assert!(verify_singleton_control(2).is_err());
    }
}
