//! Source identities for the optional planned biography exposure audit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::Config;
use crate::detector::DetectorDoc;

const ORIGINALS: [(&str, usize); 5] = [
    ("data/interim/silver/us-person-long-reviewed-v2.jsonl", 3),
    ("data/interim/silver/us-person-long-reviewed-v2.jsonl", 4),
    ("data/interim/silver/us-person-long-reviewed-v3.jsonl", 4),
    ("data/interim/silver/us-person-long-reviewed-v3.jsonl", 1),
    (
        "data/interim/silver/us-person-keltner-policy-reviewed-v1.jsonl",
        1,
    ),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub scope: String,
    pub config_sha256: String,
    pub input_sha256: BTreeMap<String, String>,
    pub reference_audit: Receipt,
    pub documents: Vec<Target>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    pub path: String,
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Target {
    pub name: String,
    pub source_path: String,
    pub source_line: usize,
    pub source_sha256: String,
    pub row_sha256: String,
    pub text_sha256: String,
    pub model_gold_sha256: String,
    pub content_tokens: usize,
}

pub(super) struct Resolved {
    pub identity: Target,
    pub original_index: usize,
    pub tokens: Vec<Value>,
    pub encoding_sha256: String,
    pub label_token_counts: [usize; 7],
}

pub(super) fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&std::fs::read(path)?))
}

pub(super) fn validate_manifest(manifest: &Manifest) -> anyhow::Result<()> {
    ensure!(
        manifest.scope == "planned-fullmix-biography-exposure-v1",
        "unknown exposure scope"
    );
    let mut identities = BTreeSet::new();
    let mut names = BTreeSet::new();
    for target in &manifest.documents {
        ensure!(
            !target.name.is_empty() && names.insert(target.name.as_str()),
            "duplicate or empty target name"
        );
        ensure!(
            identities.insert((target.source_path.as_str(), target.source_line)),
            "duplicate target source row"
        );
        ensure!(
            target.content_tokens > 0 && target.content_tokens <= 900,
            "target must be a whole nonempty source row"
        );
    }
    ensure!(
        identities == ORIGINALS.into_iter().collect(),
        "exposure requires exactly the five frozen biographies"
    );
    Ok(())
}

fn validate_row(target: &Target, raw: &str) -> anyhow::Result<Value> {
    ensure!(
        crate::export::sha256_hex(raw.as_bytes()) == target.row_sha256,
        "target source row changed: {}",
        target.name
    );
    let row: Value = serde_json::from_str(raw)?;
    ensure!(row["country"] == "US", "target is not a US source row");
    let text = row["text"].as_str().context("target row has no text")?;
    ensure!(
        crate::export::sha256_hex(text.as_bytes()) == target.text_sha256,
        "target text changed"
    );
    Ok(row)
}

fn validate_native(
    target: &Target,
    row: &Value,
    doc: &DetectorDoc,
    features: &tessera::internal::FeatureConfig,
) -> anyhow::Result<Vec<Value>> {
    let text = row["text"].as_str().context("target source text absent")?;
    ensure!(doc.text == text, "native target text differs from source");
    ensure!(
        doc.enc.labels.len() == target.content_tokens,
        "target native token count changed"
    );
    let mut gold = doc.gold.clone();
    gold.sort_by_key(|span| (span.start, span.end, span.kind));
    let identity: Vec<_> = gold
        .iter()
        .map(|span| {
            json!([
                crate::detector::KINDS[span.kind].as_str(),
                span.start,
                span.end
            ])
        })
        .collect();
    ensure!(
        crate::export::sha256_hex(&serde_json::to_vec(&identity)?) == target.model_gold_sha256,
        "target native model gold changed"
    );
    let mut encoded = crate::detector::encode_document(text, &gold, features)
        .map_err(|error| anyhow::anyhow!("target encoding: {error}"))?;
    encoded.country = "US".to_owned();
    ensure!(
        encoded == doc.enc,
        "source-backed native target encoding differs from training"
    );
    let source_entities = row["entities"]
        .as_array()
        .context("target entities absent")?;
    let mut source_gold: Vec<_> = source_entities
        .iter()
        .filter(|span| matches!(span["kind"].as_str(), Some("person" | "org" | "address")))
        .map(|span| json!([span["kind"], span["start"], span["end"]]))
        .collect();
    source_gold.sort_by_key(|span| {
        (
            span[1].as_u64(),
            span[2].as_u64(),
            span[0].as_str().map(str::to_owned),
        )
    });
    ensure!(
        source_gold == identity,
        "native target dropped or changed original gold"
    );
    let mut tokens = Vec::new();
    let mut covered = BTreeSet::new();
    for (gold_index, span) in gold.iter().enumerate() {
        let first = encoded
            .token_spans
            .iter()
            .position(|token| token.0 == span.start)
            .context("target gold start not encoded")?;
        let last = encoded
            .token_spans
            .iter()
            .position(|token| token.1 == span.end)
            .context("target gold end not encoded")?;
        ensure!(first <= last, "target gold token range reversed");
        for token_index in first..=last {
            let label = (1 + 2 * span.kind + usize::from(token_index != first)) as u8;
            ensure!(
                encoded.labels[token_index] == label && covered.insert(token_index),
                "target native gold BIO mismatch or overlap"
            );
            tokens.push(json!({"gold_index": gold_index, "kind": crate::detector::KINDS[span.kind].as_str(), "gold_byte_range": [span.start, span.end], "token_index": token_index, "token_byte_range": encoded.token_spans[token_index], "bio_label": label}));
        }
    }
    ensure!(
        covered.len() == encoded.labels.iter().filter(|&&label| label != 0).count(),
        "target nonzero native labels were not fully indexed"
    );
    Ok(tokens)
}

pub(super) fn resolve(
    manifest: Manifest,
    cfg: &Config,
    docs: &[DetectorDoc],
    source_pieces: &[usize],
    synthetic_count: usize,
    pins: &mut BTreeMap<String, String>,
) -> anyhow::Result<Vec<Resolved>> {
    validate_manifest(&manifest)?;
    let detector = cfg
        .detector
        .as_ref()
        .context("missing detector configuration")?;
    ensure!(
        source_pieces.len() == detector.silver.len()
            && source_pieces.iter().sum::<usize>() == docs.len(),
        "source piece boundaries differ from native silver"
    );
    let mut resolved = Vec::new();
    let mut selected_indices = BTreeSet::new();
    for target in manifest.documents {
        let source_index = detector
            .silver
            .iter()
            .position(|path| path == &target.source_path)
            .context("target source is absent from training")?;
        ensure!(
            detector
                .silver
                .iter()
                .filter(|path| *path == &target.source_path)
                .count()
                == 1,
            "target source path is duplicated in configuration"
        );
        let source = std::fs::read_to_string(&target.source_path)?;
        ensure!(
            crate::export::sha256_hex(source.as_bytes()) == target.source_sha256,
            "target source file changed"
        );
        pins.insert(target.source_path.clone(), target.source_sha256.clone());
        let raw = source
            .lines()
            .nth(target.source_line - 1)
            .context("target source line is absent")?;
        let row = validate_row(&target, raw)?;
        let text = row["text"].as_str().context("target text absent")?;
        let offset: usize = source_pieces[..source_index].iter().sum();
        let matches: Vec<_> = docs[offset..offset + source_pieces[source_index]]
            .iter()
            .enumerate()
            .filter(|(_, doc)| doc.text == text)
            .collect();
        ensure!(
            matches.len() == 1,
            "target must resolve to one whole native source piece"
        );
        let (local_index, doc) = matches[0];
        let index = synthetic_count + offset + local_index;
        ensure!(
            selected_indices.insert(index),
            "targets resolve to the same encoded training item"
        );
        let tokens = validate_native(&target, &row, doc, &cfg.features.to_tessera())?;
        let encoding_sha256 = crate::export::sha256_hex(&serde_json::to_vec(&json!([
            doc.enc.token_spans,
            doc.enc.ngram_ids,
            doc.enc.script,
            doc.enc.shape,
            doc.enc.flags,
            doc.enc.labels,
            doc.enc.country,
        ]))?);
        let mut label_token_counts = [0usize; 7];
        for &label in &doc.enc.labels {
            *label_token_counts
                .get_mut(usize::from(label))
                .context("invalid target native label")? += 1;
        }
        resolved.push(Resolved {
            identity: target,
            original_index: index,
            tokens,
            encoding_sha256,
            label_token_counts,
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            scope: "planned-fullmix-biography-exposure-v1".into(),
            config_sha256: "config".into(),
            input_sha256: BTreeMap::new(),
            reference_audit: Receipt {
                path: "audit".into(),
                sha256: "audit".into(),
            },
            documents: ORIGINALS
                .iter()
                .enumerate()
                .map(|(index, (path, line))| Target {
                    name: index.to_string(),
                    source_path: (*path).into(),
                    source_line: *line,
                    source_sha256: "source".into(),
                    row_sha256: "row".into(),
                    text_sha256: "text".into(),
                    model_gold_sha256: "gold".into(),
                    content_tokens: 10,
                })
                .collect(),
        }
    }

    #[test]
    fn targets_require_complete_unique_frozen_identities() {
        validate_manifest(&manifest()).unwrap();
        let mut missing = manifest();
        missing.documents.pop();
        assert!(validate_manifest(&missing).is_err());
        let mut duplicate = manifest();
        duplicate.documents[1].source_line = duplicate.documents[0].source_line;
        assert!(validate_manifest(&duplicate).is_err());
        let mut names = manifest();
        names.documents[1].name = names.documents[0].name.clone();
        assert!(validate_manifest(&names).is_err());
        let mut wrong = manifest();
        wrong.documents[0].source_line = 99;
        assert!(validate_manifest(&wrong).is_err());
    }

    #[test]
    fn ordinal_alone_cannot_identify_a_target() {
        let raw = r#"{"country":"US","text":"Ada","entities":[]}"#;
        let mut target = manifest().documents.remove(0);
        target.row_sha256 = crate::export::sha256_hex(raw.as_bytes());
        target.text_sha256 = crate::export::sha256_hex(b"Ada");
        validate_row(&target, raw).unwrap();
        assert!(validate_row(&target, &raw.replace("Ada", "Eve")).is_err());
        target.text_sha256 = "changed".into();
        assert!(validate_row(&target, raw).is_err());
    }
    #[test]
    fn native_gold_and_feature_identity_must_match_original_source() {
        let features = tessera::internal::FeatureConfig::default();
        let gold = vec![crate::detector::KindSpan {
            kind: 0,
            start: 0,
            end: 3,
        }];
        let mut encoded = crate::detector::encode_document("Ada works", &gold, &features).unwrap();
        encoded.country = "US".into();
        let mut doc = DetectorDoc {
            text: "Ada works".into(),
            enc: encoded,
            gold,
            breaks: vec![false; 2],
        };
        let row = json!({"country":"US","text":"Ada works","entities":[{"kind":"person","start":0,"end":3}]});
        let mut target = manifest().documents.remove(0);
        target.content_tokens = 2;
        target.model_gold_sha256 = crate::export::sha256_hex(br#"[["person",0,3]]"#);
        let tokens = validate_native(&target, &row, &doc, &features).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0]["bio_label"], 1);
        let mut missing = row.clone();
        missing["entities"] = json!([]);
        assert!(validate_native(&target, &missing, &doc, &features).is_err());
        target.model_gold_sha256 = "changed".into();
        assert!(validate_native(&target, &row, &doc, &features).is_err());
        target.model_gold_sha256 = crate::export::sha256_hex(br#"[["person",0,3]]"#);
        doc.enc.labels[0] = 0;
        assert!(validate_native(&target, &row, &doc, &features).is_err());
    }
}
