//! Preregistered complete-document draw plans with explicit source/layout identities.

use crate::reviewed_data::{Cohort, Receipt};
use anyhow::{Context, ensure};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Declared metadata in the exact native TRAIN order; absent source family stays absent.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Member {
    pub(super) name: String,
    pub(super) source_family_id: Option<String>,
    pub(super) layout: String,
}

/// Complete-pass ordering preserves exposure ratios; declared batches permit explicit weights.
#[derive(Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Sampling {
    CompletePassStratified,
    DeclaredBatches { recipe: Receipt },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declared {
    scope: String,
    canonical_preflight_sha256: String,
    train_names: Vec<String>,
    batches: Vec<Vec<usize>>,
}

/// Every optimizer batch is planned before model initialization.
pub(super) struct Plan {
    pub(super) batches: Vec<Vec<usize>>,
    pub(super) identity: Value,
    pub(super) receipt: Option<Receipt>,
}

/// Resource and reproducibility inputs from the bound recipe.
pub(super) struct Budget {
    pub(super) seed: u64,
    pub(super) updates: usize,
    pub(super) size: usize,
}

pub(super) fn validate_batches(
    batches: &[Vec<usize>],
    documents: usize,
    updates: usize,
    size: usize,
) -> anyhow::Result<Vec<usize>> {
    ensure!(
        batches.len() == updates,
        "draw recipe differs from fixed optimizer count"
    );
    let mut counts = vec![0usize; documents];
    for batch in batches {
        ensure!(
            !batch.is_empty() && batch.len() <= size,
            "draw batch empty or exceeds configured size"
        );
        let mut seen = BTreeSet::new();
        for &index in batch {
            ensure!(
                index < documents && seen.insert(index),
                "draw has missing or repeated within-batch TRAIN index"
            );
            counts[index] = counts[index]
                .checked_add(1)
                .context("draw count overflow")?;
        }
    }
    ensure!(
        counts.iter().all(|&count| count > 0),
        "fixed draw plan does not expose every declared TRAIN row"
    );
    Ok(counts)
}

/// Resolve and validate all fixed batches from authoritative native membership.
pub(super) fn prepare(
    sampling: &Sampling,
    members: &[Member],
    train: &Cohort,
    native: &Value,
    canonical_sha256: &str,
    budget: Budget,
) -> anyhow::Result<Plan> {
    let Budget {
        seed,
        updates,
        size,
    } = budget;
    ensure!(
        members.len() == train.docs.len() && train.names.len() == members.len(),
        "sampling membership incomplete"
    );
    let rows = native["train"]
        .as_array()
        .context("native sampling identity missing")?;
    ensure!(rows.len() == members.len(), "sampling/native counts differ");
    let mut groups = BTreeMap::<(Option<String>, String, bool), Vec<usize>>::new();
    let mut identities = Vec::new();
    for (index, member) in members.iter().enumerate() {
        ensure!(
            member.name == train.names[index]
                && !member.layout.trim().is_empty()
                && member
                    .source_family_id
                    .as_ref()
                    .is_none_or(|s| !s.trim().is_empty())
                && rows[index]["origin"]["source_family_id"]
                    == serde_json::to_value(&member.source_family_id)?,
            "sampling member order/family differs from native origin"
        );
        let org_present = train.docs[index].gold.iter().any(|s| s.kind == 1);
        groups
            .entry((
                member.source_family_id.clone(),
                member.layout.clone(),
                org_present,
            ))
            .or_default()
            .push(index);
        identities.push(json!({"index":index,"name":member.name,"native":rows[index],
            "source_family_id":member.source_family_id,"declared_layout":member.layout,"org_present":org_present}));
    }
    let mut receipt = None;
    let batches = match sampling {
        Sampling::DeclaredBatches { recipe } => {
            let declared: Declared = serde_json::from_slice(&recipe.bytes()?)?;
            ensure!(
                declared.scope == "reviewed-native-explicit-draws-v1"
                    && declared.canonical_preflight_sha256 == canonical_sha256
                    && declared.train_names == train.names,
                "explicit draws bind another canonical dataset"
            );
            receipt = Some(recipe.clone());
            declared.batches
        }
        Sampling::CompletePassStratified => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut batches = Vec::new();
            while batches.len() < updates {
                let mut pools: Vec<_> = groups.values().cloned().collect();
                for pool in &mut pools {
                    pool.shuffle(&mut rng);
                }
                pools.shuffle(&mut rng);
                let mut order = Vec::with_capacity(train.docs.len());
                while pools.iter().any(|pool| !pool.is_empty()) {
                    for pool in &mut pools {
                        if let Some(index) = pool.pop() {
                            order.push(index);
                        }
                    }
                }
                for batch in order.chunks(size) {
                    if batches.len() == updates {
                        break;
                    }
                    batches.push(batch.to_vec());
                }
            }
            batches
        }
    };
    let counts = validate_batches(&batches, train.docs.len(), updates, size)?;
    let mut strata = Vec::new();
    for ((family, layout, org_present), indices) in groups {
        strata.push(json!({"source_family_id":family,"declared_layout":layout,"org_present":org_present,
            "members":indices,"planned_document_draws":indices.iter().map(|&i|counts[i]).sum::<usize>()}));
    }
    let identity = json!({"sampling":sampling,"members":identities,"strata":strata,
        "batch_indices_sha256":crate::reviewed_data::digest(&serde_json::to_vec(&batches)?),
        "planned_document_draw_counts":counts,"fixed_updates":updates,
        "semantics":"complete-pass ordering preserves source proportions, not balanced epoch exposure; explicit declared batches may repeat across batches; no gradient-share or heldout-gain claim"});
    Ok(Plan {
        batches,
        identity,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_draws_reject_missing_repeated_dev_and_incomplete_rows() {
        assert!(validate_batches(&[vec![0, 0]], 2, 1, 2).is_err());
        assert!(validate_batches(&[vec![0, 2]], 2, 1, 2).is_err());
        assert!(validate_batches(&[vec![0]], 2, 1, 2).is_err());
        assert_eq!(
            validate_batches(&[vec![0, 1], vec![1]], 2, 2, 2).unwrap(),
            [1, 2]
        );
    }
}
