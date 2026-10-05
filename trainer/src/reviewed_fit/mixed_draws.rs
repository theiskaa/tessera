//! Complete mixed document passes and actual planned batch token accounting.

use super::{Index, LoadedMixed, Settings, objective, sampling, supervision};
use crate::reviewed_data::{Cohort, digest};
use anyhow::{Context, ensure};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Order complete whole-document passes across explicit origin/parent/layout strata.
pub(super) fn complete_passes(
    groups: &BTreeMap<String, Vec<usize>>,
    n: usize,
    budget: sampling::Budget,
) -> anyhow::Result<Vec<Vec<usize>>> {
    ensure!(
        n > 0 && budget.size > 0 && budget.updates > 0,
        "empty mixed draw budget"
    );
    let per_pass = n.div_ceil(budget.size);
    ensure!(
        budget.updates.is_multiple_of(per_pass),
        "mixed budget must finish complete passes"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(budget.seed);
    let mut batches = Vec::new();
    for _ in 0..budget.updates / per_pass {
        let mut queues: Vec<_> = groups.values().cloned().collect();
        for queue in &mut queues {
            queue.shuffle(&mut rng);
        }
        let mut order = Vec::new();
        while order.len() < n {
            let mut keys: Vec<_> = (0..queues.len()).collect();
            keys.shuffle(&mut rng);
            let before = order.len();
            for key in keys {
                if let Some(row) = queues[key].pop() {
                    order.push(row);
                }
            }
            ensure!(order.len() > before, "mixed strata do not cover membership");
        }
        ensure!(
            order.len() == n && order.iter().copied().collect::<BTreeSet<_>>() == (0..n).collect(),
            "mixed strata repeat or omit rows"
        );
        batches.extend(order.chunks(budget.size).map(|rows| rows.to_vec()));
    }
    sampling::validate_batches(&batches, n, budget.updates, budget.size)?;
    Ok(batches)
}

/// Loss inputs of planned batch accounting: reference class weights and reviewed exclusions.
pub(in crate::reviewed_fit) struct Loss<'a> {
    pub(in crate::reviewed_fit) reference_weights: &'a [f32],
    pub(in crate::reviewed_fit) exclusions: Option<&'a supervision::Resolved>,
}

/// Every full mixed pass is completed; document exposure equality is not token/loss balance.
pub(in crate::reviewed_fit) fn plan(
    loaded: &LoadedMixed,
    native: &Cohort,
    settings: &Settings,
    members: &[sampling::Member],
    native_identity: &Value,
    loss: Loss<'_>,
    budget: sampling::Budget,
) -> anyhow::Result<sampling::Plan> {
    ensure!(
        members.len() == native.docs.len() && members.len() == native.names.len(),
        "native sampling membership incomplete"
    );
    let rows = native_identity["train"]
        .as_array()
        .context("native sampling rows missing")?;
    ensure!(
        rows.len() == members.len(),
        "native sampling identity count differs"
    );
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    let mut strata = Vec::new();
    for (index, owner) in loaded.indices.iter().enumerate() {
        let (parent, layout, origin) = match owner {
            Index::Native { row } => (
                *row,
                members
                    .get(*row)
                    .context("native member missing")?
                    .layout
                    .as_str(),
                "native",
            ),
            Index::AuthoredTrainOnly { packet, row } => {
                let identity = &loaded.authored[*packet].identities[*row];
                let base = identity["base_variant_group"]["native_name"]
                    .as_str()
                    .context("authored parent missing")?;
                let parent = native
                    .names
                    .iter()
                    .position(|n| n == base)
                    .context("authored parent outside native TRAIN")?;
                (
                    parent,
                    identity["source"]["authored_source"]["format"]
                        .as_str()
                        .context("authored layout missing")?,
                    "authored_train_only",
                )
            }
        };
        let member = &members[parent];
        ensure!(
            member.name == native.names[parent]
                && !layout.trim().is_empty()
                && member
                    .source_family_id
                    .as_ref()
                    .is_none_or(|s| !s.trim().is_empty())
                && rows[parent]["origin"]["source_family_id"]
                    == serde_json::to_value(&member.source_family_id)?,
            "mixed draw member differs from native parent authority"
        );
        let org_present = loaded.doc(native, index)?.gold.iter().any(|s| s.kind == 1);
        let stratum = json!({"origin":origin,"parent_source_family_id":member.source_family_id,"layout":layout,"org_present":org_present});
        let key = serde_json::to_string(&stratum)?;
        groups.entry(key.clone()).or_default().push(index);
        strata.push(json!({"index":index,"parent_native_name":member.name,"stratum":stratum,"stratum_key":key}));
    }
    let n = loaded.len();
    let updates = budget.updates;
    let size = budget.size;
    let batches = complete_passes(&groups, n, budget)?;
    let counts = sampling::validate_batches(&batches, n, updates, size)?;
    let passes = updates / n.div_ceil(size);
    ensure!(
        counts.iter().all(|&c| c == passes),
        "mixed full-pass exposure unequal"
    );
    let weights = settings
        .loss_weights
        .as_deref()
        .unwrap_or(loss.reference_weights);
    let mut token_batches = Vec::new();
    for indices in &batches {
        let items = loaded.items(native, indices)?;
        let ambiguity = loss
            .exclusions
            .map(|resolved| resolved.batch(&loaded.native_owners(indices)?, &items))
            .transpose()?;
        token_batches
            .push(objective::batch_loss_mask(&items, ambiguity.as_deref(), weights)?.report);
    }
    let groups:Vec<_>=groups.iter().map(|(key,indices)|Ok(json!({"stratum":serde_json::from_str::<Value>(key)?,"members":indices,"planned_draws":indices.len().checked_mul(passes).context("stratum draw count overflow")?}))).collect::<anyhow::Result<_>>()?;
    let identity = json!({"sampling":"deterministic complete mixed passes stratified by origin/parent family/layout/ORG presence","canonical_input_sha256":digest(&serde_json::to_vec(&loaded.record)?),
        "indices":loaded.indices,"strata":strata,"groups":groups,"planned_document_draw_counts":counts,"passes":passes,"fixed_updates":updates,
        "batch_indices_sha256":digest(&serde_json::to_vec(&batches)?),"token_batches":token_batches,"accounting_weights":weights,
        "accounting_weights_are_execution_choice":settings.loss_weights.is_some(),
        "semantics":"equal draws per full document; repeated authored TRAIN variants, not new source families or token/gradient balance"});
    Ok(sampling::Plan {
        batches,
        identity,
        receipt: settings.canonical_preflight.clone(),
    })
}
