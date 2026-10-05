//! Retained canonical token counts over all documents and declared whole-document draws.

use super::{LoadedMixed, sampling, supervision, token_counts, weights};
use crate::reviewed_data::Cohort;
use anyhow::{Context, ensure};
use serde_json::{Value, json};

/// Measured canonical encodings and exact fixed draws; counts are not gradients.
pub(in crate::reviewed_fit) fn accounting(
    loaded: &LoadedMixed,
    native: &Cohort,
    plan: &sampling::Plan,
    selected_weights: &[f32],
    explicit_choice: bool,
    exclusions: Option<&supervision::Resolved>,
) -> anyhow::Result<Value> {
    weights(selected_weights)?;
    let mut docs = Vec::new();
    let mut unique = [0usize; 7];
    let mut drawn = [0usize; 7];
    let mut active_unique = [0usize; 7];
    let mut active_drawn = [0usize; 7];
    let mut unique_rule = 0usize;
    let mut drawn_rule = 0usize;
    let mut unique_ambiguity = 0usize;
    let mut drawn_ambiguity = 0usize;
    let counts = plan.identity["planned_document_draw_counts"]
        .as_array()
        .context("draw accounting missing")?;
    ensure!(
        counts.len() == loaded.len(),
        "draw accounting row count differs"
    );
    for (index, count) in counts.iter().enumerate() {
        let draws = usize::try_from(count.as_u64().context("draw count missing")?)?;
        let doc = loaded.doc(native, index)?;
        let reviewed = exclusions
            .map(|resolved| resolved.row(loaded.native_owner(index)?, doc.enc.token_spans.len()))
            .transpose()?;
        let row = token_counts(doc, selected_weights, reviewed.as_deref())?;
        if reviewed.is_some() {
            let excluded = usize::try_from(
                row["excluded_reviewed_ambiguity_o_tokens"]
                    .as_u64()
                    .context("reviewed ambiguity count missing")?,
            )?;
            unique_ambiguity = unique_ambiguity
                .checked_add(excluded)
                .context("reviewed ambiguity token overflow")?;
            drawn_ambiguity = drawn_ambiguity
                .checked_add(
                    excluded
                        .checked_mul(draws)
                        .context("drawn reviewed ambiguity overflow")?,
                )
                .context("drawn reviewed ambiguity overflow")?;
        }
        for kind in 0..7 {
            let all = usize::try_from(
                row["all_label_counts"][kind]
                    .as_u64()
                    .context("token count missing")?,
            )?;
            let active = usize::try_from(
                row["active_label_counts"][kind]
                    .as_u64()
                    .context("active count missing")?,
            )?;
            unique[kind] = unique[kind]
                .checked_add(all)
                .context("unique token overflow")?;
            active_unique[kind] = active_unique[kind]
                .checked_add(active)
                .context("active unique overflow")?;
            drawn[kind] = drawn[kind]
                .checked_add(all.checked_mul(draws).context("drawn token overflow")?)
                .context("drawn token overflow")?;
            active_drawn[kind] = active_drawn[kind]
                .checked_add(active.checked_mul(draws).context("active drawn overflow")?)
                .context("active drawn overflow")?;
        }
        let ruled = usize::try_from(
            row["excluded_rule_o_tokens"]
                .as_u64()
                .context("rule count missing")?,
        )?;
        unique_rule = unique_rule
            .checked_add(ruled)
            .context("rule token overflow")?;
        drawn_rule = drawn_rule
            .checked_add(ruled.checked_mul(draws).context("drawn rule overflow")?)
            .context("drawn rule overflow")?;
        docs.push(json!({"index":index,"owner":loaded.indices[index],"name":loaded.record["train_inputs"][index]["name"],"draws":draws,"counts":row}));
    }
    let mass = |counts: &[usize; 7]| -> f64 {
        counts
            .iter()
            .zip(selected_weights)
            .map(|(n, w)| *n as f64 * f64::from(*w))
            .sum()
    };
    let mut record = json!({"scope":"reviewed-mixed-canonical-token-accounting-v1","weights":selected_weights,"weights_are_execution_choice":explicit_choice,
        "documents":docs,"unique_all_label_counts":unique,"unique_active_label_counts":active_unique,"unique_rule_o":unique_rule,"unique_active_weighted_mass":mass(&active_unique),
        "drawn_all_label_counts":drawn,"drawn_active_label_counts":active_drawn,"drawn_rule_o":drawn_rule,"drawn_active_weighted_mass":mass(&active_drawn),
        "batches":plan.identity["token_batches"],"padding_is_excluded_from_counts":true,"gold_selected_loss_masks":false,
        "weighted_mass_is_not_gradient_share":true,"normalization":"each batch separately, denominator clamp_min(1.0)","new_independent_entities_or_source_families":false});
    if exclusions.is_some() {
        record["unique_reviewed_ambiguity_o"] = json!(unique_ambiguity);
        record["drawn_reviewed_ambiguity_o"] = json!(drawn_ambiguity);
    }
    Ok(record)
}
