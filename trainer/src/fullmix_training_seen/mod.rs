//! Exact reviewed training-seen gold, native membership and family diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::detector::KindSpan;
use anyhow::{Context, ensure};
use serde_json::{Value, json};

use super::{Receipt, hash};
use crate::config::Config;
use crate::detector::DetectorDoc;

mod legacy_bindings;

mod hard_bindings;

pub(super) struct Seen {
    pub(super) plan: Value,
    gold: BTreeMap<String, Receipt>,
    names: BTreeMap<String, Vec<String>>,
}

impl Seen {
    pub(super) fn load(plan_pin: &Receipt, review_pin: &Receipt) -> anyhow::Result<Self> {
        let plan: Value = serde_json::from_slice(&plan_pin.bytes()?)?;
        let review: Value = serde_json::from_slice(&review_pin.bytes()?)?;
        ensure!(
            plan["scope"]
                == "complete TRAINING-SEEN candidate diagnostics fixed before any future fit"
                && plan["training_input"] == false
                && plan["model_execution"] == false
                && plan["training_ready"] == false,
            "training-seen plan scope differs"
        );
        ensure!(
            review["passed"] == true
                && review["blockers"].as_array().is_some_and(Vec::is_empty)
                && review["approved_for_training_seen_gold_and_plan_only"] == true
                && review["approved_for_fit_or_source_application"] == false
                && review["reviewed_files"][plan_pin.path.to_string_lossy().as_ref()]
                    == plan_pin.sha256,
            "training-seen independent plan review differs"
        );
        let real_supports = if plan["schema"] == hard_bindings::SCHEMA {
            hard_bindings::supports(&plan, &review)?
        } else {
            legacy_bindings::supports(&plan, &review)?
        };
        let mut gold = BTreeMap::new();
        let mut names = BTreeMap::new();
        for (group, count, supports) in [
            ("authored", 569, [600, 600, 120]),
            ("real", 1799, real_supports),
        ] {
            let pin: Receipt = serde_json::from_value(plan["gold"][group].clone())?;
            let bytes = pin.bytes()?;
            ensure!(
                review["reviewed_files"][pin.path.to_string_lossy().as_ref()] == pin.sha256
                    && plan["documents"][group] == count,
                "training-seen gold binding/count differs"
            );
            let rows: Vec<Value> = std::str::from_utf8(&bytes)?
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            ensure!(rows.len() == count, "training-seen row count differs");
            let mut row_names = Vec::new();
            let mut actual = [0usize; 3];
            for row in &rows {
                ensure!(row["country"] == "US", "training-seen gold is not US");
                let text = row["input"]
                    .as_str()
                    .context("training-seen input absent")?;
                row_names.push(
                    row["name"]
                        .as_str()
                        .context("training-seen name absent")?
                        .to_owned(),
                );
                for e in row["expected"]
                    .as_array()
                    .context("training-seen expected absent")?
                {
                    let kind = ["person", "org", "address"]
                        .iter()
                        .position(|k| e["kind"] == *k);
                    ensure!(
                        kind.is_some() || e["kind"] == "email" || e["kind"] == "phone",
                        "unsupported training-seen declared gold kind"
                    );
                    let start = e["start"].as_u64().context("training-seen start absent")? as usize;
                    let end = e["end"].as_u64().context("training-seen end absent")? as usize;
                    ensure!(
                        text.get(start..end)
                            .is_some_and(|span| !span.is_empty() && e["text"] == span),
                        "training-seen exact UTF8 span differs"
                    );
                    if let Some(kind) = kind {
                        actual[kind] += 1;
                    }
                }
            }
            ensure!(
                row_names.iter().collect::<BTreeSet<_>>().len() == count,
                "training-seen names duplicate"
            );
            for (i, kind) in ["person", "org", "address"].iter().enumerate() {
                ensure!(
                    actual[i] == supports[i]
                        && plan["model_gold_support"][group][kind] == supports[i]
                        && plan["required_exact_f1"][group][kind] == 0.95,
                    "training-seen supported target differs"
                );
            }
            gold.insert(group.to_owned(), pin);
            names.insert(group.to_owned(), row_names);
        }
        for key in ["inputs", "phase_evidence"] {
            for value in plan[key]
                .as_array()
                .context("training-seen lineage absent")?
            {
                let pin: Receipt = serde_json::from_value(value.clone())?;
                pin.bytes()?;
            }
        }
        let builder: Receipt = serde_json::from_value(plan["builder"].clone())?;
        builder.bytes()?;
        Ok(Self { plan, gold, names })
    }

    pub(super) fn input_paths(&self) -> anyhow::Result<Vec<PathBuf>> {
        let mut paths: Vec<_> = self.gold.values().map(|p| p.path.clone()).collect();
        for key in ["inputs", "phase_evidence"] {
            for v in self.plan[key]
                .as_array()
                .context("training-seen lineage absent")?
            {
                paths.push(serde_json::from_value::<Receipt>(v.clone())?.path);
            }
        }
        paths.push(serde_json::from_value::<Receipt>(self.plan["builder"].clone())?.path);
        if self.plan["schema"] == hard_bindings::SCHEMA {
            paths.extend(hard_bindings::input_paths(&self.plan)?);
        }
        Ok(paths)
    }

    pub(super) fn group(&self, path: &Path) -> anyhow::Result<Option<&str>> {
        let actual = hash(path)?;
        Ok(self
            .gold
            .iter()
            .find(|(_, p)| p.sha256 == actual)
            .map(|(group, _)| group.as_str()))
    }

    pub(super) fn docs(&self, group: &str, cfg: &Config) -> anyhow::Result<Vec<DetectorDoc>> {
        crate::detector::load_development_gold(
            &self
                .gold
                .get(group)
                .context("unsupported training-seen group")?
                .path,
            &cfg.features.to_tessera(),
        )
    }

    pub(super) fn validate_membership(
        &self,
        cfg: &Config,
        authored: &[DetectorDoc],
        real: &[DetectorDoc],
        metadata: &[Value],
    ) -> anyhow::Result<()> {
        for (group, members) in [("authored", authored), ("real", real)] {
            let gold = self.docs(group, cfg)?;
            ensure!(
                gold.len() == members.len(),
                "complete training-seen membership count differs: {group}"
            );
            for (g, d) in gold.iter().zip(members) {
                ensure!(
                    g.text == d.text && g.gold == d.gold,
                    "complete training-seen text/gold differs: {group}"
                );
                let model_gold: Vec<_> = g.gold.iter().map(|span| json!({"kind":crate::detector::KINDS[span.kind].as_str(),"start":span.start,"end":span.end})).collect();
                let annotated = crate::detector::annotated_document(
                    &g.text,
                    "US",
                    &serde_json::to_string(&model_gold)?,
                    &cfg.features.to_tessera(),
                )?;
                ensure!(
                    annotated.enc == d.enc,
                    "complete training-seen annotated native encoding differs: {group}"
                );
            }
        }
        let mut families: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut negatives = BTreeSet::new();
        ensure!(
            metadata.len() == authored.len(),
            "training-seen metadata count differs"
        );
        for ((doc, meta), name) in authored.iter().zip(metadata).zip(&self.names["authored"]) {
            ensure!(
                *name
                    == format!(
                        "seen-authored-{}",
                        meta["id"].as_str().context("authored identity absent")?
                    ),
                "training-seen identity order differs"
            );
            families
                .entry(
                    meta["family"]
                        .as_str()
                        .context("authored family absent")?
                        .to_owned(),
                )
                .or_default()
                .push(name.clone());
            if doc.gold.is_empty() {
                negatives.insert(name.clone());
            }
        }
        ensure!(
            serde_json::to_value(families)? == self.plan["families"],
            "training-seen family partition differs"
        );
        let expected: BTreeSet<String> =
            serde_json::from_value(self.plan["negative_document_names"].clone())?;
        ensure!(
            expected.len() == 249 && expected == negatives,
            "training-seen negative controls differ"
        );
        Ok(())
    }

    pub(super) fn metrics(
        &self,
        group: &str,
        gold: &[Vec<KindSpan>],
        raw: &[Vec<KindSpan>],
        filtered: &[Vec<KindSpan>],
    ) -> anyhow::Result<Value> {
        let names = &self.names[group];
        ensure!(
            gold.len() == names.len() && raw.len() == names.len() && filtered.len() == names.len(),
            "training-seen prediction alignment differs"
        );
        let groups = if group == "authored" {
            self.plan["families"]
                .as_object()
                .context("authored families absent")?
                .clone()
        } else {
            serde_json::Map::from_iter([("all_corrected_real".into(), json!(names))])
        };
        let mut family_scores = serde_json::Map::new();
        for (family, members) in groups {
            let members: BTreeSet<String> = serde_json::from_value(members)?;
            let indices: Vec<_> = names
                .iter()
                .enumerate()
                .filter_map(|(i, n)| members.contains(n).then_some(i))
                .collect();
            ensure!(
                indices.len() == members.len(),
                "training-seen family name absent"
            );
            let g: Vec<_> = indices.iter().map(|&i| gold[i].clone()).collect();
            let r: Vec<_> = indices.iter().map(|&i| raw[i].clone()).collect();
            let f: Vec<_> = indices.iter().map(|&i| filtered[i].clone()).collect();
            family_scores.insert(family,json!({"documents":indices.len(),"raw":supported_scores(&g,&r)?,"filtered":supported_scores(&g,&f)?}));
        }
        let text_rows: Vec<Value> = std::str::from_utf8(&self.gold[group].bytes()?)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        let negatives: Vec<_> = names.iter().enumerate().filter(|(i,_)| gold[*i].is_empty()).map(|(i,name)| -> anyhow::Result<Value> {let text = text_rows[i]["input"].as_str().context("training-seen input absent")?;Ok(json!({"name":name,"raw_false_positives":raw[i].iter().map(|s| span_json(s,text)).collect::<anyhow::Result<Vec<_>>>()?,"filtered_false_positives":filtered[i].iter().map(|s| span_json(s,text)).collect::<anyhow::Result<Vec<_>>>()?}))}).collect::<anyhow::Result<_>>()?;
        Ok(
            json!({"role":"TRAINING-SEEN","group":group,"aggregate":{"raw":supported_scores(gold,raw)?,"filtered":supported_scores(gold,filtered)?},"families":family_scores,"negative_documents":negatives,"all_required_supported_kind_targets_passed":crate::detector::score(gold,filtered).per_kind.values().all(|s| s.gold>0 && s.exact.f1>=0.95),"limits":"seen fitting only; no unseen accuracy or RMS-only causal claim"}),
        )
    }
}

fn supported_scores(gold: &[Vec<KindSpan>], pred: &[Vec<KindSpan>]) -> anyhow::Result<Value> {
    let scores = crate::detector::score(gold, pred);
    let mut actual = serde_json::to_value(&scores)?;
    actual["supported_kind_count"] = json!(scores.per_kind.values().filter(|s| s.gold > 0).count());
    if scores.per_kind.values().any(|s| s.gold == 0) {
        actual["macro_exact_f1"] = Value::Null;
    }
    for (kind, score) in &scores.per_kind {
        let k = crate::detector::KINDS
            .iter()
            .position(|k| k.as_str() == *kind)
            .context("score kind absent")?;
        let mut tp = 0usize;
        let mut predicted = 0usize;
        let mut support = 0usize;
        for (g, p) in gold.iter().zip(pred) {
            let mut gs = BTreeMap::new();
            let mut ps = BTreeMap::new();
            for s in g.iter().filter(|s| s.kind == k) {
                *gs.entry((s.start, s.end)).or_insert(0usize) += 1;
                support += 1;
            }
            for s in p.iter().filter(|s| s.kind == k) {
                *ps.entry((s.start, s.end)).or_insert(0usize) += 1;
                predicted += 1;
            }
            tp += ps
                .iter()
                .map(|(span, n)| (*n).min(gs.get(span).copied().unwrap_or(0)))
                .sum::<usize>();
        }
        actual["per_kind"][kind]["exact_counts"] =
            json!({"tp":tp,"fp":predicted-tp,"fn":support-tp,"gold":support,"predicted":predicted});
        if score.gold == 0 {
            for key in ["precision", "recall", "f1"] {
                actual["per_kind"][kind]["exact"][key] = Value::Null;
                actual["per_kind"][kind]["lenient"][key] = Value::Null;
            }
            actual["per_kind"][kind]["support_role"] =
                json!("N/A: zero gold support; false positives still reported");
        }
    }
    Ok(actual)
}

fn span_json(span: &KindSpan, text: &str) -> anyhow::Result<Value> {
    Ok(
        json!({"kind":crate::detector::KINDS[span.kind].as_str(),"start":span.start,"end":span.end,"text":text.get(span.start as usize..span.end as usize).context("predicted training-seen UTF8 slice differs")?}),
    )
}

#[cfg(test)]
mod tests;
