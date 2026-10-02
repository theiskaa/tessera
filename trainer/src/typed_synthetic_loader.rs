//! Native annotation loading with explicit authored train-only membership.

use super::manifest::manifest;
use super::*;

pub(super) fn markers(
    origin: Option<&str>,
    split: Option<&str>,
    family: Option<&str>,
    expected: &str,
) -> anyhow::Result<()> {
    ensure!(
        origin == Some("synthetic_authored") && split == Some("train") && family == Some(expected),
        "typed origin/split/family block differs"
    );
    Ok(())
}

pub(super) fn mapping_crosslink(map: &Value, original: &Value) -> anyhow::Result<()> {
    ensure!(
        map.as_object()
            .context("mapping row not object")?
            .iter()
            .filter(|(k, _)| k.as_str() != "prototype_encoded_index")
            .all(|(k, v)| original[k] == *v)
            && original
                .as_object()
                .context("native row invalid")?
                .iter()
                .all(|(k, v)| map[k] == *v),
        "typed/native mapping crosslink differs"
    );
    Ok(())
}

/// Reads typed train parquet, preserving native first-seen dedup and generic family blocks.
pub(crate) fn load(
    cfg: &Config,
    fc: &FeatureConfig,
    base: &[DetectorDoc],
    real: &[DetectorDoc],
) -> anyhow::Result<Loaded> {
    let Some(m) = manifest(cfg)? else {
        return Ok(Loaded::default());
    };
    generation_sources(cfg)?;
    let mapping: Vec<Value> = serde_json::from_value(read(&m.mapping)?)?;
    let dedup = read(&m.native_dedup)?;
    let originals = dedup["documents"]
        .as_array()
        .context("dedup documents absent")?;
    let packet: Vec<Value> = std::fs::read_to_string(&m.packet.path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let frame = ParquetReader::new(File::open(&m.parquet.path)?).finish()?;
    let column = |name: &str| -> anyhow::Result<_> { Ok(frame.column(name)?.str()?.clone()) };
    let (texts, countries, entities, origins, splits, families, ids) = (
        column("text")?,
        column("country")?,
        column("entities_json")?,
        column("origin")?,
        column("split")?,
        column("family")?,
        column("doc_id")?,
    );
    let total = m.families.iter().map(|f| f.documents).sum::<usize>();
    ensure!(
        frame.height() == total && mapping.len() == total,
        "typed parquet/mapping/family sizes differ"
    );
    let protected: BTreeSet<_> = base
        .iter()
        .chain(real)
        .map(|d| crate::export::sha256_hex(d.text.as_bytes()))
        .collect();
    let mut seen = AnnotationDedup::default();
    let mut loaded = Loaded::default();
    let mut aliases = BTreeSet::new();
    let mut native_indices = BTreeSet::new();
    let mut index = 0;
    for family in &m.families {
        for _ in 0..family.documents {
            let text = texts.get(index).context("null typed text")?;
            markers(
                origins.get(index),
                splits.get(index),
                families.get(index),
                &family.name,
            )
            .with_context(|| format!("typed row {index}"))?;
            let doc = crate::detector::annotated_document(
                text,
                countries.get(index).context("null country")?,
                entities.get(index).context("null gold")?,
                fc,
            )?;
            let hash = crate::export::sha256_hex(text.as_bytes());
            ensure!(
                !protected.contains(&hash)
                    && seen.insert(text, &doc.gold, &format!("typed row {index}"))?,
                "typed document duplicates base/real or another typed row"
            );
            let map = &mapping[index];
            let old_index = map["encoded_index"]
                .as_u64()
                .context("native dedup index absent")? as usize;
            ensure!(
                native_indices.insert(old_index),
                "duplicate native dedup mapping"
            );
            let original = originals
                .get(old_index)
                .context("native dedup index invalid")?;
            mapping_crosslink(map, original)?;
            let model_gold:Vec<_>=doc.gold.iter().map(|s| json!({"kind":crate::detector::KINDS[s.kind].as_str(),"start":s.start,"end":s.end})).collect();
            ensure!(
                map["prototype_encoded_index"] == base.len() + index
                    && map["origin"] == "synthetic_authored"
                    && map["family"] == family.name
                    && map["text_sha256"] == hash
                    && map["content_tokens"] == doc.enc.token_spans.len()
                    && map["model_labels"] == json!(model_gold),
                "typed actual encoding differs from native mapping"
            );
            let raw = map["raw_packet_rows"]
                .as_array()
                .context("raw alias set absent")?;
            ensure!(!raw.is_empty(), "typed row has no raw origin");
            ensure!(
                map["raw_ids"]
                    .as_array()
                    .is_some_and(|ids| ids.len() == raw.len()),
                "raw alias/id mapping size differs"
            );
            for (alias_position, alias) in raw.iter().enumerate() {
                let row = alias.as_u64().context("raw alias invalid")? as usize;
                ensure!(aliases.insert(row), "raw alias used twice");
                let p = packet.get(row).context("raw alias outside packet")?;
                ensure!(
                    map["raw_ids"][alias_position] == p["id"],
                    "raw alias document id differs"
                );
                let raw_doc = crate::detector::annotated_document(
                    p["text"].as_str().context("raw text absent")?,
                    p["country"].as_str().context("raw country absent")?,
                    &serde_json::to_string(&p["entities"])?,
                    fc,
                )?;
                ensure!(
                    raw_doc.text == text
                        && raw_doc.gold == doc.gold
                        && p["source_group"] == family.name,
                    "raw alias text/gold/dose source_group differs"
                );
            }
            let id = ids.get(index).context("typed id absent")?;
            ensure!(
                packet[raw[0].as_u64().context("raw first index")? as usize]["id"] == id,
                "typed first-seen document id differs"
            );
            loaded.metadata.push(json!({"id":id,"origin":"synthetic_authored","split":"train","family":family.name,"native_index":base.len()+index,"native_dedup_index":old_index,"text_sha256":hash,"raw_packet_rows":raw}));
            loaded.docs.push(doc);
            index += 1;
        }
        loaded.pieces.push(family.documents);
        loaded.repeats.push(family.repeat);
    }
    ensure!(
        aliases.len() == packet.len()
            && aliases.iter().copied().eq(0..packet.len())
            && native_indices.len() == originals.len(),
        "typed packet/native dedup coverage incomplete"
    );
    Ok(loaded)
}
