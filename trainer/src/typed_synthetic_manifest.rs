//! Exact reviewed amendment and packet provenance bindings.

use super::*;

fn review_bindings(review: &Value) -> anyhow::Result<BTreeMap<String, String>> {
    let files = review
        .get("reviewed_files")
        .context("review has no exact reviewed files")?;
    if let Some(object) = files.as_object() {
        return object
            .iter()
            .map(|(p, h)| {
                Ok((
                    p.clone(),
                    h.as_str().context("review hash not string")?.to_owned(),
                ))
            })
            .collect();
    }
    let mut bindings = BTreeMap::new();
    for row in files.as_array().context("review file list invalid")? {
        let pin = file_pin(row)?;
        ensure!(
            bindings.insert(pin.path, pin.sha256).is_none(),
            "duplicate reviewed path"
        );
    }
    Ok(bindings)
}

fn review_set(pins: &[Pin], count: usize) -> anyhow::Result<()> {
    ensure!(
        pins.len() == count
            && pins
                .iter()
                .map(|p| &p.sha256)
                .collect::<BTreeSet<_>>()
                .len()
                == count,
        "review set needs exactly {count} distinct receipt contents"
    );
    let paths = pins
        .iter()
        .map(|p| std::fs::canonicalize(&p.path))
        .collect::<Result<BTreeSet<_>, _>>()?;
    ensure!(paths.len() == count, "review set aliases the same receipt");
    Ok(())
}

pub(super) fn review(pin: &Pin, required: &[&Pin]) -> anyhow::Result<()> {
    let value = read(pin)?;
    ensure!(
        value["passed"] == true && value["blockers"].as_array().is_some_and(Vec::is_empty),
        "review did not pass: {}",
        pin.path
    );
    let bindings = review_bindings(&value)?;
    for artifact in required {
        ensure!(
            bindings.get(&artifact.path) == Some(&artifact.sha256),
            "review {} does not bind {}",
            pin.path,
            artifact.path
        );
        artifact.verify()?;
    }
    Ok(())
}

pub(super) fn manifest(cfg: &Config) -> anyhow::Result<Option<Manifest>> {
    let Some(settings) = cfg
        .detector
        .as_ref()
        .and_then(|d| d.typed_synthetic.as_ref())
    else {
        return Ok(None);
    };
    settings.validate()?;
    let value = read(&settings.manifest)?;
    let m: Manifest = serde_json::from_value(value)?;
    ensure!(
        m.schema == "typed_synthetic_train_only_v1"
            && m.origin == "synthetic_authored"
            && m.split == "train",
        "typed supplement must be explicitly authored and train-only"
    );
    ensure!(
        !m.families.is_empty()
            && m.families
                .iter()
                .all(|f| !f.name.is_empty() && f.documents > 0 && f.repeat > 0)
            && m.families
                .iter()
                .map(|f| &f.name)
                .collect::<BTreeSet<_>>()
                .len()
                == m.families.len(),
        "typed families need unique names and positive document/repeat counts"
    );
    let total = m.families.iter().try_fold(0usize, |n, f| {
        n.checked_add(f.documents)
            .context("typed document count overflow")
    })?;
    m.families.iter().try_fold(0usize, |n, f| {
        f.documents
            .checked_mul(f.repeat)
            .and_then(|v| n.checked_add(v))
            .context("typed draw count overflow")
    })?;
    for artifact in [
        &m.parquet,
        &m.packet,
        &m.mapping,
        &m.native_dedup,
        &m.native_proof,
        &m.real_amendments,
        &m.real_amendment_review,
        &m.real_corrections,
    ] {
        artifact.verify()?;
    }
    let processed = Path::new(&cfg.data.processed);
    for split in ["train", "valid", "test"] {
        ensure!(
            std::fs::canonicalize(&m.parquet.path)?
                != std::fs::canonicalize(processed.join(format!("{split}.parquet")))?,
            "authored parquet cannot replace a generated split"
        );
    }
    ensure!(
        m.label_reviews.len() == 2 && m.label_reviews[0].path != m.label_reviews[1].path,
        "typed packet needs two distinct exact independent label reviews"
    );
    review_set(&m.label_reviews, 2)?;
    for r in &m.label_reviews {
        review(r, &[&m.packet])?;
    }
    ensure!(
        m.assembly_reviews.len() == 2 && m.assembly_reviews[0].path != m.assembly_reviews[1].path,
        "typed native mapping/transport needs two distinct assembly reviews"
    );
    review_set(&m.assembly_reviews, 2)?;
    for review_pin in &m.assembly_reviews {
        review(
            review_pin,
            &[
                &m.packet,
                &m.native_dedup,
                &m.native_proof,
                &m.mapping,
                &m.parquet,
            ],
        )?;
    }
    review(
        &m.real_amendment_review,
        &[&m.real_amendments, &m.real_corrections],
    )?;
    let proof = read(&m.native_proof)?;
    let dedup = read(&m.native_dedup)?;
    ensure!(
        proof["passed"] == true
            && proof["counts"]["documents"] == total
            && proof["counts"]["pieces"] == total
            && proof["counts"]["unreachable_spans"] == 0
            && proof["model_initialized"] == false
            && proof["model_forwards"] == 0
            && proof["optimizer_updates_executed"] == 0,
        "native packet proof differs from typed families"
    );
    for evidence in [&proof, &dedup] {
        ensure!(
            evidence["input_sha256"][&m.packet.path] == m.packet.sha256,
            "native proof/dedup is not for the reviewed packet"
        );
    }
    ensure!(
        dedup["negative_duplicates_are_not_implicit_sampling_weights"] == true
            && dedup["documents"]
                .as_array()
                .is_some_and(|d| d.len() == total),
        "native first-seen dedup coverage differs"
    );
    Ok(Some(m))
}

/// Verifies original generation history and the independently reviewed training substitutions.
pub(crate) fn generation_sources(cfg: &Config) -> anyhow::Result<Option<Vec<String>>> {
    let Some(m) = manifest(cfg)? else {
        return Ok(None);
    };
    let successor = super::successor::verify(cfg)?;
    let selected = successor
        .as_ref()
        .map(|s| s.prior_sources.as_slice())
        .unwrap_or(&cfg.detector.as_ref().context("missing detector")?.silver);
    generation_sources_selected(cfg, &m, selected)
}

fn generation_sources_selected(
    cfg: &Config,
    m: &Manifest,
    selected: &[String],
) -> anyhow::Result<Option<Vec<String>>> {
    let amendment = read(&m.real_amendments)?;
    let correction_count = match amendment["schema"].as_str() {
        Some("reviewed_training_amendments_v3") => 3,
        Some("reviewed_training_amendments_v4") => 5,
        _ => anyhow::bail!("unsupported real amendment chain"),
    };
    ensure!(
        amendment["training_ready"] == false
            && amendment["stock_trainer_accepts_amendments"] == false
            && amendment["model_execution"] == false,
        "amendment must retain zero-update safety flags"
    );
    super::amendment::verify(m, &amendment)?;
    let base_config_pin = file_pin(&amendment["base_config"])?;
    base_config_pin.verify()?;
    let base_cfg = crate::config::load(Path::new(&base_config_pin.path))?;
    ensure!(
        cfg.seed == base_cfg.seed
            && cfg.features == base_cfg.features
            && serde_json::to_value(&cfg.generate)? == serde_json::to_value(&base_cfg.generate)?
            && serde_json::to_value(&cfg.names)? == serde_json::to_value(&base_cfg.names)?
            && cfg.data.countries == base_cfg.data.countries
            && std::fs::canonicalize(&cfg.data.processed)?
                == std::fs::canonicalize(&base_cfg.data.processed)?
            && std::fs::canonicalize(&cfg.data.manifests)?
                == std::fs::canonicalize(&base_cfg.data.manifests)?,
        "typed amendment base config/generation/features/splits differ"
    );
    let generator_pin = file_pin(&amendment["base_generation_manifest"])?;
    let generator = read(&generator_pin)?;
    let configured_generator = Path::new(&cfg.data.manifests).join("detector-synthetic.json");
    ensure!(
        hash(&configured_generator)? == generator_pin.sha256,
        "amendment is not for configured original generator manifest"
    );
    let original = generator["real_silver_sha256"]
        .as_object()
        .context("original generator sources absent")?;
    ensure!(
        amendment["generation_real_sources"].as_object() == Some(original),
        "amendment changed original generation sources"
    );
    let training = amendment["training_real_sources"]
        .as_array()
        .context("training source slots absent")?;
    ensure!(
        training.len() == original.len()
            && training.len() == selected.len()
            && m.original_real_sources.len() == original.len(),
        "training source slot set differs"
    );
    ensure!(
        base_cfg
            .detector
            .as_ref()
            .context("base detector absent")?
            .silver
            .iter()
            .zip(training)
            .all(|(path, row)| row["generation_source"] == *path)
            && base_cfg
                .detector
                .as_ref()
                .context("base detector absent")?
                .silver
                .len()
                == training.len(),
        "base config original source order differs from amendment"
    );
    let mut sources = Vec::new();
    let mut required = vec![generator_pin, m.real_corrections.clone(), base_config_pin];
    let corrections = read(&m.real_corrections)?;
    let changes = corrections["changes"]
        .as_array()
        .context("real correction changes absent")?;
    ensure!(
        corrections["correction_count"] == correction_count && changes.len() == correction_count,
        "real correction set differs from reviewed amendment version"
    );
    if correction_count == 5 {
        required.push(super::label_delta::verify_prior(&corrections)?);
        let prior_pin = file_pin(&amendment["prior_real_amendment"])?;
        let prior = read(&prior_pin)?;
        ensure!(
            prior["schema"] == "reviewed_training_amendments_v3"
                && prior["base_config"] == amendment["base_config"]
                && prior["base_generation_manifest"] == amendment["base_generation_manifest"]
                && prior["generation_real_sources"] == amendment["generation_real_sources"]
                && prior["unchanged_synthetic_shards"] == amendment["unchanged_synthetic_shards"]
                && prior["authored_augmentation"] == amendment["authored_augmentation"],
            "prior amendment generation/authored lineage differs"
        );
        required.push(prior_pin);
    }
    let mut replacements = 0;
    for (slot, (record, path)) in training.iter().zip(selected).enumerate() {
        ensure!(
            record["source_slot"] == slot
                && record["origin"] == "reviewed_real"
                && record["path"] == *path,
            "training source slot/path/origin differs"
        );
        let pin = file_pin(record)?;
        pin.verify()?;
        required.push(pin);
        let source = record["generation_source"]
            .as_str()
            .context("original source reference absent")?;
        let expected = original
            .get(source)
            .and_then(Value::as_str)
            .context("original source not in generation history")?;
        let original_pin = &m.original_real_sources[slot];
        original_pin.verify()?;
        ensure!(
            original_pin.sha256 == expected && Path::new(&original_pin.path).ends_with(source),
            "original source path/hash crosslink differs"
        );
        required.push(original_pin.clone());
        let matching: Vec<_> = changes
            .iter()
            .filter(|change| change["original_path"] == original_pin.path)
            .collect();
        if matching.is_empty() {
            ensure!(
                std::fs::canonicalize(path)? == std::fs::canonicalize(&original_pin.path)?
                    && record["sha256"] == original_pin.sha256,
                "unreviewed real source substitution"
            );
        } else {
            ensure!(
                matching.len() == 1,
                "multiple corrections for one source are not in this reviewed chain"
            );
            let change = matching[0];
            ensure!(
                change["original_sha256"] == original_pin.sha256
                    && change["candidate_path"] == *path
                    && change["candidate_sha256"] == record["sha256"],
                "correction source crosslink differs"
            );
            if correction_count == 3 {
                verify_correction(change)?;
            } else {
                super::label_delta::verify(change)?;
            }
            replacements += 1;
        }
        sources.push(source.to_owned());
    }
    ensure!(
        replacements == correction_count,
        "required reviewed real corrections were not selected"
    );
    ensure!(
        sources.iter().collect::<BTreeSet<_>>().len() == original.len(),
        "generation source aliases or omissions"
    );
    let shards = amendment["unchanged_synthetic_shards"]
        .as_object()
        .context("amendment shard set absent")?;
    ensure!(
        shards.len() == 3
            && ["train", "valid", "test"]
                .iter()
                .all(|s| shards.contains_key(*s)),
        "amendment needs exact train/valid/test split set"
    );
    for split in ["train", "valid", "test"] {
        let pin = file_pin(&shards[split])?;
        pin.verify()?;
        ensure!(
            std::fs::canonicalize(&pin.path)?
                == std::fs::canonicalize(
                    Path::new(&cfg.data.processed).join(format!("{split}.parquet"))
                )?
                && generator["synthetic_parquet_sha256"][split] == pin.sha256,
            "amendment split differs from original generator"
        );
        required.push(pin);
    }
    let reviews = amendment["real_amendment_reviews"]
        .as_array()
        .context("real amendment reviews absent")?;
    ensure!(
        reviews.len() == 3,
        "real amendments need readiness receipt and two independent reviews"
    );
    let review_pins: Vec<_> = reviews
        .iter()
        .map(file_pin)
        .collect::<anyhow::Result<_>>()?;
    ensure!(
        review_pins
            .iter()
            .map(|p| &p.path)
            .collect::<BTreeSet<_>>()
            .len()
            == 3,
        "duplicate real amendment reviews"
    );
    review_set(&review_pins, 3)?;
    for pin in &review_pins {
        let value = read(pin)?;
        ensure!(value["passed"] == true, "real amendment review not passed");
    }
    for pin in &review_pins {
        required.push(pin.clone());
    }
    let correction_refs: Vec<_> = std::iter::once(&m.real_corrections)
        .chain(m.original_real_sources.iter().filter(|pin| {
            changes
                .iter()
                .any(|change| change["original_path"] == pin.path)
        }))
        .collect();
    for pin in &review_pins[1..] {
        review(pin, &correction_refs)?;
    }
    let refs: Vec<_> = required.iter().collect();
    review(&m.real_amendment_review, &refs)?;
    Ok(Some(sources))
}

pub(super) fn verify_correction(change: &Value) -> anyhow::Result<()> {
    let original = change["original_path"]
        .as_str()
        .context("correction original path absent")?;
    let candidate = change["candidate_path"]
        .as_str()
        .context("correction candidate path absent")?;
    let old_text = std::fs::read_to_string(original)?;
    let new_text = std::fs::read_to_string(candidate)?;
    let old: Vec<_> = old_text.lines().collect();
    let new: Vec<_> = new_text.lines().collect();
    ensure!(
        old.len() == new.len() && change["rows"] == old.len(),
        "correction row count differs"
    );
    let row = change["physical_row_1_based"]
        .as_u64()
        .context("correction physical row absent")? as usize;
    ensure!(row > 0 && row <= old.len(), "correction row outside source");
    for (index, (before, after)) in old.iter().zip(&new).enumerate() {
        if index + 1 != row {
            ensure!(
                before == after,
                "unreviewed changed row in correction source"
            );
            continue;
        }
        let mut expected: Value = serde_json::from_str(before)?;
        let actual: Value = serde_json::from_str(after)?;
        ensure!(
            expected["id"] == change["id"]
                && crate::export::sha256_hex(
                    expected["text"]
                        .as_str()
                        .context("correction text absent")?
                        .as_bytes()
                ) == change["text_sha256"],
            "correction identity/text differs"
        );
        let entities = expected["entities"]
            .as_array_mut()
            .context("correction entities absent")?;
        match change["action"].as_str() {
            Some("remove") => {
                let matches: Vec<_> = entities
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| *e == &change["entity"])
                    .map(|(i, _)| i)
                    .collect();
                ensure!(matches.len() == 1, "correction removal entity differs");
                entities.remove(matches[0]);
            }
            Some("add") => {
                ensure!(
                    !entities.contains(&change["entity"]),
                    "correction entity already present"
                );
                entities.push(change["entity"].clone());
                entities.sort_by_key(|e| (e["start"].as_u64(), e["end"].as_u64()));
            }
            _ => anyhow::bail!("unsupported correction action"),
        }
        ensure!(
            expected == actual,
            "correction is not the exact reviewed entities-only edit"
        );
    }
    Ok(())
}

/// All explicit data and proof artifacts captured by the input snapshot.
pub(crate) fn input_paths(cfg: &Config) -> anyhow::Result<Vec<PathBuf>> {
    let Some(m) = manifest(cfg)? else {
        return Ok(Vec::new());
    };
    generation_sources(cfg)?;
    let mut paths = vec![PathBuf::from(
        &cfg.detector
            .as_ref()
            .context("missing detector")?
            .typed_synthetic
            .as_ref()
            .context("missing typed settings")?
            .manifest
            .path,
    )];
    for p in [
        &m.parquet,
        &m.packet,
        &m.mapping,
        &m.native_dedup,
        &m.native_proof,
        &m.real_amendments,
        &m.real_amendment_review,
        &m.real_corrections,
    ] {
        paths.push(PathBuf::from(&p.path));
    }
    for p in m.label_reviews.iter().chain(&m.assembly_reviews) {
        paths.push(PathBuf::from(&p.path));
    }
    let amendment = read(&m.real_amendments)?;
    if amendment["schema"] == "reviewed_training_amendments_v4" {
        let corrections = read(&m.real_corrections)?;
        paths.push(PathBuf::from(
            file_pin(&corrections["prior_corrections"])?.path,
        ));
        paths.push(PathBuf::from(
            file_pin(&amendment["prior_real_amendment"])?.path,
        ));
    }
    paths.push(PathBuf::from(
        file_pin(&amendment["authored_augmentation"]["reviewed_phase"])?.path,
    ));
    paths.extend(super::archival::verify(&read(&m.native_proof)?)?);
    paths.push(PathBuf::from(file_pin(&amendment["base_config"])?.path));
    for p in amendment["real_amendment_reviews"]
        .as_array()
        .context("real reviews absent")?
    {
        paths.push(PathBuf::from(file_pin(p)?.path));
    }
    paths.push(PathBuf::from(
        file_pin(&amendment["base_generation_manifest"])?.path,
    ));
    for pin in &m.original_real_sources {
        paths.push(PathBuf::from(&pin.path));
    }
    for evidence in [&m.native_proof, &m.native_dedup] {
        let value = read(evidence)?;
        for (path, expected) in value["input_sha256"]
            .as_object()
            .context("proof input references absent")?
        {
            ensure!(
                expected.as_str() == Some(hash(Path::new(path))?.as_str()),
                "native proof dependency changed: {path}"
            );
            paths.push(PathBuf::from(path));
        }
    }
    if let Some(successor) = super::successor::verify(cfg)? {
        paths.extend(successor.paths);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}
