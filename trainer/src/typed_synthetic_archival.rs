//! Explicit byte-identical historical source and native-library proof resolution.

use super::*;

pub(super) fn verify(proof: &Value) -> anyhow::Result<Vec<PathBuf>> {
    let Some(source_aliases) = proof.get("source_archive_aliases") else {
        ensure!(
            proof["schema"] != "explicit_archival_combined_native_annotation_proof_v1",
            "archival native proof source aliases absent"
        );
        return Ok(Vec::new());
    };
    ensure!(
        proof["schema"] == "explicit_archival_combined_native_annotation_proof_v1"
            && proof["independent_crosslink_reviews_required"] == 2,
        "archival native proof schema differs"
    );
    let source_pin = file_pin(source_aliases)?;
    let sources = read(&source_pin)?;
    let archive_pin = file_pin(&sources["archive_receipt"])?;
    let archive = read(&archive_pin)?;
    ensure!(
        sources["passed"] == true
            && sources["fallback_to_live_source"] == false
            && archive["passed"] == true
            && archive["files_archived"] == 327
            && archive["live_files_changed"] == false
            && archive["historical_proofs_rewritten"] == false
            && sources["archive_files"] == archive["files"],
        "original source archive aliases differ"
    );
    let mut paths = vec![
        PathBuf::from(source_pin.path),
        PathBuf::from(archive_pin.path),
    ];
    let mut aliases = BTreeMap::new();
    for alias in archive["files"]
        .as_array()
        .context("archive files absent")?
    {
        let original = alias["original_path"]
            .as_str()
            .context("original archive path absent")?;
        let archived = Pin {
            path: alias["archived_path"]
                .as_str()
                .context("archive path absent")?
                .to_owned(),
            sha256: alias["sha256"]
                .as_str()
                .context("archive hash absent")?
                .to_owned(),
        };
        archived.verify()?;
        ensure!(
            aliases
                .insert(original.to_owned(), archived.clone())
                .is_none(),
            "duplicate original source archive alias"
        );
        paths.push(PathBuf::from(archived.path));
    }
    ensure!(
        aliases.len() == 327,
        "original source archive membership differs"
    );
    let library_pin = file_pin(&proof["compiled_artifact_aliases"])?;
    let libraries = read(&library_pin)?;
    let library_receipt_pin = file_pin(&libraries["archive_receipt"])?;
    let library_receipt = read(&library_receipt_pin)?;
    let library_rows = libraries["aliases"]
        .as_array()
        .context("native library alias absent")?;
    ensure!(
        libraries["passed"] == true
            && libraries["fallback_to_current_artifact"] == false
            && libraries["original_proofs_rewritten"] == false
            && library_rows.len() == 1
            && library_receipt["passed"] == true
            && library_receipt["live_file_changed"] == false
            && library_receipt["historical_proofs_rewritten"] == false
            && library_receipt["training_ready"] == false
            && library_receipt["model_initialized"] == false
            && library_receipt["model_forwards"] == 0,
        "native library archive safety or alias set differs"
    );
    let library = &library_rows[0];
    for key in ["original_path", "archived_path", "sha256"] {
        ensure!(
            library[key] == library_receipt[key],
            "native library archive crosslink differs"
        );
    }
    let original = library["original_path"]
        .as_str()
        .context("original library path absent")?;
    let archived = Pin {
        path: library["archived_path"]
            .as_str()
            .context("library archive path absent")?
            .to_owned(),
        sha256: library["sha256"]
            .as_str()
            .context("library archive hash absent")?
            .to_owned(),
    };
    archived.verify()?;
    ensure!(
        aliases
            .insert(original.to_owned(), archived.clone())
            .is_none(),
        "library aliases original source entry"
    );
    paths.extend([
        PathBuf::from(library_pin.path),
        PathBuf::from(library_receipt_pin.path),
        PathBuf::from(archived.path),
    ]);
    let lineage = proof["old_native_proof_lineage"]
        .as_array()
        .context("native proof lineage absent")?;
    ensure!(
        lineage.len() == 2 && sources["old_native_proofs"] == proof["old_native_proof_lineage"],
        "native proof lineage set differs"
    );
    let mut used_sources = BTreeMap::new();
    let mut totals = [0u64; 6];
    for old in lineage {
        let old_pin = file_pin(old)?;
        let old_proof = read(&old_pin)?;
        ensure!(
            old_proof["passed"] == true
                && old_proof["model_initialized"] == false
                && old_proof["model_forwards"] == 0
                && old_proof["optimizer_updates_executed"] == 0
                && old_proof["training_ready"] == false,
            "historical native proof is not zero-update"
        );
        let counts = old_proof
            .get("counts")
            .or_else(|| old_proof.get("native_counts"))
            .context("historical native counts absent")?;
        for (index, key) in [
            "documents",
            "duplicate_documents",
            "pieces",
            "content_tokens",
            "spans",
            "unreachable_spans",
        ]
        .iter()
        .enumerate()
        {
            totals[index] = totals[index]
                .checked_add(
                    counts[key]
                        .as_u64()
                        .context("historical native count invalid")?,
                )
                .context("historical native count overflow")?;
        }
        for (path, expected) in old_proof["input_sha256"]
            .as_object()
            .context("historical proof dependencies absent")?
        {
            let expected = expected.as_str().context("historical proof hash invalid")?;
            let resolved = if let Some(alias) = aliases.get(path) {
                ensure!(
                    alias.sha256 == expected,
                    "historical proof archive hash differs"
                );
                if path != original {
                    used_sources.insert(path.clone(), json!({"original_path":path,"archived_path":alias.path,"sha256":alias.sha256}));
                }
                &alias.path
            } else {
                path
            };
            ensure!(
                hash(Path::new(resolved))? == expected,
                "historical proof resolved dependency changed: {resolved}"
            );
            paths.push(PathBuf::from(resolved));
        }
        paths.push(PathBuf::from(old_pin.path));
    }
    for (index, key) in [
        "documents",
        "duplicate_documents",
        "pieces",
        "content_tokens",
        "spans",
        "unreachable_spans",
    ]
    .iter()
    .enumerate()
    {
        ensure!(
            proof["counts"][key] == totals[index],
            "combined native counts do not conserve historical proof totals"
        );
    }
    let declared: BTreeMap<_, _> = sources["native_proof_source_aliases"]
        .as_array()
        .context("native source aliases absent")?
        .iter()
        .map(|v| {
            Ok((
                v["original_path"]
                    .as_str()
                    .context("native alias original absent")?
                    .to_owned(),
                v.clone(),
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    ensure!(
        declared == used_sources
            && declared.len()
                == sources["native_proof_source_aliases"]
                    .as_array()
                    .context("native source alias set absent")?
                    .len(),
        "native source alias dependency set differs"
    );
    let input = proof["input_sha256"]
        .as_object()
        .context("archival proof inputs absent")?;
    for path in &paths {
        ensure!(
            input
                .get(&path.to_string_lossy().into_owned())
                .and_then(Value::as_str)
                == Some(hash(path)?.as_str()),
            "archival proof does not bind resolved dependency {}",
            path.display()
        );
    }
    Ok(paths)
}
