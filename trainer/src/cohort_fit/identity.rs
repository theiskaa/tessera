//! Code, binary and immutable source identities for a bounded cohort diagnostic.

use super::contracts::digest;
use anyhow::{Context, ensure};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(digest(&std::fs::read(path)?))
}

fn code_files(path: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(!kind.is_symlink(), "code identity refuses symlinked source");
        if kind.is_dir() {
            code_files(&entry.path(), files)?;
        } else if entry.path().extension().is_some_and(|s| s == "rs") {
            files.push(entry.path());
        }
    }
    Ok(())
}

/// Bind live source hashes and require selected files to match their compiled bytes.
pub(crate) fn code_identity() -> anyhow::Result<BTreeMap<String, String>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("workspace root missing")?;
    let mut files = Vec::new();
    for dir in ["trainer/src", "tessera/src"] {
        code_files(&root.join(dir), &mut files)?;
    }
    for file in [
        "Cargo.toml",
        "Cargo.lock",
        "trainer/Cargo.toml",
        "tessera/Cargo.toml",
    ] {
        files.push(root.join(file));
    }
    let mut hashes = BTreeMap::new();
    for file in files {
        hashes.insert(
            file.strip_prefix(root)?.to_string_lossy().into_owned(),
            hash(&file)?,
        );
    }
    for (name, bytes) in [
        (
            "tessera/src/detector_postprocess.rs",
            include_bytes!("../../../tessera/src/detector_postprocess.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/mixed_draws.rs",
            include_bytes!("../reviewed_fit/mixed_draws.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/mixed_accounting.rs",
            include_bytes!("../reviewed_fit/mixed_accounting.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/mixed_tests.rs",
            include_bytes!("../reviewed_fit/mixed_tests.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored.rs",
            include_bytes!("../reviewed_authored.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_loader.rs",
            include_bytes!("../reviewed_authored_loader.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_source.rs",
            include_bytes!("../reviewed_authored_source.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_adjudication.rs",
            include_bytes!("../reviewed_authored_adjudication.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_reflow.rs",
            include_bytes!("../reviewed_authored_reflow.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_tests.rs",
            include_bytes!("../reviewed_authored_tests.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_successor.rs",
            include_bytes!("../reviewed_authored_successor.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_authored_successor_tests.rs",
            include_bytes!("../reviewed_authored_successor_tests.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/supervision.rs",
            include_bytes!("../reviewed_fit/supervision.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/supervision_tests.rs",
            include_bytes!("../reviewed_fit/supervision_tests.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/mixed.rs",
            include_bytes!("../reviewed_fit/mixed.rs").as_slice(),
        ),
        (
            "tessera/src/model/address_field_boundaries.rs",
            include_bytes!("../../../tessera/src/model/address_field_boundaries.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/objective.rs",
            include_bytes!("../reviewed_fit/objective.rs").as_slice(),
        ),
        (
            "tessera/src/detector_features.rs",
            include_bytes!("../../../tessera/src/detector_features.rs").as_slice(),
        ),
        (
            "trainer/src/tab_cell_migration.rs",
            include_bytes!("../tab_cell_migration.rs").as_slice(),
        ),
        (
            "tessera/src/rules/phone_vanity.rs",
            include_bytes!("../../../tessera/src/rules/phone_vanity.rs").as_slice(),
        ),
        (
            "trainer/src/native_checkpoint.rs",
            include_bytes!("../native_checkpoint.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit.rs",
            include_bytes!("../reviewed_fit.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/contract.rs",
            include_bytes!("../reviewed_fit/contract.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/fit.rs",
            include_bytes!("../reviewed_fit/fit.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/observations.rs",
            include_bytes!("../reviewed_fit/observations.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/probe.rs",
            include_bytes!("../reviewed_fit/probe.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/sampling.rs",
            include_bytes!("../reviewed_fit/sampling.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_fit/snapshot.rs",
            include_bytes!("../reviewed_fit/snapshot.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data/composition.rs",
            include_bytes!("../reviewed_data/composition.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_train.rs",
            include_bytes!("../reviewed_train.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_train/preflight.rs",
            include_bytes!("../reviewed_train/preflight.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_train/recipe.rs",
            include_bytes!("../reviewed_train/recipe.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_train/source.rs",
            include_bytes!("../reviewed_train/source.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_train/initialize.rs",
            include_bytes!("../reviewed_train/initialize.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data.rs",
            include_bytes!("../reviewed_data.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data/data.rs",
            include_bytes!("../reviewed_data/data.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data/readiness.rs",
            include_bytes!("../reviewed_data/readiness.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data/receipt.rs",
            include_bytes!("../reviewed_data/receipt.rs").as_slice(),
        ),
        (
            "trainer/src/reviewed_data/preflight.rs",
            include_bytes!("../reviewed_data/preflight.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit.rs",
            include_bytes!("../cohort_fit.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/contracts.rs",
            include_bytes!("contracts.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/data.rs",
            include_bytes!("data.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/fit.rs",
            include_bytes!("fit.rs").as_slice(),
        ),
        (
            "trainer/src/main.rs",
            include_bytes!("../main.rs").as_slice(),
        ),
        (
            "trainer/src/train.rs",
            include_bytes!("../train.rs").as_slice(),
        ),
        ("trainer/src/net.rs", include_bytes!("../net.rs").as_slice()),
        (
            "trainer/src/generate.rs",
            include_bytes!("../generate.rs").as_slice(),
        ),
        (
            "trainer/src/config.rs",
            include_bytes!("../config.rs").as_slice(),
        ),
        (
            "trainer/src/detector.rs",
            include_bytes!("../detector.rs").as_slice(),
        ),
        (
            "trainer/src/dataset.rs",
            include_bytes!("../dataset.rs").as_slice(),
        ),
        (
            "trainer/src/quantize.rs",
            include_bytes!("../quantize.rs").as_slice(),
        ),
        (
            "trainer/src/diagnostic_operator.rs",
            include_bytes!("../diagnostic_operator.rs").as_slice(),
        ),
        (
            "trainer/src/training_diagnostic.rs",
            include_bytes!("../training_diagnostic.rs").as_slice(),
        ),
        (
            "trainer/src/residual_rms.rs",
            include_bytes!("../residual_rms.rs").as_slice(),
        ),
        (
            "trainer/src/exact_metrics.rs",
            include_bytes!("../exact_metrics.rs").as_slice(),
        ),
        (
            "trainer/src/fullmix_rms.rs",
            include_bytes!("../fullmix_rms.rs").as_slice(),
        ),
        (
            "trainer/src/fullmix_checkpoint.rs",
            include_bytes!("../fullmix_checkpoint.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/identity.rs",
            include_bytes!("identity.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/readiness.rs",
            include_bytes!("readiness.rs").as_slice(),
        ),
        (
            "trainer/src/diagnostic_decode.rs",
            include_bytes!("../diagnostic_decode.rs").as_slice(),
        ),
        (
            "trainer/src/eval.rs",
            include_bytes!("../eval.rs").as_slice(),
        ),
        (
            "Cargo.toml",
            include_bytes!("../../../Cargo.toml").as_slice(),
        ),
        (
            "Cargo.lock",
            include_bytes!("../../../Cargo.lock").as_slice(),
        ),
        (
            "trainer/Cargo.toml",
            include_bytes!("../../Cargo.toml").as_slice(),
        ),
        (
            "tessera/Cargo.toml",
            include_bytes!("../../../tessera/Cargo.toml").as_slice(),
        ),
        (
            "tessera/src/bin/tessera.rs",
            include_bytes!("../../../tessera/src/bin/tessera.rs").as_slice(),
        ),
        (
            "tessera/src/chunk.rs",
            include_bytes!("../../../tessera/src/chunk.rs").as_slice(),
        ),
        (
            "tessera/src/detect.rs",
            include_bytes!("../../../tessera/src/detect.rs").as_slice(),
        ),
        (
            "tessera/src/features.rs",
            include_bytes!("../../../tessera/src/features.rs").as_slice(),
        ),
        (
            "tessera/src/group.rs",
            include_bytes!("../../../tessera/src/group.rs").as_slice(),
        ),
        (
            "tessera/src/lib.rs",
            include_bytes!("../../../tessera/src/lib.rs").as_slice(),
        ),
        (
            "tessera/src/markdown.rs",
            include_bytes!("../../../tessera/src/markdown.rs").as_slice(),
        ),
        (
            "tessera/src/model/address_continuation.rs",
            include_bytes!("../../../tessera/src/model/address_continuation.rs").as_slice(),
        ),
        (
            "tessera/src/model/bio.rs",
            include_bytes!("../../../tessera/src/model/bio.rs").as_slice(),
        ),
        (
            "tessera/src/model/context.rs",
            include_bytes!("../../../tessera/src/model/context.rs").as_slice(),
        ),
        (
            "tessera/src/model/json.rs",
            include_bytes!("../../../tessera/src/model/json.rs").as_slice(),
        ),
        (
            "tessera/src/model/kernels.rs",
            include_bytes!("../../../tessera/src/model/kernels.rs").as_slice(),
        ),
        (
            "tessera/src/model/mod.rs",
            include_bytes!("../../../tessera/src/model/mod.rs").as_slice(),
        ),
        (
            "tessera/src/model/rms.rs",
            include_bytes!("../../../tessera/src/model/rms.rs").as_slice(),
        ),
        (
            "tessera/src/model/sha256.rs",
            include_bytes!("../../../tessera/src/model/sha256.rs").as_slice(),
        ),
        (
            "tessera/src/model/weights.rs",
            include_bytes!("../../../tessera/src/model/weights.rs").as_slice(),
        ),
        (
            "tessera/src/policy.rs",
            include_bytes!("../../../tessera/src/policy.rs").as_slice(),
        ),
        (
            "tessera/src/profile.rs",
            include_bytes!("../../../tessera/src/profile.rs").as_slice(),
        ),
        (
            "tessera/src/rules/email.rs",
            include_bytes!("../../../tessera/src/rules/email.rs").as_slice(),
        ),
        (
            "tessera/src/rules/mod.rs",
            include_bytes!("../../../tessera/src/rules/mod.rs").as_slice(),
        ),
        (
            "tessera/src/rules/phone.rs",
            include_bytes!("../../../tessera/src/rules/phone.rs").as_slice(),
        ),
        (
            "tessera/src/rules/phone_tables.rs",
            include_bytes!("../../../tessera/src/rules/phone_tables.rs").as_slice(),
        ),
        (
            "tessera/src/rules/region.rs",
            include_bytes!("../../../tessera/src/rules/region.rs").as_slice(),
        ),
        (
            "tessera/src/token.rs",
            include_bytes!("../../../tessera/src/token.rs").as_slice(),
        ),
        (
            "tessera/src/wasm.rs",
            include_bytes!("../../../tessera/src/wasm.rs").as_slice(),
        ),
        (
            "trainer/src/cohort_fit/scoring.rs",
            include_bytes!("scoring.rs").as_slice(),
        ),
    ] {
        ensure!(
            hashes.get(name) == Some(&digest(bytes)),
            "compiled fit source differs from disk: {name}; rebuild before preflight"
        );
    }
    Ok(hashes)
}

/// Hash every original source-run artifact, rejecting symlinked provenance.
pub(crate) fn source_identity(root: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    fn visit(
        root: &Path,
        path: &Path,
        hashes: &mut BTreeMap<String, String>,
    ) -> anyhow::Result<()> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            ensure!(
                !kind.is_symlink(),
                "immutable cohort source refuses symlinked artifacts"
            );
            if kind.is_dir() {
                visit(root, &entry.path(), hashes)?;
            } else if kind.is_file() {
                hashes.insert(
                    entry
                        .path()
                        .strip_prefix(root)?
                        .to_string_lossy()
                        .into_owned(),
                    hash(&entry.path())?,
                );
            }
        }
        Ok(())
    }
    let mut hashes = BTreeMap::new();
    visit(root, root, &mut hashes)?;
    Ok(hashes)
}
