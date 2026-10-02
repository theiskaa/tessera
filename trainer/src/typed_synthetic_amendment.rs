//! Current authored packet and zero-update reviewed amendment crosslinks.

use super::manifest::review;
use super::*;

pub(super) fn verify(m: &Manifest, amendment: &Value) -> anyhow::Result<()> {
    let authored = &amendment["authored_augmentation"];
    let packet_rows = std::fs::read_to_string(&m.packet.path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    let native = read(&m.native_dedup)?;
    let distinct = native["documents"]
        .as_array()
        .context("authored dedup documents absent")?
        .len();
    ensure!(
        authored["origin"] == "synthetic_authored"
            && authored["split"] == "train"
            && authored["raw_rows"] == packet_rows
            && authored["distinct_documents"] == distinct
            && authored["duplicates_are_weights"] == false
            && authored["packet"] == serde_json::to_value(&m.packet)?
            && authored["dedup_map"] == serde_json::to_value(&m.native_dedup)?,
        "amendment authored packet/dedup/origin/split/counts differ"
    );
    let phase_pin = file_pin(&authored["reviewed_phase"])?;
    let phase = read(&phase_pin)?;
    ensure!(
        phase["passed"] == true
            && phase["training_ready"] == false
            && phase["model_initialized"] == false
            && phase["model_forwards"] == 0
            && phase["optimizer_updates_executed"] == 0
            && phase["input_sha256"][&m.packet.path] == m.packet.sha256
            && phase["input_sha256"][&m.native_dedup.path] == m.native_dedup.sha256,
        "authored reviewed phase does not bind current packet/dedup or zero-update scope"
    );
    let proof = read(&m.native_proof)?;
    let archival_paths = super::archival::verify(&proof)?;
    if !archival_paths.is_empty() {
        ensure!(
            phase["stock_integration_approved"] == false
                && phase["raw_rows"] == packet_rows
                && phase["distinct_documents"] == distinct
                && phase["duplicates_are_weights"] == false,
            "combined authored phase counts or safety scope differ"
        );
        for (field, pin) in [
            ("native_proof", &m.native_proof),
            ("typed_transport_map", &m.mapping),
            ("typed_parquet", &m.parquet),
        ] {
            ensure!(
                authored[field] == serde_json::to_value(pin)?,
                "combined authored amendment artifact crosslink differs: {field}"
            );
        }
        for field in ["source_archive_aliases", "compiled_artifact_aliases"] {
            ensure!(
                authored[field] == proof[field] && phase[field] == proof[field],
                "combined authored archival crosslink differs: {field}"
            );
        }
        ensure!(
            phase["native_proof"] == serde_json::to_value(&m.native_proof)?,
            "combined phase native proof differs"
        );
        let source_pin = file_pin(&proof["source_archive_aliases"])?;
        let library_pin = file_pin(&proof["compiled_artifact_aliases"])?;
        for review_pin in &m.assembly_reviews {
            review(
                review_pin,
                &[&m.real_amendments, &phase_pin, &source_pin, &library_pin],
            )?;
        }
    }
    review(
        &m.real_amendment_review,
        &[&m.packet, &m.native_dedup, &phase_pin],
    )?;
    Ok(())
}
