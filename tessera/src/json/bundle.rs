//! A model directory: `bundle.json` naming the weights file beside it and the file's digest.

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

/// The fields of `bundle.json` read here; the rest are ignored.
#[derive(Deserialize)]
struct Manifest {
    runtime: String,
    file: String,
    sha256: String,
    bytes: Option<u64>,
    #[serde(default)]
    metadata: Metadata,
}

/// The `metadata` fields read here.
#[derive(Deserialize, Default)]
struct Metadata {
    /// How the detector was trained to read its input; `known_us` means with a `US` hint.
    detector_input_policy: Option<String>,
}

/// A weights file read from a model directory, with the checksum it must match.
pub(super) struct Bundle {
    /// Where the weights were read from, for error messages.
    pub(super) path: PathBuf,
    /// The weights file's contents, not yet verified.
    pub(super) bytes: Vec<u8>,
    /// The recorded digest in the `sha256-<hex>` form `Config::expected_checksum` takes.
    pub(super) checksum: String,
    /// The region hint a request without one is answered with: the one the detector was
    /// trained to expect.
    pub(super) country_hint: Vec<String>,
}

/// Read `dir/bundle.json` and the weights file it names. Errors are complete messages.
pub(super) fn read(dir: &Path) -> Result<Bundle, String> {
    let manifest_path = dir.join("bundle.json");
    let raw =
        std::fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: Manifest =
        serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let invalid = |why: String| format!("{}: {why}", manifest_path.display());
    if manifest.runtime != "tessera" {
        return Err(invalid(format!(
            "runtime `{}` is not `tessera`",
            manifest.runtime
        )));
    }
    if manifest.sha256.len() != 64 || !manifest.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("sha256 is not a 64-digit hex digest".into()));
    }
    let mut parts = Path::new(&manifest.file).components();
    if !matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    ) {
        return Err(invalid(format!(
            "file `{}` does not name a file in the bundle directory",
            manifest.file
        )));
    }
    let path = dir.join(&manifest.file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(expected) = manifest.bytes
        && expected != bytes.len() as u64
    {
        return Err(format!(
            "{}: expected {expected} bytes, found {}",
            path.display(),
            bytes.len()
        ));
    }
    let country_hint = match manifest.metadata.detector_input_policy.as_deref() {
        Some("known_us") => vec!["US".to_owned()],
        _ => Vec::new(),
    };
    Ok(Bundle {
        path,
        bytes,
        checksum: format!("sha256-{}", manifest.sha256),
        country_hint,
    })
}
