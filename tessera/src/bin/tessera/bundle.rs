//! A model directory: `bundle.json` naming the weights file beside it and the file's digest.

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

/// The fields of `bundle.json` the CLI reads; the rest, such as `metadata`, are ignored.
#[derive(Deserialize)]
struct Manifest {
    runtime: String,
    file: String,
    sha256: String,
    bytes: Option<u64>,
}

/// A weights file read from a model directory, with the checksum it must match.
pub(crate) struct Bundle {
    /// Where the weights were read from, for error messages.
    pub(crate) path: PathBuf,
    /// The weights file's contents, not yet verified.
    pub(crate) bytes: Vec<u8>,
    /// The recorded digest in the `sha256-<hex>` form `Config::expected_checksum` takes.
    pub(crate) checksum: String,
}

/// Read `dir/bundle.json` and the weights file it names. Errors are complete messages.
pub(crate) fn read(dir: &Path) -> Result<Bundle, String> {
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
    Ok(Bundle {
        path,
        bytes,
        checksum: format!("sha256-{}", manifest.sha256),
    })
}
