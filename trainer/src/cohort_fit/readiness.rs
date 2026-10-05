//! Recheck the fixed fit protocol's nested data receipts through shared validation.

use super::contracts::Manifest;

pub(super) fn verify_nested_receipts(manifest: &Manifest) -> anyhow::Result<()> {
    crate::reviewed_data::verify_nested_receipts(&super::data::inputs(manifest))
}
