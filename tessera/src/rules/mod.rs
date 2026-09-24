//! Stage 1: deterministic scanners for the kinds with a strict grammar.
//! Their spans are later passed to the detector as features and masked from
//! its output, so the model never labels an email or a phone.

pub(crate) mod email;
pub(crate) mod phone;
#[cfg(feature = "phone-metadata")]
mod phone_tables;

use crate::Entity;

/// Run every scanner over `text` and return their entities sorted by start,
/// non-overlapping. Emails win over phones when spans overlap.
pub fn scan(text: &str, country_hint: &[&str]) -> Vec<Entity> {
    let emails = email::scan(text);
    let phones = phone::scan(text, country_hint);
    let mut out: Vec<Entity> = phones
        .into_iter()
        .filter(|p| !emails.iter().any(|e| p.start < e.end && e.start < p.end))
        .collect();
    out.extend(emails);
    out.sort_by_key(|e| (e.start, e.end));
    out
}

/// Drops rule entities the mask does not fully contain; plain text has no mask.
pub(crate) fn retain_in_mask(entities: &mut Vec<Entity>, mask: Option<&crate::chunk::Mask>) {
    if let Some(mask) = mask {
        entities.retain(|e| mask.contains(e.start, e.end));
    }
}
