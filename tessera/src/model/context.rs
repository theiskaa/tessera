//! The single deployable detector graph beyond the legacy six-block format.

use crate::Error;

/// Exact architecture identifier for the seven-block RMS detector.
pub const CONTEXT96_RMS_NAME: &str = "detector-context96-rms-v2";
/// The retained-token dilation graph, preserving the legacy six-block prefix.
pub const CONTEXT96_RMS_DILATIONS: [usize; 7] = [1, 2, 4, 8, 16, 1, 64];
/// Parser legacy BIO plus detector ADDRESS continuation for format 2.
pub const CONTEXT96_DECODER_CONTRACT: &str = "tessera-bio-context96-address-continuation-v1";
/// Canonical graph, operator, decoding, and window contract stored in format 2.
pub const CONTEXT96_RMS_CONTRACT: &str = "{\"name\":\"detector-context96-rms-v2\",\"dilations\":[1,2,4,8,16,1,64],\"kernel\":3,\"channels\":96,\"operator\":\"post-residual-channel-rms-v2\",\"epsilon\":0.00001,\"learned_parameters\":0,\"margin_tokens\":96,\"overlap_tokens\":448,\"window_tokens\":2048,\"max_entity_tokens\":256,\"decoder_contract\":\"tessera-bio-context96-address-continuation-v1\"}";
/// The canonical contract's normalization epsilon.
pub const CONTEXT96_RMS_EPSILON: f32 = 1e-5;
/// Trusted retained-token context at internal window edges.
pub const CONTEXT96_MARGIN: usize = 96;
/// Overlap preserving the 256-token whole-entity guarantee.
pub const CONTEXT96_OVERLAP: usize = 448;

pub(crate) fn from_manifest(format: &str, contract: Option<&str>) -> Result<bool, Error> {
    match (format, contract) {
        ("1", None) => Ok(false),
        ("1", Some(_)) => Err(Error::BundleInvalid),
        ("2", Some(CONTEXT96_RMS_CONTRACT)) => Ok(true),
        ("2", None) => Err(Error::BundleInvalid),
        _ => Err(Error::UnsupportedVersion),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn context96_geometry_preserves_prefix_and_covers_every_offset() {
        assert_eq!(&CONTEXT96_RMS_DILATIONS[..6], &[1, 2, 4, 8, 16, 1]);
        let mut offsets = BTreeSet::from([0isize]);
        for dilation in CONTEXT96_RMS_DILATIONS {
            offsets = offsets
                .into_iter()
                .flat_map(|x| [-1, 0, 1].map(|k| x + k * dilation as isize))
                .collect();
        }
        assert_eq!(offsets, (-96..=96).collect());
        assert_eq!(
            CONTEXT96_OVERLAP - 2 * CONTEXT96_MARGIN,
            crate::chunk::MAX_ENTITY_TOKENS
        );
    }

    #[test]
    fn context96_identity_refuses_legacy_reinterpretation_and_unknown_graphs() {
        assert!(!from_manifest("1", None).unwrap());
        assert!(from_manifest("2", Some(CONTEXT96_RMS_CONTRACT)).unwrap());
        for (format, contract) in [
            ("1", Some(CONTEXT96_RMS_CONTRACT)),
            ("2", None),
            ("2", Some("residual-rms-v1")),
            ("3", Some(CONTEXT96_RMS_CONTRACT)),
        ] {
            assert!(from_manifest(format, contract).is_err());
        }
        for changed in [
            CONTEXT96_RMS_CONTRACT.replace("64]", "32]"),
            CONTEXT96_RMS_CONTRACT.replace("0.00001", "0.0001"),
            CONTEXT96_RMS_CONTRACT.replace("448", "384"),
            CONTEXT96_RMS_CONTRACT
                .replace(CONTEXT96_DECODER_CONTRACT, super::super::DECODER_CONTRACT),
        ] {
            assert!(from_manifest("2", Some(&changed)).is_err());
        }
    }
}
