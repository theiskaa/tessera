//! Versioned detector-only layout signals; parser features keep the legacy contract.

use crate::chunk::Mask;
use crate::features::{FeatureConfig, TokenFeatures, featurize, is_content};
use crate::token::{Token, TokenClass};

/// Numeric flag layout consumed by one detector projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectorFeatureContract {
    /// Existing bits 0 through 22; no table-cell adjacency.
    #[default]
    Legacy23,
    /// Existing bits plus AFTER_TAB at 23 and BEFORE_TAB at 24.
    TabCells25,
}

impl DetectorFeatureContract {
    /// Stable metadata identifier, separate from tokenizer and decoder contracts.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Legacy23 => "tessera-detector-legacy23-v1",
            Self::TabCells25 => "tessera-detector-tab-cells25-v1",
        }
    }

    /// Projection flag input width.
    pub const fn flag_bits(self) -> usize {
        match self {
            Self::Legacy23 => 23,
            Self::TabCells25 => 25,
        }
    }

    /// Reject unknown names or names paired with another width.
    pub fn from_name_and_width(name: &str, width: usize) -> Option<Self> {
        [Self::Legacy23, Self::TabCells25]
            .into_iter()
            .find(|v| v.name() == name && v.flag_bits() == width)
    }
}

pub use crate::features::flag::{AFTER_TAB, BEFORE_TAB};

/// Shared detector featurization; the legacy branch is the unchanged feature function.
pub fn featurize_detector(
    text: &str,
    tokens: &[Token],
    rule_spans: &[(usize, usize)],
    country: Option<&str>,
    config: &FeatureConfig,
    mask: Option<&Mask>,
    contract: DetectorFeatureContract,
) -> Vec<TokenFeatures> {
    let mut out = featurize(text, tokens, rule_spans, country, config, mask);
    if contract == DetectorFeatureContract::Legacy23 {
        return out;
    }
    for (i, token) in tokens.iter().enumerate() {
        if token.class != TokenClass::Space
            || !token.text(text).contains('\t')
            || token.text(text).contains(['\u{b}', '\u{c}'])
            || mask.is_some_and(|m| !m.covers(token))
        {
            continue;
        }
        // The tokenizer emits one horizontal run. Never skip newlines, punctuation,
        // masked content, or a different cell to find an endpoint.
        if let Some(left) = i.checked_sub(1)
            && is_content(&tokens[left])
            && mask.is_none_or(|m| m.covers(&tokens[left]))
        {
            out[left].flags |= BEFORE_TAB;
        }
        if let Some(right) = tokens.get(i + 1)
            && is_content(right)
            && mask.is_none_or(|m| m.covers(right))
        {
            out[i + 1].flags |= AFTER_TAB;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::tokenize;
    fn retained(text: &str, contract: DetectorFeatureContract) -> Vec<TokenFeatures> {
        let tokens = tokenize(text);
        featurize_detector(
            text,
            &tokens,
            &[],
            None,
            &FeatureConfig::default(),
            None,
            contract,
        )
        .into_iter()
        .zip(tokens)
        .filter(|(_, t)| is_content(t))
        .map(|(f, _)| f)
        .collect()
    }
    #[test]
    fn colliding_tables_gain_only_the_two_cell_bits() {
        let a = "Name\tPosition\nMary Vice\tPresident";
        let b = "Name\tPosition\nMary\tVice President";
        let legacy = retained(a, DetectorFeatureContract::Legacy23);
        assert_eq!(legacy, retained(b, DetectorFeatureContract::Legacy23));
        let mut ta = retained(a, DetectorFeatureContract::TabCells25);
        let mut tb = retained(b, DetectorFeatureContract::TabCells25);
        assert_ne!(ta, tb);
        assert!(ta[3].flags & BEFORE_TAB != 0);
        assert!(ta[4].flags & AFTER_TAB != 0);
        assert!(tb[2].flags & BEFORE_TAB != 0);
        assert!(tb[3].flags & AFTER_TAB != 0);
        for fs in [&mut ta, &mut tb] {
            for f in fs.iter_mut() {
                f.flags &= (1 << 23) - 1;
            }
            assert_eq!(*fs, legacy);
        }
    }
    #[test]
    fn horizontal_cells_unicode_punctuation_and_empty_edges() {
        let bits = |text: &str| {
            retained(text, DetectorFeatureContract::TabCells25)
                .into_iter()
                .map(|f| f.flags & (AFTER_TAB | BEFORE_TAB))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits("Zoë \t\t \u{a0} Vice"), vec![BEFORE_TAB, AFTER_TAB]);
        assert_eq!(bits("Mary,\tPresident"), vec![0, BEFORE_TAB, AFTER_TAB]);
        assert_eq!(bits("\tMary\t"), vec![AFTER_TAB | BEFORE_TAB]);
        for text in [
            "Mary President",
            "Mary\n\tPresident",
            "Mary\t\nPresident",
            "Mary\u{2029}President",
        ] {
            let f = bits(text);
            // A leading/trailing TAB is an edge of its own row, not a bridge across rows.
            if text == "Mary\n\tPresident" {
                assert_eq!(f, vec![0, AFTER_TAB]);
            } else if text == "Mary\t\nPresident" {
                assert_eq!(f, vec![BEFORE_TAB, 0]);
            } else {
                assert_eq!(f, vec![0, 0]);
            }
        }
    }
    #[test]
    fn masked_cells_do_not_receive_new_bits() {
        let text = "Mary\tPresident";
        let tokens = tokenize(text);
        let mask = Mask::new(&[(0, 5)], &[]);
        let f = featurize_detector(
            text,
            &tokens,
            &[],
            None,
            &FeatureConfig::default(),
            Some(&mask),
            DetectorFeatureContract::TabCells25,
        );
        assert_eq!(f[2].flags & AFTER_TAB, 0);
        assert_ne!(f[0].flags & BEFORE_TAB, 0);
    }
    #[test]
    fn width_name_pairs_are_closed() {
        assert!(
            DetectorFeatureContract::from_name_and_width(
                DetectorFeatureContract::TabCells25.name(),
                23
            )
            .is_none()
        );
        assert!(DetectorFeatureContract::from_name_and_width("unknown", 25).is_none());
    }
}
