//! One-to-one exact matching shared by fixture, detector, and contact scoring.

use serde::Serialize;

/// Whether each prediction consumes an as-yet unmatched gold item.
pub(crate) fn matches<T: PartialEq>(gold: &[T], predicted: &[T]) -> Vec<bool> {
    let mut used = vec![false; gold.len()];
    predicted
        .iter()
        .map(|prediction| {
            if let Some(index) = gold
                .iter()
                .enumerate()
                .position(|(index, expected)| !used[index] && expected == prediction)
            {
                used[index] = true;
                true
            } else {
                false
            }
        })
        .collect()
}

/// Unrounded counts; a wrong boundary or owner contributes both a false positive and a miss.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Counts {
    pub(crate) tp: usize,
    pub(crate) predicted: usize,
    pub(crate) gold: usize,
}

impl Counts {
    /// Accumulate independent documents without averaging their percentages.
    pub(crate) fn add(&mut self, other: &Self) {
        self.tp += other.tp;
        self.predicted += other.predicted;
        self.gold += other.gold;
    }

    /// Precision, recall, and F1, with zero for missing denominators.
    pub(crate) fn prf(&self) -> (f64, f64, f64) {
        crate::eval::prf(self.tp, self.predicted, self.gold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_predictions_only_consume_one_gold_item() {
        assert_eq!(matches(&[1], &[1, 1]), [true, false]);
        assert_eq!(matches(&[1, 1], &[1, 1, 1]), [true, true, false]);
    }

    #[test]
    fn boundary_and_owner_are_part_of_identity() {
        let gold = [("owner-a", "phone", 4, 15)];
        let predicted = [
            ("owner-b", "phone", 4, 15),
            ("owner-a", "phone", 4, 14),
            ("owner-a", "phone", 4, 15),
        ];
        assert_eq!(matches(&gold, &predicted), [false, false, true]);
        assert_eq!(matches::<usize>(&[], &[1]), [false]);
        assert!(matches(&[1], &[]).is_empty());
    }
}
