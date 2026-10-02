//! Substitute context views into an existing source schedule without adding draws.

use std::collections::HashSet;

use anyhow::{Context, Result, ensure};

/// Replace alternating presentations of eligible silver parents with context views.
///
/// Original indices contain synthetics first, followed by silver parents. Each
/// replacement list belongs to the corresponding silver parent, and view indices
/// must be appended after all originals. Batch order, sizes, and source budgets
/// are preserved; synthetics and parents without views remain unchanged.
///
/// `epoch` is zero based. With the same per-parent budget each epoch, presentations
/// alternate continuously between whole rows and views, starting with the whole
/// row at epoch zero. Views rotate in the supplied order across epoch boundaries.
pub(crate) fn substitute_batches(
    batches: &[Vec<usize>],
    synthetic_count: usize,
    original_count: usize,
    replacements: &[Vec<usize>],
    epoch: usize,
) -> Result<Vec<Vec<usize>>> {
    let parent_count = original_count
        .checked_sub(synthetic_count)
        .context("synthetic count exceeds original count")?;
    ensure!(
        replacements.len() == parent_count,
        "replacement mapping must contain one entry per silver parent"
    );
    ensure!(!batches.is_empty(), "batch schedule must not be empty");

    let mut view_indices = HashSet::new();
    for views in replacements {
        for &view in views {
            ensure!(
                view >= original_count,
                "context view {view} overlaps original indices"
            );
            ensure!(
                view_indices.insert(view),
                "context view {view} is mapped more than once"
            );
        }
    }

    let mut parent_presentations = vec![0usize; parent_count];
    for batch in batches {
        ensure!(!batch.is_empty(), "batch must not be empty");
        for &index in batch {
            ensure!(
                index < original_count,
                "scheduled index {index} is not an original row"
            );
            if index >= synthetic_count {
                let count = &mut parent_presentations[index - synthetic_count];
                *count = count
                    .checked_add(1)
                    .context("parent presentation count overflow")?;
            }
        }
    }

    let mut offsets = vec![0usize; parent_count];
    for (parent, views) in replacements.iter().enumerate() {
        if !views.is_empty() {
            offsets[parent] = epoch
                .checked_mul(parent_presentations[parent])
                .context("epoch presentation offset overflow")?;
        }
    }

    let mut occurrences = vec![0usize; parent_count];
    let mut substituted = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut output = Vec::with_capacity(batch.len());
        for &index in batch {
            if index < synthetic_count {
                output.push(index);
                continue;
            }
            let parent = index - synthetic_count;
            let views = &replacements[parent];
            let presentation = offsets[parent]
                .checked_add(occurrences[parent])
                .context("parent presentation index overflow")?;
            occurrences[parent] = occurrences[parent]
                .checked_add(1)
                .context("parent occurrence count overflow")?;
            if !views.is_empty() && presentation % 2 == 1 {
                output.push(views[(presentation / 2) % views.len()]);
            } else {
                output.push(index);
            }
        }
        substituted.push(output);
    }
    Ok(substituted)
}

#[cfg(test)]
mod tests {
    use super::substitute_batches;

    #[test]
    fn substitutions_preserve_every_source_slot_and_batch_budget() {
        let batches = vec![vec![0, 2, 3, 2], vec![1, 2, 4], vec![3, 2, 4, 0]];
        let replacements = vec![vec![5, 6], vec![7], vec![]];
        for epoch in 0..4 {
            let output = substitute_batches(&batches, 2, 5, &replacements, epoch).unwrap();
            assert_eq!(output.len(), batches.len());
            let mut original_counts = [0; 5];
            let mut output_parent_counts = [0; 5];
            for (original, substituted) in batches.iter().zip(&output) {
                assert_eq!(original.len(), substituted.len());
                for (&parent, &index) in original.iter().zip(substituted) {
                    let mapped_parent = match index {
                        5 | 6 => 2,
                        7 => 3,
                        _ => index,
                    };
                    assert_eq!(mapped_parent, parent);
                    if parent < 2 {
                        assert_eq!(index, parent);
                    }
                    original_counts[parent] += 1;
                    output_parent_counts[mapped_parent] += 1;
                }
            }
            assert_eq!(original_counts, output_parent_counts);
        }
    }

    #[test]
    fn parents_without_views_and_synthetics_stay_unchanged() {
        let batches = vec![vec![0, 1, 2], vec![2, 1, 0, 2]];
        assert_eq!(
            substitute_batches(&batches, 1, 3, &[vec![], vec![]], usize::MAX).unwrap(),
            batches
        );
    }

    #[test]
    fn alternation_and_view_rotation_continue_across_odd_epoch_budgets() {
        let batches = vec![vec![0, 0], vec![0]];
        let replacements = vec![vec![1, 2]];
        let expected = [
            vec![vec![0, 1], vec![0]],
            vec![vec![2, 0], vec![1]],
            vec![vec![0, 2], vec![0]],
            vec![vec![1, 0], vec![2]],
        ];
        for (epoch, expected) in expected.iter().enumerate() {
            let output = substitute_batches(&batches, 0, 1, &replacements, epoch).unwrap();
            assert_eq!(&output, expected);
            assert_eq!(
                output,
                substitute_batches(&batches, 0, 1, &replacements, epoch).unwrap()
            );
        }
    }

    #[test]
    fn many_views_do_not_increase_repeated_parent_presentations() {
        let batches = vec![vec![0; 32]];
        let replacements = vec![(1..31).collect()];
        for epoch in 0..3 {
            let output = substitute_batches(&batches, 0, 1, &replacements, epoch).unwrap();
            assert_eq!(output[0].len(), 32);
            assert_eq!(output[0].iter().filter(|&&index| index == 0).count(), 16);
            assert_eq!(output[0].iter().filter(|&&index| index != 0).count(), 16);
            for (slot, &index) in output[0].iter().enumerate() {
                if slot % 2 == 0 {
                    assert_eq!(index, 0);
                } else {
                    assert_eq!(index, (epoch * 16 + slot / 2) % 30 + 1);
                }
            }
        }
    }

    #[test]
    fn capped_fixed_batches_preserve_parent_counts_and_continuous_retention() {
        let replacements = vec![vec![8, 9], vec![10], vec![]];
        let budgets = [32usize, 3, 2];
        let mut original_counts = [0usize; 8];
        let mut mapped_counts = [0usize; 8];
        let mut whole_counts = [0usize; 3];
        let mut view_counts = [0usize; 3];
        let mut checked_batches = 0usize;
        for epoch in 0..8 {
            let mut schedule: Vec<Vec<usize>> = (0..557)
                .map(|batch| (0..32).map(|slot| (batch + slot + epoch) % 5).collect())
                .collect();
            for (parent, &budget) in budgets.iter().enumerate() {
                for occurrence in 0..budget {
                    schedule[(occurrence * 13 + epoch * 7) % 557][parent] = 5 + parent;
                }
            }
            let output = substitute_batches(&schedule, 5, 8, &replacements, epoch).unwrap();
            for (original, substituted) in schedule.iter().zip(output).take(4000 - checked_batches)
            {
                assert_eq!(original.len(), substituted.len());
                for (&parent, index) in original.iter().zip(substituted) {
                    let mapped = match index {
                        8 | 9 => 5,
                        10 => 6,
                        _ => index,
                    };
                    assert_eq!(mapped, parent);
                    original_counts[parent] += 1;
                    mapped_counts[mapped] += 1;
                    if parent < 5 {
                        assert_eq!(index, parent);
                    } else if index == parent {
                        whole_counts[parent - 5] += 1;
                    } else {
                        view_counts[parent - 5] += 1;
                    }
                }
                checked_batches += 1;
            }
            if checked_batches == 4000 {
                break;
            }
        }
        assert_eq!(checked_batches, 4000);
        assert_eq!(original_counts, mapped_counts);
        for parent in 0..2 {
            assert_eq!(
                whole_counts[parent],
                original_counts[5 + parent].div_ceil(2)
            );
            assert_eq!(view_counts[parent], original_counts[5 + parent] / 2);
        }
        assert_eq!(whole_counts[2], original_counts[7]);
        assert_eq!(view_counts[2], 0);
    }

    #[test]
    fn invalid_schedules_and_mappings_fail_closed() {
        assert!(substitute_batches(&[vec![0]], 2, 1, &[], 0).is_err());
        assert!(substitute_batches(&[vec![0]], 0, 1, &[], 0).is_err());
        assert!(substitute_batches(&[], 0, 1, &[vec![]], 0).is_err());
        assert!(substitute_batches(&[vec![]], 0, 1, &[vec![]], 0).is_err());
        assert!(substitute_batches(&[vec![1]], 0, 1, &[vec![1]], 0).is_err());
        assert!(substitute_batches(&[vec![0]], 0, 1, &[vec![0]], 0).is_err());
        assert!(substitute_batches(&[vec![0]], 0, 1, &[vec![1, 1]], 0).is_err());
        assert!(substitute_batches(&[vec![0]], 0, 2, &[vec![2], vec![2]], 0).is_err());
        assert!(substitute_batches(&[vec![0, 0]], 0, 1, &[vec![1]], usize::MAX).is_err());
    }
}
