//! Address marks: the lines of a sample marked by hand as addresses, which are given to
//! `parse_address`. A mark keeps its text and position; edits before it move it, and an edit
//! that touches it drops it.

/// One marked address: its text and where that text starts in the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mark {
    pub(crate) text: String,
    pub(crate) near: usize,
}

impl Mark {
    /// The mark on `text[start..end]` with surrounding whitespace trimmed, or `None` for an
    /// empty or misaligned span.
    pub(crate) fn at(text: &str, start: usize, end: usize) -> Option<Mark> {
        let span = text.get(start..end)?;
        let trimmed = span.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(Mark {
            text: trimmed.to_string(),
            near: start + (span.len() - span.trim_start().len()),
        })
    }

    fn span(&self) -> (usize, usize) {
        (self.near, self.near + self.text.len())
    }
}

/// The byte ranges of `marks` in `text`, sorted, without overlaps: a mark is kept while its text
/// is still at its position and no earlier mark holds that span.
pub(crate) fn locate(text: &str, marks: &[Mark]) -> Vec<(Mark, (usize, usize))> {
    let mut placed: Vec<(Mark, (usize, usize))> = Vec::new();
    for mark in marks {
        let (start, end) = mark.span();
        let present = text.get(start..end) == Some(mark.text.as_str());
        let free = placed.iter().all(|(_, (a, b))| end <= *a || start >= *b);
        if present && free {
            placed.push((mark.clone(), (start, end)));
        }
    }
    placed.sort_by_key(|(_, span)| *span);
    placed
}

/// `marks` moved from `old` to `new`, a text differing from it in one edited stretch: marks
/// before the stretch stay, marks after it shift by the change in length, and marks it touches
/// are dropped.
pub(crate) fn shift(old: &str, new: &str, marks: &[Mark]) -> Vec<Mark> {
    let prefix = old
        .char_indices()
        .zip(new.chars())
        .find(|((_, a), b)| a != b)
        .map_or(old.len().min(new.len()), |((i, _), _)| i);
    let room = old.len().min(new.len()) - prefix;
    let suffix = old
        .chars()
        .rev()
        .zip(new.chars().rev())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf8())
        .scan(0, |sum, n| {
            *sum += n;
            Some(*sum)
        })
        .take_while(|sum| *sum <= room)
        .last()
        .unwrap_or(0);
    let edited_end = old.len() - suffix;
    marks
        .iter()
        .filter_map(|mark| {
            let (start, end) = mark.span();
            if end <= prefix {
                Some(mark.clone())
            } else if start >= edited_end {
                Some(Mark {
                    text: mark.text.clone(),
                    near: start + new.len() - old.len(),
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(text: &str, near: usize) -> Mark {
        Mark {
            text: text.to_string(),
            near,
        }
    }

    #[test]
    fn a_mark_stays_on_its_own_copy_of_repeated_text() {
        let text = "at 12 12 12 St";
        let m = Mark::at(text, 6, 11).unwrap();
        assert_eq!(m, mark("12 12", 6));
        assert_eq!(locate(text, &[m])[0].1, (6, 11));
    }

    #[test]
    fn edits_move_marks_after_them_and_drop_marks_they_touch() {
        let old = "a: 1 Main St\nb: 1 Main St\n";
        let marks = [mark("1 Main St", 3), mark("1 Main St", 16)];
        let new = format!("new line\n{old}");
        assert_eq!(
            shift(old, &new, &marks),
            [mark("1 Main St", 12), mark("1 Main St", 25)]
        );
        let deleted = "a: 1 Main St\nb: \n";
        assert_eq!(shift(old, deleted, &marks), [mark("1 Main St", 3)]);
        let appended = format!("{old}more");
        assert_eq!(shift(old, &appended, &marks), marks);
        assert_eq!(shift(old, old, &marks), marks);
    }

    #[test]
    fn a_mark_is_trimmed() {
        let text = "at  1 Main St, Leeds  ok";
        assert_eq!(Mark::at(text, 3, 22).unwrap(), mark("1 Main St, Leeds", 4));
        assert!(Mark::at(text, 2, 4).is_none());
        assert!(Mark::at("é", 0, 1).is_none());
    }
}
