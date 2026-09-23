//! When the output is written during a scan. Everything runs on one clock: the scan line crosses
//! the document in a set number of seconds, and an entity's rows and JSON object are written the
//! moment the line reaches the entity's first byte.

use crate::protocol::Found;

/// Seconds the scan line takes to cross a document that was picked or finished editing.
pub(crate) const SCAN: f64 = 1.8;
/// Seconds for a rescan after a change the reader is still making: an edit or a hint.
pub(crate) const QUICK_SCAN: f64 = 0.6;
/// Seconds the last highlight takes to settle after the line reaches the end.
pub(crate) const SETTLE: f64 = 0.4;

/// When a `scan`-second scan line reaches byte `at` of a `len`-byte document, in seconds.
pub(crate) fn reach(at: usize, len: usize, scan: f64) -> f64 {
    at as f64 / len.max(1) as f64 * scan
}

/// Something the scan writes at a moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Step {
    /// Every entity starting before this byte is written.
    Reach(usize),
    /// Everything is written: the JSON array closes and the count shows.
    Done,
}

/// Every write of a `scan`-second scan over a `len`-byte document, with its time in seconds, in
/// time order. `Done` comes with the last write.
pub(crate) fn schedule(found: &[Found], len: usize, scan: f64) -> Vec<(f64, Step)> {
    let mut steps: Vec<(f64, Step)> = found
        .iter()
        .map(|e| (reach(e.start, len, scan), Step::Reach(e.start + 1)))
        .collect();
    steps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let last = steps.last().map_or(0.0, |s| s.0);
    steps.push((last, Step::Done));
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FoundKind;

    fn found(start: usize) -> Found {
        Found {
            kind: FoundKind::Email,
            start,
            end: start + 1,
            confidence: 1.0,
            review_recommended: false,
            source: "rules".to_string(),
            normalized: None,
            region: None,
            components: Vec::new(),
        }
    }

    #[test]
    fn each_entity_is_written_when_the_line_reaches_it() {
        let steps = schedule(&[found(10), found(50)], 100, SCAN);
        assert_eq!(
            steps,
            vec![
                (reach(10, 100, SCAN), Step::Reach(11)),
                (reach(50, 100, SCAN), Step::Reach(51)),
                (reach(50, 100, SCAN), Step::Done),
            ]
        );
        assert_eq!(schedule(&[], 100, SCAN), vec![(0.0, Step::Done)]);
    }
}
