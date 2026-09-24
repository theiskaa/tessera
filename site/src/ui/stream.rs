//! When the output is written during a scan. Everything runs on one clock: the scan line crosses
//! the scanned span (the document, or the section the reader selected) in a set number of
//! seconds, and an entity's rows and JSON object are written the moment the line reaches the
//! entity's first byte.

use crate::protocol::Found;

/// Seconds the scan line takes to cross a document that was picked or finished editing.
pub(crate) const SCAN: f64 = 1.8;
/// Seconds for a rescan after a change the reader is still making: an edit or a hint.
pub(crate) const QUICK_SCAN: f64 = 0.6;
/// Seconds the last highlight takes to settle after the line reaches the end.
pub(crate) const SETTLE: f64 = 0.4;

/// How far through the bytes `span` byte `at` lies, from 0 at its start to 1 at its end.
pub(crate) fn share(at: usize, span: (usize, usize)) -> f64 {
    let through = at.saturating_sub(span.0) as f64 / span.1.saturating_sub(span.0).max(1) as f64;
    through.min(1.0)
}

/// When a `scan`-second scan line over the bytes `span` reaches byte `at`, in seconds.
pub(crate) fn reach(at: usize, span: (usize, usize), scan: f64) -> f64 {
    share(at, span) * scan
}

/// Something the scan writes at a moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Step {
    /// Every entity starting before this byte is written.
    Reach(usize),
    /// Everything is written: the JSON array closes and the count shows.
    Done,
}

/// Every write of a `scan`-second scan over the bytes `span`, with its time in seconds, in time
/// order. `Done` comes with the last write.
pub(crate) fn schedule(found: &[Found], span: (usize, usize), scan: f64) -> Vec<(f64, Step)> {
    let mut steps: Vec<(f64, Step)> = found
        .iter()
        .map(|e| (reach(e.start, span, scan), Step::Reach(e.start + 1)))
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
        let steps = schedule(&[found(10), found(50)], (0, 100), SCAN);
        assert_eq!(
            steps,
            vec![
                (reach(10, (0, 100), SCAN), Step::Reach(11)),
                (reach(50, (0, 100), SCAN), Step::Reach(51)),
                (reach(50, (0, 100), SCAN), Step::Done),
            ]
        );
        assert_eq!(schedule(&[], (0, 100), SCAN), vec![(0.0, Step::Done)]);
    }

    #[test]
    fn a_section_scan_runs_over_the_section_only() {
        assert_eq!(reach(40, (40, 60), SCAN), 0.0);
        assert_eq!(reach(60, (40, 60), SCAN), SCAN);
        assert_eq!(reach(50, (40, 60), SCAN), SCAN / 2.0);
    }
}
