//! Feature `profile`: per-stage timing for the benchmark harness. Never enabled in a published
//! build.
//!
//! The pipeline's own stage calls are wrapped in `stage!`, which without this feature is the
//! bare expression. With it, a timed call installs a recorder for its thread, and each stage
//! adds the time spent in it minus the time of stages nested inside it, so the parser run on a
//! detected address counts as parsing, not detection. The clock is injected: native callers
//! pass `std::time::Instant`, the browser passes `performance.now`, and this module depends
//! on neither.

use std::cell::RefCell;
use std::rc::Rc;

use crate::{Entity, Error, Extraction, Query, Tessera};

/// Milliseconds spent in each stage of one call. Stages not run are zero; `total_ms` also holds
/// the glue between stages, such as windowing and confidence policy.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    pub tokenize_ms: f64,
    pub featurize_ms: f64,
    pub rules_ms: f64,
    pub detect_ms: f64,
    pub parse_ms: f64,
    pub group_ms: f64,
    pub total_ms: f64,
}

/// A timed stage of the pipeline.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Stage {
    Tokenize,
    Featurize,
    Rules,
    Detect,
    Parse,
    Group,
}

struct Recorder {
    now: Rc<dyn Fn() -> f64>,
    spent: [f64; 6],
    /// Open stages: which, when it began, and the time of the stages nested inside it.
    open: Vec<(Stage, f64, f64)>,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// Runs `f` as `stage`, recording its exclusive time when a timed call is in progress.
pub(crate) fn stage<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    let recording = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let Some(rec) = r.as_mut() else {
            return false;
        };
        let began = (rec.now)();
        rec.open.push((stage, began, 0.0));
        true
    });
    let out = f();
    if recording {
        RECORDER.with(|r| {
            if let Some(rec) = r.borrow_mut().as_mut()
                && let Some((stage, began, nested)) = rec.open.pop()
            {
                let elapsed = (rec.now)() - began;
                rec.spent[stage as usize] += elapsed - nested;
                if let Some(parent) = rec.open.last_mut() {
                    parent.2 += elapsed;
                }
            }
        });
    }
    out
}

/// Runs `call` with a recorder installed and returns its result with the stage times.
fn timed<T>(
    now: impl Fn() -> f64 + 'static,
    call: impl FnOnce() -> Result<T, Error>,
) -> Result<(StageTimings, T), Error> {
    let now: Rc<dyn Fn() -> f64> = Rc::new(now);
    RECORDER.with(|r| {
        *r.borrow_mut() = Some(Recorder {
            now: Rc::clone(&now),
            spent: [0.0; 6],
            open: Vec::new(),
        })
    });
    let began = now();
    let result = call();
    let total_ms = now() - began;
    let spent = RECORDER
        .with(|r| r.borrow_mut().take())
        .map_or([0.0; 6], |r| r.spent);
    let [
        tokenize_ms,
        featurize_ms,
        rules_ms,
        detect_ms,
        parse_ms,
        group_ms,
    ] = spent;
    let timings = StageTimings {
        tokenize_ms,
        featurize_ms,
        rules_ms,
        detect_ms,
        parse_ms,
        group_ms,
        total_ms,
    };
    Ok((timings, result?))
}

/// One [`Tessera::extract_contacts`] call, timed per stage. `now` returns milliseconds; only
/// differences are used.
pub fn extract_timed(
    tessera: &Tessera,
    text: &str,
    query: &Query<'_>,
    now: impl Fn() -> f64 + 'static,
) -> Result<(StageTimings, Extraction), Error> {
    timed(now, || tessera.extract_contacts(text, query))
}

/// One [`Tessera::parse_address`] call, timed per stage.
pub fn parse_timed(
    tessera: &Tessera,
    text: &str,
    query: &Query<'_>,
    now: impl Fn() -> f64 + 'static,
) -> Result<(StageTimings, Entity), Error> {
    timed(now, || tessera.parse_address(text, query))
}

impl StageTimings {
    fn fields(&self) -> [f64; 7] {
        [
            self.tokenize_ms,
            self.featurize_ms,
            self.rules_ms,
            self.detect_ms,
            self.parse_ms,
            self.group_ms,
            self.total_ms,
        ]
    }

    fn from_fields(f: [f64; 7]) -> StageTimings {
        StageTimings {
            tokenize_ms: f[0],
            featurize_ms: f[1],
            rules_ms: f[2],
            detect_ms: f[3],
            parse_ms: f[4],
            group_ms: f[5],
            total_ms: f[6],
        }
    }
}

/// Per-field median; the default for an empty slice.
pub fn median(runs: &[StageTimings]) -> StageTimings {
    if runs.is_empty() {
        return StageTimings::default();
    }
    let mut out = [0.0; 7];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut column: Vec<f64> = runs.iter().map(|r| r.fields()[i]).collect();
        column.sort_by(|a, b| a.total_cmp(b));
        *slot = column[column.len() / 2];
    }
    StageTimings::from_fields(out)
}

/// 95th percentile of `total_ms`; 0 for an empty slice.
pub fn p95_total(runs: &[StageTimings]) -> f64 {
    if runs.is_empty() {
        return 0.0;
    }
    let mut totals: Vec<f64> = runs.iter().map(|r| r.total_ms).collect();
    totals.sort_by(|a, b| a.total_cmp(b));
    let idx = ((totals.len() as f64) * 0.95).ceil() as usize;
    totals[idx.saturating_sub(1).min(totals.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(t: f64) -> StageTimings {
        StageTimings {
            total_ms: t,
            ..StageTimings::default()
        }
    }

    #[test]
    fn median_and_p95() {
        assert_eq!(median(&[total(3.0), total(1.0), total(2.0)]).total_ms, 2.0);
        let twenty: Vec<StageTimings> = (1..=20).map(|t| total(t as f64)).collect();
        assert_eq!(p95_total(&twenty), 19.0);
        assert_eq!(median(&[]), StageTimings::default());
        assert_eq!(p95_total(&[]), 0.0);
    }

    #[test]
    fn nested_stages_count_exclusive_time() {
        let clock = Rc::new(std::cell::Cell::new(0.0));
        let tick = Rc::clone(&clock);
        let now = move || {
            tick.set(tick.get() + 1.0);
            tick.get()
        };
        let (t, ()) = timed(now, || {
            stage(Stage::Detect, || {
                stage(Stage::Parse, || ());
            });
            Ok(())
        })
        .unwrap();
        // Clock reads: begin 1, detect in 2, parse in 3, parse out 4, detect out 5, end 6.
        assert_eq!((t.parse_ms, t.detect_ms, t.total_ms), (1.0, 2.0, 5.0));
        assert!(RECORDER.with(|r| r.borrow().is_none()));
        assert_eq!(stage(Stage::Rules, || 7), 7);
    }
}
