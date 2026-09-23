//! The showcase page and its inference worker. Both binaries link this library: the page mounts
//! [`ui::App`], the worker registers [`inference::Inference`], and [`protocol`] is what passes
//! between them.

pub mod inference;
pub mod protocol;
pub mod ui;

mod samples;
mod stats;
