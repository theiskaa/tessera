//! Exact staff-table reconstruction and original native source binding.

use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::Value;

use super::{Row, Span};
use crate::reviewed_data::{Receipt, digest};

#[path = "reviewed_authored_reflow.rs"]
mod reflow;

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum CopyMap {
    Staff(StaffMap),
    Reflow(reflow::ReflowMap),
}

pub(super) fn verify(
    row: &Row,
    map: &CopyMap,
    prepared: &Value,
    read: &mut dyn FnMut(&Receipt) -> anyhow::Result<Vec<u8>>,
) -> anyhow::Result<Value> {
    match map {
        CopyMap::Staff(map) => verify_staff(row, map, prepared, read),
        CopyMap::Reflow(map) => reflow::verify(row, map, prepared, read),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Copy {
    source_start: usize,
    source_end: usize,
    output_start: usize,
    output_end: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StaffMap {
    name: String,
    native_source_document: String,
    native_input_sha256: String,
    input_sha256: String,
    literal_copies: Vec<Copy>,
    all_nonwhitespace_source_bytes_retained_exactly_once: bool,
    original_gold_occurrences_retained: usize,
    meaning_changed: bool,
}

fn cell(source: &str, range: (usize, usize), out: &mut Vec<u8>, copies: &mut Vec<Copy>) {
    let start = out.len();
    out.extend_from_slice(&source.as_bytes()[range.0..range.1]);
    copies.push(Copy {
        source_start: range.0,
        source_end: range.1,
        output_start: start,
        output_end: out.len(),
    });
}

fn render(source: &str, format: &str) -> anyhow::Result<(String, Vec<Copy>)> {
    ensure!(
        !source.contains('\r'),
        "staff source must retain its native LF/TAB serialization"
    );
    let mut lines = Vec::new();
    let mut cursor = 0;
    for line in source.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        let mut at = cursor;
        let cells: Vec<_> = content
            .split('\t')
            .map(|value| {
                let range = (at, at + value.len());
                at += value.len() + 1;
                range
            })
            .collect();
        lines.push(cells);
        cursor += line.len();
    }
    ensure!(
        cursor == source.len()
            && lines.len() >= 3
            && lines[0].len() == 1
            && lines[1..].iter().all(|line| line.len() == 4),
        "whole native staff table required"
    );
    let headers = ["Name", "Position", "Phone Number", "Email"];
    ensure!(
        lines[1]
            .iter()
            .zip(headers)
            .all(|(&(a, b), expected)| &source[a..b] == expected),
        "native table headers differ"
    );
    ensure!(
        [
            "pipe_table",
            "labeled_cards",
            "compact_cards",
            "reversed_pipe_table"
        ]
        .contains(&format),
        "unreviewed authored format"
    );
    let order = if format == "reversed_pipe_table" {
        [3, 2, 1, 0]
    } else {
        [0, 1, 2, 3]
    };
    let cards = matches!(format, "labeled_cards" | "compact_cards");
    let mut out = Vec::new();
    let mut copies = Vec::new();
    cell(source, lines[0][0], &mut out, &mut copies);
    out.push(b'\n');
    for (i, &column) in order.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b" | ");
        }
        cell(source, lines[1][column], &mut out, &mut copies);
    }
    for row in &lines[2..] {
        out.extend_from_slice(if cards { b"\n\n" } else { b"\n" });
        for (i, &column) in order.iter().enumerate() {
            if i > 0 {
                out.extend_from_slice(match format {
                    "labeled_cards" => b"\n",
                    "compact_cards" => b"; ",
                    _ => b" | ",
                });
            }
            if cards {
                out.extend_from_slice(headers[column].as_bytes());
                out.extend_from_slice(b": ");
            }
            cell(source, row[column], &mut out, &mut copies);
        }
    }
    Ok((String::from_utf8(out)?, copies))
}

fn verify_staff(
    row: &Row,
    map: &StaffMap,
    prepared: &Value,
    read: &mut dyn FnMut(&Receipt) -> anyhow::Result<Vec<u8>>,
) -> anyhow::Result<Value> {
    let source = prepared["input"]
        .as_str()
        .context("source prepared text missing")?;
    ensure!(
        prepared["name"] == row.authored_source.native_name
            && prepared["country"] == "US"
            && prepared["status"] == "resolved"
            && prepared["annotation_policy_sha256"] == row.annotation_policy_sha256
            && prepared["uncertainties"]
                .as_array()
                .is_some_and(Vec::is_empty)
            && prepared["unresolved"].as_array().is_some_and(Vec::is_empty)
            && digest(source.as_bytes()) == row.authored_source.native_text_sha256,
        "source prepared identity/status/policy differs"
    );
    let origin = &prepared["origin"];
    let original_pin: Receipt = serde_json::from_value(origin["dataset"].clone())?;
    let original_bytes = read(&original_pin)?;
    let index = origin["row_index"]
        .as_u64()
        .context("original row index absent")? as usize;
    let original: Value = serde_json::from_str(
        std::str::from_utf8(&original_bytes)?
            .lines()
            .nth(index)
            .context("original native row missing")?,
    )?;
    ensure!(
        original["name"] == prepared["name"]
            && origin["row_id"] == prepared["name"]
            && original["input"] == prepared["input"]
            && original["country"] == "US"
            && origin["text_sha256"] == row.authored_source.native_text_sha256,
        "source prepared row differs from full original native row"
    );
    for (field, native) in [
        ("source_document_id", "source_document_key"),
        ("parent_source_id", "parent_source_id"),
        ("source_family_id", "source_family_id"),
        ("domain", "publisher_host"),
    ] {
        ensure!(
            origin[field].is_string() && origin[field] == original[native],
            "native source {field} differs"
        );
    }
    ensure!(
        original["source_block"]["complete_native_table"] == true,
        "original source is not a complete table"
    );
    let raw_pin: Receipt = serde_json::from_value(original["raw_source"].clone())?;
    let raw = read(&raw_pin)?;
    let block = &original["source_block"];
    let a = block["raw_html_start"]
        .as_u64()
        .context("raw table start absent")? as usize;
    let b = block["raw_html_end"]
        .as_u64()
        .context("raw table end absent")? as usize;
    ensure!(
        a < b
            && raw
                .get(a..b)
                .is_some_and(|bytes| Value::String(digest(bytes)) == block["raw_html_sha256"]),
        "captured native table hash differs"
    );
    let projection = Receipt {
        path: original["projection"]["script"]
            .as_str()
            .context("projection path absent")?
            .into(),
        sha256: original["projection"]["script_sha256"]
            .as_str()
            .context("projection hash absent")?
            .to_owned(),
    };
    read(&projection)?;
    let (reconstructed, copies) = render(source, &row.authored_source.format)?;
    ensure!(
        reconstructed == row.input
            && copies == map.literal_copies
            && map.name == row.name
            && map.native_source_document == row.authored_source.native_name
            && map.native_input_sha256 == row.authored_source.native_text_sha256
            && map.input_sha256 == digest(row.input.as_bytes())
            && map.all_nonwhitespace_source_bytes_retained_exactly_once
            && !map.meaning_changed,
        "whole-source copy-map reconstruction differs"
    );
    let native_gold: Vec<Span> = serde_json::from_value(prepared["expected"].clone())?;
    super::loader::validate_spans(source, &native_gold)?;
    let mut transferred = Vec::new();
    for span in &native_gold {
        let matches: Vec<_> = copies
            .iter()
            .filter(|c| c.source_start <= span.start as usize && span.end as usize <= c.source_end)
            .collect();
        ensure!(
            matches.len() == 1,
            "source expected span is not copied exactly once"
        );
        let copy = matches[0];
        let start = copy.output_start + span.start as usize - copy.source_start;
        let end = copy.output_start + span.end as usize - copy.source_start;
        ensure!(
            row.input.get(start..end) == Some(span.text.as_str()),
            "mapped source span quote differs"
        );
        transferred.push(Span {
            kind: span.kind.clone(),
            text: span.text.clone(),
            start: u32::try_from(start)?,
            end: u32::try_from(end)?,
        });
    }
    transferred.sort();
    let mut final_gold = row.expected.clone();
    final_gold.sort();
    ensure!(
        transferred == final_gold && map.original_gold_occurrences_retained == native_gold.len(),
        "source transfer differs from separately reviewed authored gold"
    );
    Ok(
        serde_json::json!({"authored_source":row.authored_source, "native_source_metadata_reference":origin,
        "raw_capture":raw_pin, "raw_table_sha256":block["raw_html_sha256"], "projection":projection,
        "full_source_copy_reconstructed":true, "not_native_origin":true}),
    )
}
