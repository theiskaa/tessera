//! Whole-source ordered reflow reconstruction; original native evidence stays distinct.

use super::super::{Row, Span};
use crate::reviewed_data::{Receipt, digest};
use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Copy {
    source_start: usize,
    source_end: usize,
    output_start: usize,
    output_end: usize,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Insertion {
    literal: String,
    source_boundary: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct ReflowMap {
    name: String,
    native_source_document: String,
    source_dataset: Receipt,
    source_row_index: usize,
    source_row_sha256: String,
    source_context_evidence: Vec<Value>,
    forbidden_inferences: Vec<String>,
    all_source_bytes_in_order_exactly_once: bool,
    literal_copy_map: Vec<Copy>,
    inserted_whitespace_or_delimiters: Vec<Insertion>,
    contact_regions: Vec<(usize, usize)>,
}

fn regions(name: &str, source: &str) -> anyhow::Result<Vec<(usize, usize)>> {
    let marker = "FOR FURTHER INFORMATION CONTACT:\n";
    if let Some(at) = source.find(marker) {
        let start = at + marker.len();
        return Ok(vec![(
            start,
            source[start..]
                .find("\n\n")
                .map_or(source.len(), |n| start + n),
        )]);
    }
    if name == "seen-real-32-0006" {
        return Ok(vec![(
            0,
            source
                .find("Biography\n")
                .context("original biography boundary absent")?,
        )]);
    }
    Ok(vec![(0, source.len())])
}

type Reconstruction = (String, Vec<Copy>, Vec<Insertion>, Vec<usize>);

fn reconstruct(
    source: &str,
    gold: &[Span],
    format: &str,
    regions: &[(usize, usize)],
) -> anyhow::Result<Reconstruction> {
    ensure!(
        [
            "contact_lines",
            "contact_tabs",
            "contact_pipes",
            "soft_wrap"
        ]
        .contains(&format),
        "unreviewed whole-source reflow format"
    );
    let mut insertions = BTreeMap::new();
    let mut width = 0;
    for (index, &byte) in source.as_bytes().iter().enumerate() {
        width = if byte == b'\n' { 0 } else { width + 1 };
        if format == "soft_wrap" {
            if byte == b' ' && width >= 50 {
                insertions.insert(index + 1, "\n");
                width = 0;
            }
        } else if matches!(byte, b',' | b';')
            && regions.iter().any(|&(a, b)| a <= index && index < b)
            && !gold
                .iter()
                .any(|s| s.start as usize <= index && index < s.end as usize)
        {
            insertions.insert(
                index + 1,
                match format {
                    "contact_lines" => "\n",
                    "contact_tabs" => "\t",
                    _ => " | ",
                },
            );
        }
    }
    ensure!(
        !insertions.is_empty(),
        "whole-source reflow has no actual formatting change"
    );
    let mut output = Vec::new();
    let mut copies = Vec::new();
    let mut positions = Vec::new();
    let mut previous = 0;
    let mut extras = Vec::new();
    for (&end, &literal) in &insertions {
        let output_start = output.len();
        output.extend_from_slice(
            source
                .as_bytes()
                .get(previous..end)
                .context("invalid source insertion boundary")?,
        );
        positions.extend(output_start..output.len());
        copies.push(Copy {
            source_start: previous,
            source_end: end,
            output_start,
            output_end: output.len(),
        });
        output.extend_from_slice(literal.as_bytes());
        extras.push(Insertion {
            literal: literal.to_owned(),
            source_boundary: end,
        });
        previous = end;
    }
    let output_start = output.len();
    output.extend_from_slice(&source.as_bytes()[previous..]);
    positions.extend(output_start..output.len());
    copies.push(Copy {
        source_start: previous,
        source_end: source.len(),
        output_start,
        output_end: output.len(),
    });
    ensure!(
        positions.len() == source.len(),
        "source-byte mapping incomplete"
    );
    Ok((String::from_utf8(output)?, copies, extras, positions))
}

pub(in super::super) fn verify(
    row: &Row,
    map: &ReflowMap,
    prepared: &Value,
    read: &mut dyn FnMut(&Receipt) -> anyhow::Result<Vec<u8>>,
) -> anyhow::Result<Value> {
    let source = prepared["input"]
        .as_str()
        .context("source prepared text absent")?;
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
        "reflow source prepared identity/status/policy differs"
    );
    ensure!(
        map.name == row.name
            && map.native_source_document == row.authored_source.native_name
            && map.source_dataset.path == row.authored_source.native_dataset.path
            && map.source_dataset.sha256 == row.authored_source.native_dataset.sha256
            && map.source_row_index == row.authored_source.native_row_index
            && map.all_source_bytes_in_order_exactly_once,
        "reflow map changes source prepared identity"
    );
    let source_bytes = read(&map.source_dataset)?;
    let raw_line = std::str::from_utf8(&source_bytes)?
        .lines()
        .nth(map.source_row_index)
        .context("source prepared physical row absent")?;
    ensure!(
        digest(raw_line.as_bytes()) == map.source_row_sha256
            && serde_json::from_str::<Value>(raw_line)? == *prepared,
        "reflow source physical row hash/payload differs"
    );
    let origin = &prepared["origin"];
    let original_pin: Receipt = serde_json::from_value(origin["dataset"].clone())?;
    let original_bytes = read(&original_pin)?;
    let index = usize::try_from(
        origin["row_index"]
            .as_u64()
            .context("native original index absent")?,
    )?;
    let original: Value = serde_json::from_str(
        std::str::from_utf8(&original_bytes)?
            .lines()
            .nth(index)
            .context("native original row absent")?,
    )?;
    ensure!(
        original["country"] == "US"
            && original["text"] == prepared["input"]
            && original["id"] == origin["row_id"]
            && original["id"].is_string()
            && origin["text_sha256"] == row.authored_source.native_text_sha256
            && original["source"]
                .as_str()
                .is_some_and(|source| !source.trim().is_empty())
            && (original["source_url"].is_null()
                || original["source_url"]
                    .as_str()
                    .is_some_and(|url| url.starts_with("https://"))),
        "full native original id/text/source URL binding differs"
    );
    let native_gold: Vec<Span> = serde_json::from_value(prepared["expected"].clone())?;
    super::super::loader::validate_spans(source, &native_gold)?;
    let contact_regions = regions(&row.authored_source.native_name, source)?;
    let (output, copies, extras, positions) = reconstruct(
        source,
        &native_gold,
        &row.authored_source.format,
        &contact_regions,
    )?;
    ensure!(
        output == row.input
            && copies == map.literal_copy_map
            && extras == map.inserted_whitespace_or_delimiters
            && contact_regions == map.contact_regions,
        "whole-source reflow reconstruction or copy-map differs"
    );
    for context in &map.source_context_evidence {
        let a = usize::try_from(
            context["start"]
                .as_u64()
                .context("source context start absent")?,
        )?;
        let b = usize::try_from(
            context["end"]
                .as_u64()
                .context("source context end absent")?,
        )?;
        ensure!(
            a < b && source.get(a..b) == context["quote"].as_str(),
            "source context exact quote differs"
        );
    }
    let mut transferred = Vec::new();
    for span in native_gold {
        let a = *positions
            .get(span.start as usize)
            .context("mapped target start absent")?;
        let b = positions
            .get(span.end as usize - 1)
            .context("mapped target end absent")?
            + 1;
        let quote = output.get(a..b).context("mapped target not UTF8 aligned")?;
        ensure!(
            quote.split_whitespace().collect::<String>()
                == span.text.split_whitespace().collect::<String>(),
            "nonwhitespace entity bytes changed"
        );
        transferred.push(Span {
            kind: span.kind,
            text: quote.to_owned(),
            start: u32::try_from(a)?,
            end: u32::try_from(b)?,
        });
    }
    transferred.sort();
    let mut final_gold = row.expected.clone();
    final_gold.sort();
    ensure!(
        transferred == final_gold,
        "source projected occurrences differ from source-adjudicated final gold"
    );
    Ok(
        json!({"authored_source":row.authored_source,"native_source_metadata_reference":origin,
        "original_source":original_pin,"original_source_url":original["source_url"],"original_source_url_recorded":original["source_url"].is_string(),"source_context_evidence":map.source_context_evidence,
        "forbidden_inferences":map.forbidden_inferences,"full_source_copy_reconstructed":true,"not_native_origin":true,
        "capture_limit":"Original native prepared text and recorded source URL only; this packet does not declare a new raw HTML capture."}),
    )
}
