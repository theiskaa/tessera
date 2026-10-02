//! Explicit evaluation-only decoder selection; absent selection preserves legacy behavior.

use anyhow::ensure;
use serde_json::{Value, json};
use tessera::internal::{DetectedSpan, decode_detector, decode_detector_with_text};

/// A decoder option confined to reviewed final training-seen checkpoint evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Decoder {
    AddressContinuationV1,
}

impl Decoder {
    pub(crate) fn identifier(self) -> &'static str {
        match self {
            Self::AddressContinuationV1 => "address-continuation-v1",
        }
    }
}

/// Literal input root of a special read-only evaluation build, independent of its source root.
pub(crate) fn compiled_input_root() -> Option<&'static str> {
    option_env!("TESSERA_DIAGNOSTIC_INPUT_ROOT")
}

/// Literal original producer executable used by a special read-only evaluation build.
pub(crate) fn compiled_producer_binary() -> Option<&'static str> {
    option_env!("TESSERA_DIAGNOSTIC_PRODUCER_BINARY")
}

pub(crate) fn validate_build_context(command: &crate::Command) -> anyhow::Result<()> {
    validate_context_command(compiled_input_root(), compiled_producer_binary(), command)
}

fn validate_context_paths(root: Option<&str>, producer: Option<&str>) -> anyhow::Result<()> {
    ensure!(
        root.is_some() == producer.is_some(),
        "diagnostic input root and producer binary must be compiled together"
    );
    if let (Some(root), Some(producer)) = (root, producer) {
        ensure!(
            !root.is_empty()
                && std::path::Path::new(root).is_absolute()
                && std::path::Path::new(root).is_dir(),
            "compiled diagnostic input root must be an existing absolute directory"
        );
        ensure!(
            !producer.is_empty()
                && std::path::Path::new(producer).is_absolute()
                && std::path::Path::new(producer).is_file(),
            "compiled diagnostic producer binary must be an existing absolute file"
        );
    }
    Ok(())
}

fn validate_context_command(
    root: Option<&str>,
    producer: Option<&str>,
    command: &crate::Command,
) -> anyhow::Result<()> {
    validate_context_paths(root, producer)?;
    if root.is_none() {
        return Ok(());
    }
    let crate::Command::Eval(args) = command else {
        anyhow::bail!(
            "special diagnostic input-root build permits only explicit ADDRESS continuation evaluation"
        );
    };
    ensure!(
        args.diagnostic_decoder == Some(Decoder::AddressContinuationV1),
        "special diagnostic input-root build requires --diagnostic-decoder address-continuation-v1"
    );
    validate_route(args)
}

fn producer_binary_from_context(
    root: Option<&str>,
    producer: Option<&str>,
) -> anyhow::Result<Option<crate::fullmix_rms::Receipt>> {
    validate_context_paths(root, producer)?;
    producer
        .map(|path| {
            Ok(crate::fullmix_rms::Receipt {
                path: path.into(),
                sha256: crate::export::sha256_hex(&std::fs::read(path)?),
            })
        })
        .transpose()
}

/// Re-reads the original producer bytes for each special evaluation request.
pub(crate) fn producer_binary() -> anyhow::Result<Option<crate::fullmix_rms::Receipt>> {
    producer_binary_from_context(compiled_input_root(), compiled_producer_binary())
}

/// Matches the proof to the producer while retaining current-executable binding by default.
pub(crate) fn validate_preflight_binary(
    producer: Option<&crate::fullmix_rms::Receipt>,
    expected: &Value,
) -> anyhow::Result<bool> {
    let actual = match producer {
        Some(receipt) => receipt.sha256.clone(),
        None => crate::export::sha256_hex(&std::fs::read(std::env::current_exe()?)?),
    };
    Ok(expected.as_str() == Some(actual.as_str()))
}

/// Adds the original producer identity alongside the evaluator identity in result provenance.
pub(crate) fn record_producer(
    producer: Option<&crate::fullmix_rms::Receipt>,
    provenance: &mut Value,
) -> anyhow::Result<()> {
    if let Some(producer) = producer {
        provenance["producer_binary"] = serde_json::to_value(producer)?;
    }
    Ok(())
}

pub(crate) fn validate_route(args: &crate::EvalArgs) -> anyhow::Result<()> {
    ensure!(
        args.diagnostic_decoder.is_none()
            || (args.run.is_some()
                && args.gold.is_some()
                && !args.baseline
                && args.grouper.is_none()
                && args.addresses.is_none()
                && args.report.is_none()
                && args.errors.is_none()
                && args.country_hint_mode == crate::detect_eval::CountryHintMode::Known
                && matches!(args.backend, crate::train::BackendKind::Ndarray)),
        "diagnostic decoder requires CPU model-only --run --gold evaluation"
    );
    Ok(())
}

pub(crate) fn validate_seen(decoder: Option<Decoder>, role: Option<&str>) -> anyhow::Result<()> {
    ensure!(
        decoder.is_none() || matches!(role, Some("authored" | "real")),
        "diagnostic decoder requires pinned final4000 authored or real TRAINING-SEEN gold"
    );
    Ok(())
}

pub(crate) fn decode(
    decoder: Option<Decoder>,
    text: &str,
    token_spans: &[(u32, u32)],
    probs: &[f32],
    masked: &[bool],
    breaks: &[bool],
) -> Vec<DetectedSpan> {
    match decoder {
        None => decode_detector(probs, masked, breaks),
        Some(Decoder::AddressContinuationV1) => {
            let bounds: Vec<_> = token_spans
                .iter()
                .map(|&(start, end)| (start as usize, end as usize))
                .collect();
            decode_detector_with_text(text, &bounds, probs, masked, breaks)
        }
    }
}

pub(crate) fn record(decoder: Option<Decoder>, mut value: Value) -> Value {
    if let Some(decoder) = decoder {
        value["diagnostic_decoder"] = json!(decoder.identifier());
        if let Some(root) = compiled_input_root() {
            value["diagnostic_input_root"] = json!(root);
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tessera::internal::{is_content, paragraph_breaks, tokenize};

    fn parse(extra: &[&str]) -> anyhow::Result<crate::EvalArgs> {
        let mut argv = vec!["trainer", "eval"];
        argv.extend_from_slice(extra);
        match crate::Cli::try_parse_from(argv)?.command {
            crate::Command::Eval(args) => Ok(*args),
            _ => unreachable!(),
        }
    }

    #[test]
    fn diagnostic_decode_cli_is_explicit_and_refuses_other_routes() {
        let old = parse(&[
            "--run",
            "snapshot",
            "--gold",
            "gold.jsonl",
            "--backend",
            "ndarray",
        ])
        .unwrap();
        assert_eq!(old.diagnostic_decoder, None);
        validate_route(&old).unwrap();
        let option = ["--diagnostic-decoder", "address-continuation-v1"];
        let mut good = vec![
            "--run",
            "snapshot",
            "--gold",
            "gold.jsonl",
            "--backend",
            "ndarray",
        ];
        good.extend(option);
        let selected = parse(&good).unwrap();
        assert_eq!(
            selected.diagnostic_decoder,
            Some(Decoder::AddressContinuationV1)
        );
        validate_route(&selected).unwrap();
        for wrong in [
            vec!["--diagnostic-decoder", "address-continuation-v1"],
            vec![
                "--run",
                "snapshot",
                "--diagnostic-decoder",
                "address-continuation-v1",
            ],
            vec![
                "--gold",
                "gold.jsonl",
                "--diagnostic-decoder",
                "address-continuation-v1",
            ],
        ] {
            assert!(parse(&wrong).is_err());
        }
        for extra in [
            vec!["--addresses", "fixtures"],
            vec!["--grouper", "fixtures"],
            vec!["--baseline"],
            vec!["--report", "report.md"],
            vec!["--errors", "1"],
        ] {
            let mut wrong = good.clone();
            wrong.extend(extra);
            assert!(parse(&wrong).is_err());
        }
        let mut gpu = good.clone();
        gpu[5] = "wgpu";
        assert!(validate_route(&parse(&gpu).unwrap()).is_err());
        let mut auto = good.clone();
        auto.extend(["--country-hint-mode", "auto"]);
        assert!(validate_route(&parse(&auto).unwrap()).is_err());
        for role in [None, Some("development"), Some("reused196")] {
            assert!(validate_seen(selected.diagnostic_decoder, role).is_err());
            validate_seen(None, role).unwrap();
        }
        for role in ["authored", "real"] {
            validate_seen(selected.diagnostic_decoder, Some(role)).unwrap();
        }
    }

    #[test]
    fn diagnostic_decode_json_absence_preserves_legacy_bytes() {
        let old = json!({"provenance":{"checkpoint":"unchanged"},"scores":{"address":0.5}});
        assert_eq!(
            serde_json::to_vec(&record(None, old.clone())).unwrap(),
            serde_json::to_vec(&old).unwrap()
        );
        let selected = record(Some(Decoder::AddressContinuationV1), old.clone());
        assert_eq!(selected["diagnostic_decoder"], "address-continuation-v1");
        assert_eq!(selected["provenance"], old["provenance"]);
        assert_eq!(selected["scores"], old["scores"]);
    }

    #[test]
    fn diagnostic_decode_fake_rows_use_shared_helper_before_unchanged_policy() {
        let text = "Office delivery location\nNorth Building\n123 Main Street\nDenver, CO 80202";
        let tokens = tokenize(text);
        let retained: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| is_content(t).then_some(i))
            .collect();
        let bounds: Vec<_> = retained
            .iter()
            .map(|&i| (tokens[i].start as u32, tokens[i].end as u32))
            .collect();
        let breaks = paragraph_breaks(&tokens, &retained);
        let left = text.find("North Building").unwrap();
        let right = text.find("123 Main Street").unwrap();
        let mut probs = vec![0.001; bounds.len() * 7];
        for (i, &(start, _)) in bounds.iter().enumerate() {
            let start = start as usize;
            let label = if start < left {
                0
            } else if start == left || start == right {
                5
            } else {
                6
            };
            probs[i * 7 + label] = if start < left {
                0.99
            } else if start < right {
                0.61
            } else {
                0.98
            };
        }
        let masked = vec![false; bounds.len()];
        let old = decode(None, text, &bounds, &probs, &masked, &breaks);
        assert_eq!(old, decode_detector(&probs, &masked, &breaks));
        assert_eq!(old.len(), 2);
        let selected = decode(
            Some(Decoder::AddressContinuationV1),
            text,
            &bounds,
            &probs,
            &masked,
            &breaks,
        );
        assert_eq!(selected.len(), 1);
        let native_bounds: Vec<_> = bounds
            .iter()
            .map(|&(s, e)| (s as usize, e as usize))
            .collect();
        assert_eq!(
            selected,
            decode_detector_with_text(text, &native_bounds, &probs, &masked, &breaks)
        );
        let scored = |spans: Vec<DetectedSpan>| {
            vec![
                spans
                    .into_iter()
                    .map(|s| crate::detector::ScoredSpan {
                        span: crate::detector::KindSpan {
                            kind: 2,
                            start: bounds[s.first].0,
                            end: bounds[s.last].1,
                        },
                        confidence: s.confidence,
                    })
                    .collect(),
            ]
        };
        assert_eq!(
            crate::detector::apply_confidence_policy(&scored(old))[0].len(),
            1
        );
        assert_eq!(
            crate::detector::apply_confidence_policy(&scored(selected))[0].len(),
            1
        );
        let mut gap = probs.clone();
        let restart = retained
            .iter()
            .position(|&i| tokens[i].start == right)
            .unwrap();
        gap[restart * 7..(restart + 1) * 7].fill(0.001);
        gap[restart * 7] = 0.99;
        assert_eq!(
            decode(
                Some(Decoder::AddressContinuationV1),
                text,
                &bounds,
                &gap,
                &masked,
                &breaks
            ),
            decode(None, text, &bounds, &gap, &masked, &breaks)
        );
    }

    #[test]
    fn diagnostic_decode_special_build_denies_commands_and_default_context_is_unchanged() {
        let valid_root = env!("CARGO_MANIFEST_DIR");
        let producer = std::env::current_exe().unwrap();
        let producer = producer.to_str().unwrap();
        let commands = [
            vec!["trainer", "train", "--config", "unused"],
            vec!["trainer", "prepare", "--config", "unused"],
            vec!["trainer", "generate", "--config", "unused"],
            vec!["trainer", "check-detector-data", "--config", "unused"],
            vec![
                "trainer",
                "diagnose-fullmix-rms",
                "--manifest",
                "unused",
                "--out",
                "unused",
            ],
            vec![
                "trainer",
                "eval",
                "--run",
                "snapshot",
                "--gold",
                "gold.jsonl",
                "--backend",
                "ndarray",
            ],
        ];
        for argv in commands {
            let command = crate::Cli::try_parse_from(argv).unwrap().command;
            assert!(validate_context_command(Some(valid_root), Some(producer), &command).is_err());
            validate_context_command(None, None, &command).unwrap();
        }
        let good = crate::Cli::try_parse_from([
            "trainer",
            "eval",
            "--run",
            "snapshot",
            "--gold",
            "gold.jsonl",
            "--backend",
            "ndarray",
            "--diagnostic-decoder",
            "address-continuation-v1",
        ])
        .unwrap()
        .command;
        validate_context_command(Some(valid_root), Some(producer), &good).unwrap();
        for root in ["", "relative", "/nonexistent-tessera-diagnostic-input-root"] {
            assert!(validate_context_command(Some(root), Some(producer), &good).is_err());
        }
        let gpu = crate::Cli::try_parse_from([
            "trainer",
            "eval",
            "--run",
            "snapshot",
            "--gold",
            "gold.jsonl",
            "--backend",
            "wgpu",
            "--diagnostic-decoder",
            "address-continuation-v1",
        ])
        .unwrap()
        .command;
        assert!(validate_context_command(Some(valid_root), Some(producer), &gpu).is_err());
        if let Some(root) = compiled_input_root() {
            assert_eq!(
                record(Some(Decoder::AddressContinuationV1), json!({}))["diagnostic_input_root"],
                root
            );
        } else {
            assert!(
                record(Some(Decoder::AddressContinuationV1), json!({}))
                    .get("diagnostic_input_root")
                    .is_none()
            );
        }
        assert_eq!(record(None, json!({})), json!({}));
    }

    #[test]
    fn diagnostic_decode_producer_context_is_paired_and_rejects_invalid_paths() {
        let root = env!("CARGO_MANIFEST_DIR");
        let producer = std::env::current_exe().unwrap();
        let producer = producer.to_str().unwrap();
        validate_context_paths(None, None).unwrap();
        validate_context_paths(Some(root), Some(producer)).unwrap();
        assert!(validate_context_paths(Some(root), None).is_err());
        assert!(validate_context_paths(None, Some(producer)).is_err());
        for path in ["", "relative", "/nonexistent-tessera-producer", root] {
            assert!(validate_context_paths(Some(root), Some(path)).is_err());
        }
        assert!(producer_binary_from_context(None, None).unwrap().is_none());
    }

    #[test]
    fn diagnostic_decode_producer_binding_rehashes_and_preserves_default_executable_identity() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("producer");
        std::fs::write(&file, b"original producer").unwrap();
        let root = dir.path().to_str().unwrap();
        let path = file.to_str().unwrap();
        let original = producer_binary_from_context(Some(root), Some(path))
            .unwrap()
            .unwrap();
        let expected = json!(original.sha256);
        assert!(validate_preflight_binary(Some(&original), &expected).unwrap());
        std::fs::write(&file, b"changed producer").unwrap();
        let changed = producer_binary_from_context(Some(root), Some(path))
            .unwrap()
            .unwrap();
        assert!(!validate_preflight_binary(Some(&changed), &expected).unwrap());
        let current =
            crate::export::sha256_hex(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
        assert!(validate_preflight_binary(None, &json!(current)).unwrap());
        assert!(!validate_preflight_binary(None, &expected).unwrap());
        std::fs::remove_file(&file).unwrap();
        assert!(producer_binary_from_context(Some(root), Some(path)).is_err());
    }

    #[test]
    fn diagnostic_decode_producer_provenance_preserves_distinct_evaluator_identity() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("producer");
        std::fs::write(&file, b"original producer").unwrap();
        let receipt = producer_binary_from_context(
            Some(dir.path().to_str().unwrap()),
            Some(file.to_str().unwrap()),
        )
        .unwrap()
        .unwrap();
        let mut provenance = json!({"evaluator_sha256": "new evaluator identity"});
        record_producer(Some(&receipt), &mut provenance).unwrap();
        let mut default = json!({"evaluator_sha256":"unchanged"});
        let old = default.clone();
        record_producer(None, &mut default).unwrap();
        assert_eq!(default, old);
        for kind in ["review", "raw", "predictions"] {
            let result = record(
                Some(Decoder::AddressContinuationV1),
                json!({"kind":kind,"provenance":provenance}),
            );
            assert_eq!(
                result["provenance"]["producer_binary"]["path"],
                file.to_str().unwrap()
            );
            assert_eq!(
                result["provenance"]["producer_binary"]["sha256"],
                receipt.sha256
            );
            assert_eq!(
                result["provenance"]["evaluator_sha256"],
                "new evaluator identity"
            );
        }
    }
}
