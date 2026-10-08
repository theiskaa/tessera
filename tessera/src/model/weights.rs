//! Safetensors bundle reader: checksum, header, manifest, version check, and typed tensors.
//!
//! The bundle is the one untrusted input the library parses, so every offset read from the
//! header is range-checked before use and every failure is a typed error. The order is fixed:
//! checksum over the whole input first, when one is expected, then header, version, layout.
//! Tensors must tile the data section exactly, as the safetensors format requires, so a file
//! cannot make the library allocate more than its own size by pointing many tensors at the
//! same bytes. Bytes are copied only when a network takes a tensor.

use super::json::Json;
use crate::Error;
use crate::features::FeatureConfig;

/// A bundle has a few dozen tensors; more is not a bundle this library wrote.
const MAX_TENSORS: usize = 256;
/// The header is parsed in full before it is checked, so its length is bounded first. The
/// shipped header is about 3 KB.
const MAX_HEADER: usize = 64 * 1024;
/// N-gram sizes a feature configuration may use.
const NGRAM_SIZES: core::ops::RangeInclusive<u8> = 1..=8;

/// An int8 tensor with one f32 scale per channel, in the file's row-major layout.
#[derive(Debug)]
pub(crate) struct QTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
}

/// What the bundle says about itself, from the safetensors metadata header. Every key is
/// required at load; the ones not read yet are for the detector and the site.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct Manifest {
    pub format: String,
    pub model_version: String,
    pub nets: Vec<String>,
    pub detector_labels: Vec<String>,
    pub parser_labels: Vec<String>,
    pub feature_config: FeatureConfig,
    pub max_ngrams_per_token: usize,
    pub flag_bits: usize,
    pub detector_feature_contract: crate::detector_features::DetectorFeatureContract,
    pub detector_postprocess_contract:
        Option<crate::detector_postprocess::DetectorPostprocessContract>,
    pub script_rows: usize,
    pub shape_rows: usize,
    pub phone_metadata_version: String,
    pub supported_regions: Vec<String>,
    pub experimental_regions: Vec<String>,
    pub training_snapshot: String,
    pub report_url: String,
    pub context96_rms: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    I8,
    F32,
}

/// One header entry: dtype, shape, and its byte range in the data section.
#[derive(Debug)]
struct Entry {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// A parsed bundle borrowing the input bytes; networks copy out the tensors they use.
#[derive(Debug)]
pub(crate) struct Bundle<'a> {
    pub manifest: Manifest,
    data: &'a [u8],
    entries: Vec<Entry>,
}

impl<'a> Bundle<'a> {
    pub(crate) fn parse(bytes: &'a [u8], expected_checksum: Option<&str>) -> Result<Self, Error> {
        if let Some(expected) = expected_checksum {
            let actual = format!(
                "sha256-{}",
                super::sha256::hex(&super::sha256::digest(bytes))
            );
            if !actual.eq_ignore_ascii_case(expected.trim()) {
                return Err(Error::ChecksumMismatch);
            }
        }
        let n = bytes
            .get(0..8)
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .map(u64::from_le_bytes)
            .ok_or(Error::BundleInvalid)?;
        let n = usize::try_from(n)
            .ok()
            .filter(|&n| n <= MAX_HEADER)
            .ok_or(Error::BundleInvalid)?;
        let header_end = 8usize.checked_add(n).ok_or(Error::BundleInvalid)?;
        let header = bytes.get(8..header_end).ok_or(Error::BundleInvalid)?;
        let header = core::str::from_utf8(header).map_err(|_| Error::BundleInvalid)?;
        let json = Json::parse(header)?;
        let data = bytes.get(header_end..).ok_or(Error::BundleInvalid)?;
        let meta = json.get("__metadata__").ok_or(Error::BundleInvalid)?;
        check_version(meta)?;
        check_contracts(meta)?;
        let manifest = Manifest::from_json(meta)?;
        manifest.check_supported()?;
        let entries = entries(&json, data.len())?;
        Ok(Bundle {
            manifest,
            data,
            entries,
        })
    }

    pub(crate) fn has_net(&self, net: &str) -> bool {
        self.manifest.nets.iter().any(|n| n == net)
    }

    pub(crate) fn check_block_count(&self, net: &str, count: usize) -> Result<(), Error> {
        let prefix = format!("{net}.block");
        for entry in &self.entries {
            if let Some(suffix) = entry.name.strip_prefix(&prefix) {
                let index = suffix
                    .split('.')
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
                    .ok_or(Error::BundleInvalid)?;
                if index >= count {
                    return Err(Error::BundleInvalid);
                }
            }
        }
        Ok(())
    }

    /// Whether the bundle holds a tensor named `name`.
    pub(crate) fn has(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// The length of the rank-1 f32 tensor `name`, if the bundle holds one.
    pub(crate) fn f32_len(&self, name: &str) -> Option<usize> {
        self.entries
            .iter()
            .find(|e| e.name == name && e.dtype == Dtype::F32)
            .and_then(|e| match e.shape.as_slice() {
                [len] => Some(*len),
                _ => None,
            })
    }

    fn entry(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&Entry, Error> {
        self.entries
            .iter()
            .find(|e| e.name == name && e.dtype == dtype && e.shape == shape)
            .ok_or(Error::BundleInvalid)
    }

    /// The int8 tensor `name`, which must have exactly `shape`, with its per-channel scales
    /// from `<name>.scale`, one per entry along `channel_axis`.
    pub(crate) fn take_i8(
        &self,
        name: &str,
        shape: &[usize],
        channel_axis: usize,
    ) -> Result<QTensor, Error> {
        let e = self.entry(name, Dtype::I8, shape)?;
        let channels = *shape.get(channel_axis).ok_or(Error::BundleInvalid)?;
        let scales = self.take_f32(&format!("{name}.scale"), channels)?;
        let data = self.data[e.begin..e.end].iter().map(|&b| b as i8).collect();
        Ok(QTensor {
            shape: shape.to_vec(),
            data,
            scales,
        })
    }

    /// The rank-1 f32 tensor `name` of length `len`.
    pub(crate) fn take_f32(&self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let e = self.entry(name, Dtype::F32, &[len])?;
        Ok(self.data[e.begin..e.end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

/// Header entries sorted by offset, checked to tile `0..data_len` exactly.
fn entries(json: &Json, data_len: usize) -> Result<Vec<Entry>, Error> {
    let mut out = Vec::new();
    for (name, info) in json.entries() {
        if name == "__metadata__" {
            continue;
        }
        if out.len() == MAX_TENSORS {
            return Err(Error::BundleInvalid);
        }
        let dtype = match info.get("dtype").and_then(Json::as_str) {
            Some("I8") => Dtype::I8,
            Some("F32") => Dtype::F32,
            _ => return Err(Error::BundleInvalid),
        };
        let shape = info
            .get("shape")
            .and_then(Json::as_arr)
            .ok_or(Error::BundleInvalid)?
            .iter()
            .map(|d| d.as_usize().ok_or(Error::BundleInvalid))
            .collect::<Result<Vec<_>, _>>()?;
        let [begin, end] = info
            .get("data_offsets")
            .and_then(Json::as_arr)
            .ok_or(Error::BundleInvalid)?
        else {
            return Err(Error::BundleInvalid);
        };
        let (begin, end) = (
            begin.as_usize().ok_or(Error::BundleInvalid)?,
            end.as_usize().ok_or(Error::BundleInvalid)?,
        );
        let size = match dtype {
            Dtype::I8 => 1,
            Dtype::F32 => 4,
        };
        let bytes = shape
            .iter()
            .try_fold(size, |acc: usize, &d| acc.checked_mul(d))
            .ok_or(Error::BundleInvalid)?;
        if end.checked_sub(begin) != Some(bytes) {
            return Err(Error::BundleInvalid);
        }
        out.push(Entry {
            name: name.clone(),
            dtype,
            shape,
            begin,
            end,
        });
    }
    out.sort_by_key(|e| (e.begin, e.end));
    let mut at = 0;
    for e in &out {
        if e.begin != at {
            return Err(Error::BundleInvalid);
        }
        at = e.end;
    }
    if at != data_len {
        return Err(Error::BundleInvalid);
    }
    Ok(out)
}

/// Format and model version series, read before the rest of the manifest: a newer bundle may
/// change any other key, and must still be reported as newer rather than as malformed. The
/// series (`major.minor`) is read first, so `0.3.0-rc1` is unsupported; a version in the
/// supported series must then be a plain `major.minor.patch`, so `0.2` or `0.2.x` is invalid.
/// Format 2 covers two series: 0.3, the first Context96 detector, and 0.4, the reviewed-fit
/// detector with tab-cell features and labeled-field postprocessing; both share the format 2
/// graph and every other manifest key.
fn check_version(meta: &Json) -> Result<(), Error> {
    let series: &[(u32, u32)] = match text(meta, "format")?.as_str() {
        super::SUPPORTED_FORMAT => &[super::SUPPORTED_MODEL_SERIES],
        "2" => &[(0, 3), (0, 4)],
        _ => return Err(Error::UnsupportedVersion),
    };
    let version = text(meta, "model_version")?;
    let parts: Vec<&str> = version.split('.').collect();
    let plain = |p: &str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
    let number = |p: Option<&&str>| {
        p.filter(|p| plain(p))
            .and_then(|p| p.parse::<u32>().ok())
            .ok_or(Error::BundleInvalid)
    };
    let (major, minor) = (number(parts.first())?, number(parts.get(1))?);
    let compatible = series.iter().any(|&(want_major, want_minor)| {
        if want_major == 0 {
            major == 0 && minor == want_minor
        } else {
            major == want_major
        }
    });
    if !compatible {
        return Err(Error::UnsupportedVersion);
    }
    match parts[..] {
        [_, _, patch] if plain(patch) => Ok(()),
        _ => Err(Error::BundleInvalid),
    }
}

fn check_contracts(meta: &Json) -> Result<(), Error> {
    let tokenizer = meta.get("tokenizer_contract");
    let decoder = meta.get("decoder_contract");
    match (tokenizer, decoder) {
        (None, None) if text(meta, "model_version")? == "0.2.0" => Ok(()),
        (None, None) => Err(Error::UnsupportedVersion),
        (Some(tokenizer), Some(decoder)) => {
            let tokenizer = tokenizer.as_str().ok_or(Error::BundleInvalid)?;
            let decoder = decoder.as_str().ok_or(Error::BundleInvalid)?;
            let expected_decoder = match text(meta, "format")?.as_str() {
                "1" => super::DECODER_CONTRACT,
                "2" => super::context::CONTEXT96_DECODER_CONTRACT,
                _ => return Err(Error::UnsupportedVersion),
            };
            if tokenizer == super::TOKENIZER_CONTRACT && decoder == expected_decoder {
                Ok(())
            } else {
                Err(Error::UnsupportedVersion)
            }
        }
        _ => Err(Error::BundleInvalid),
    }
}

fn text(m: &Json, key: &str) -> Result<String, Error> {
    m.get(key)
        .and_then(Json::as_str)
        .map(str::to_string)
        .ok_or(Error::BundleInvalid)
}

/// Metadata values are strings; structured ones hold JSON and are parsed a second time.
fn nested(m: &Json, key: &str) -> Result<Json, Error> {
    Json::parse(&text(m, key)?)
}

fn strings(j: &Json) -> Result<Vec<String>, Error> {
    j.as_arr()
        .ok_or(Error::BundleInvalid)?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or(Error::BundleInvalid))
        .collect()
}

impl Manifest {
    fn from_json(m: &Json) -> Result<Manifest, Error> {
        if m.get("detector_architecture")
            .is_some_and(|v| v.as_str().is_none())
        {
            return Err(Error::BundleInvalid);
        }
        let fc = nested(m, "feature_config")?;
        let num = |key: &str| {
            fc.get(key)
                .and_then(Json::as_usize)
                .ok_or(Error::BundleInvalid)
        };
        let ngram_sizes = fc
            .get("ngram_sizes")
            .and_then(Json::as_arr)
            .ok_or(Error::BundleInvalid)?
            .iter()
            .map(|v| {
                v.as_usize()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or(Error::BundleInvalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let detector_feature_contract = if m.get("detector_feature_config").is_some() {
            let df = nested(m, "detector_feature_config")?;
            let name = df
                .get("contract")
                .and_then(Json::as_str)
                .ok_or(Error::BundleInvalid)?;
            let width = df
                .get("flag_bits")
                .and_then(Json::as_usize)
                .ok_or(Error::BundleInvalid)?;
            crate::detector_features::DetectorFeatureContract::from_name_and_width(name, width)
                .ok_or(Error::UnsupportedVersion)?
        } else {
            crate::detector_features::DetectorFeatureContract::Legacy23
        };
        let detector_postprocess_contract = match m.get("detector_postprocess_contract") {
            None => None,
            Some(value) => Some(
                crate::detector_postprocess::DetectorPostprocessContract::from_name(
                    value.as_str().ok_or(Error::BundleInvalid)?,
                )
                .ok_or(Error::UnsupportedVersion)?,
            ),
        };
        Ok(Manifest {
            format: text(m, "format")?,
            model_version: text(m, "model_version")?,
            nets: text(m, "nets")?
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            detector_labels: strings(&nested(m, "detector_labels")?)?,
            parser_labels: strings(&nested(m, "parser_labels")?)?,
            feature_config: FeatureConfig {
                ngram_sizes,
                hash_buckets: u32::try_from(num("hash_buckets")?)
                    .map_err(|_| Error::BundleInvalid)?,
                hash_seed: fc
                    .get("hash_seed")
                    .and_then(Json::as_u64)
                    .ok_or(Error::BundleInvalid)?,
            },
            max_ngrams_per_token: num("max_ngrams_per_token")?,
            flag_bits: num("flag_bits")?,
            detector_feature_contract,
            detector_postprocess_contract,
            script_rows: num("script_rows")?,
            shape_rows: num("shape_rows")?,
            phone_metadata_version: text(m, "phone_metadata_version")?,
            supported_regions: strings(&nested(m, "supported_regions")?)?,
            experimental_regions: strings(&nested(m, "experimental_regions")?)?,
            training_snapshot: text(m, "training_snapshot")?,
            report_url: text(m, "report_url")?,
            context96_rms: super::context::from_manifest(
                &text(m, "format")?,
                m.get("detector_architecture").and_then(Json::as_str),
            )?,
        })
    }

    /// The layout constants this library was built with, and sane feature settings.
    fn check_supported(&self) -> Result<(), Error> {
        // Detector settings without the detector network are malformed; postprocessing on a
        // graph other than Context96 is a newer bundle.
        if (self.context96_rms
            || self.detector_postprocess_contract.is_some()
            || self.detector_feature_contract
                != crate::detector_features::DetectorFeatureContract::Legacy23)
            && !self.nets.iter().any(|net| net == "detector")
        {
            return Err(Error::BundleInvalid);
        }
        if self.detector_postprocess_contract.is_some() && !self.context96_rms {
            return Err(Error::UnsupportedVersion);
        }
        let shapes_match = self.max_ngrams_per_token == crate::features::MAX_NGRAMS_PER_TOKEN
            && self.flag_bits == super::FLAG_BITS
            && self.script_rows == super::SCRIPT_ROWS
            && self.shape_rows == super::SHAPE_ROWS
            && self.parser_labels.len() == super::PARSER_LABELS;
        if !shapes_match {
            return Err(Error::UnsupportedVersion);
        }
        let sizes = &self.feature_config.ngram_sizes;
        let sizes_ok = !sizes.is_empty()
            && sizes.len() <= NGRAM_SIZES.len()
            && sizes.iter().all(|n| NGRAM_SIZES.contains(n))
            && sizes
                .iter()
                .enumerate()
                .all(|(i, n)| !sizes[..i].contains(n));
        if !sizes_ok || self.feature_config.hash_buckets == 0 {
            return Err(Error::BundleInvalid);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tab_contract_is_explicit_and_parser_flags_remain_legacy() {
        use crate::detector_features::DetectorFeatureContract;
        let meta = serde_json::json!({
            "format":"1","model_version":"0.2.1","nets":"parser,detector",
            "detector_labels":"[]","parser_labels":serde_json::to_string(&crate::model::bio::parser_label_strings()).unwrap(),
            "feature_config":serde_json::json!({"ngram_sizes":[2,3,4],"hash_buckets":32768,"hash_seed":0,"max_ngrams_per_token":64,"flag_bits":23,"script_rows":13,"shape_rows":64}).to_string(),
            "phone_metadata_version":"x","supported_regions":"[]","experimental_regions":"[]","training_snapshot":"x","report_url":""
        });
        let parse =
            |v: &serde_json::Value| Manifest::from_json(&Json::parse(&v.to_string()).unwrap());
        let old = parse(&meta).unwrap();
        old.check_supported().unwrap();
        assert_eq!(
            old.detector_feature_contract,
            DetectorFeatureContract::Legacy23
        );
        let mut tab = meta.clone();
        tab["detector_feature_config"]=serde_json::json!({"contract":DetectorFeatureContract::TabCells25.name(),"flag_bits":25}).to_string().into();
        let new = parse(&tab).unwrap();
        new.check_supported().unwrap();
        assert_eq!(new.flag_bits, 23);
        assert_eq!(new.detector_feature_contract.flag_bits(), 25);
        tab["detector_feature_config"]=serde_json::json!({"contract":DetectorFeatureContract::TabCells25.name(),"flag_bits":23}).to_string().into();
        assert!(parse(&tab).is_err());
        tab["detector_feature_config"] = serde_json::json!({"contract":"unknown","flag_bits":25})
            .to_string()
            .into();
        assert!(parse(&tab).is_err());
        tab = meta;
        tab["feature_config"]=serde_json::json!({"ngram_sizes":[2,3,4],"hash_buckets":32768,"hash_seed":0,"max_ngrams_per_token":64,"flag_bits":25,"script_rows":13,"shape_rows":64}).to_string().into();
        assert!(parse(&tab).unwrap().check_supported().is_err());
    }

    #[test]
    fn postprocess_metadata_is_closed_and_absent_keeps_existing_default() {
        let meta = serde_json::json!({"format":"2","model_version":"0.3.0","nets":"parser,detector",
            "detector_architecture":crate::model::context::CONTEXT96_RMS_CONTRACT,
            "detector_labels":"[]","parser_labels":serde_json::to_string(&crate::model::bio::parser_label_strings()).unwrap(),
            "feature_config":serde_json::json!({"ngram_sizes":[2,3,4],"hash_buckets":32768,"hash_seed":0,"max_ngrams_per_token":64,"flag_bits":23,"script_rows":13,"shape_rows":64}).to_string(),
            "phone_metadata_version":"x","supported_regions":"[]","experimental_regions":"[]","training_snapshot":"x","report_url":""});
        let parse =
            |v: &serde_json::Value| Manifest::from_json(&Json::parse(&v.to_string()).unwrap());
        let old = parse(&meta).unwrap();
        old.check_supported().unwrap();
        assert_eq!(old.detector_postprocess_contract, None);
        for name in ["address_continuation_v1", "address_labeled_fields_v1"] {
            let mut selected = meta.clone();
            selected["detector_postprocess_contract"] = serde_json::json!(name);
            let parsed = parse(&selected).unwrap();
            parsed.check_supported().unwrap();
            assert_eq!(parsed.detector_postprocess_contract.unwrap().name(), name);
            for bad in [
                serde_json::Value::Null,
                serde_json::json!(true),
                serde_json::json!(0),
                serde_json::json!({}),
                serde_json::json!("unknown"),
                serde_json::json!(" address_labeled_fields_v1"),
            ] {
                let mut malformed = meta.clone();
                malformed["detector_postprocess_contract"] = bad;
                assert!(parse(&malformed).is_err());
            }
            selected["format"] = serde_json::json!("1");
            selected
                .as_object_mut()
                .unwrap()
                .remove("detector_architecture");
            assert_eq!(
                parse(&selected).unwrap().check_supported(),
                Err(Error::UnsupportedVersion)
            );
            let mut parser_only = meta.clone();
            parser_only["nets"] = serde_json::json!("parser");
            parser_only["detector_postprocess_contract"] = serde_json::json!(name);
            assert_eq!(
                parse(&parser_only).unwrap().check_supported(),
                Err(Error::BundleInvalid)
            );
        }
    }

    use super::*;

    fn bundle_bytes() -> Vec<u8> {
        crate::model::testing::parser_bundle()
    }

    #[test]
    fn every_parser_tensor_has_its_shape() {
        let bytes = bundle_bytes();
        let b = Bundle::parse(&bytes, None).unwrap();
        let buckets = b.manifest.feature_config.hash_buckets as usize;
        let mut int8: Vec<(String, Vec<usize>, usize)> = vec![
            ("parser.embed.ngram".into(), vec![buckets + 1, 48], 1),
            ("parser.embed.script".into(), vec![13, 8], 1),
            ("parser.embed.shape".into(), vec![64, 8], 1),
            ("parser.proj.weight".into(), vec![96, 87], 0),
            ("parser.head.weight".into(), vec![23, 96], 0),
        ];
        let mut f32s: Vec<(String, usize)> = vec![
            ("parser.proj.bias".into(), 96),
            ("parser.head.bias".into(), 23),
        ];
        for i in 0..4 {
            int8.push((format!("parser.block{i}.conv.weight"), vec![96, 96, 3], 0));
            f32s.push((format!("parser.block{i}.conv.bias"), 96));
        }
        for (name, shape, axis) in &int8 {
            let t = b.take_i8(name, shape, *axis).unwrap();
            assert_eq!(t.scales.len(), shape[*axis], "{name}");
            assert_eq!(t.data.len(), shape.iter().product::<usize>(), "{name}");
        }
        for (name, len) in &f32s {
            assert_eq!(b.take_f32(name, *len).unwrap().len(), *len, "{name}");
        }
        assert_eq!(b.entries.len(), int8.len() * 2 + f32s.len());
    }

    #[test]
    fn a_wrong_shape_is_invalid() {
        let bytes = bundle_bytes();
        let b = Bundle::parse(&bytes, None).unwrap();
        assert!(b.take_i8("parser.head.weight", &[96, 23], 0).is_err());
        assert!(b.take_f32("parser.head.weight", 23).is_err());
        assert!(b.take_f32("parser.missing", 1).is_err());
    }
    #[test]
    fn context96_bundle_contract_rejects_wrong_series_decoder_and_block_count() {
        let mut meta = serde_json::json!({"format":"2","model_version":"0.3.0","tokenizer_contract":super::super::TOKENIZER_CONTRACT,"decoder_contract":super::super::context::CONTEXT96_DECODER_CONTRACT});
        let parse = |v: &serde_json::Value| Json::parse(&v.to_string()).unwrap();
        check_version(&parse(&meta)).unwrap();
        check_contracts(&parse(&meta)).unwrap();
        meta["decoder_contract"] = serde_json::json!(super::super::DECODER_CONTRACT);
        assert!(check_contracts(&parse(&meta)).is_err());
        meta["model_version"] = serde_json::json!("0.4.0");
        check_version(&parse(&meta)).unwrap();
        for unsupported in ["0.2.1", "0.5.0", "1.4.0"] {
            meta["model_version"] = serde_json::json!(unsupported);
            assert!(check_version(&parse(&meta)).is_err(), "{unsupported}");
        }
        let mut bundle = Bundle {
            manifest: Manifest {
                format: "1".into(),
                model_version: "0.2.1".into(),
                nets: vec!["detector".into()],
                detector_labels: vec![],
                parser_labels: vec![],
                feature_config: FeatureConfig::default(),
                max_ngrams_per_token: crate::features::MAX_NGRAMS_PER_TOKEN,
                flag_bits: super::super::FLAG_BITS,
                detector_feature_contract:
                    crate::detector_features::DetectorFeatureContract::Legacy23,
                detector_postprocess_contract: None,
                script_rows: super::super::SCRIPT_ROWS,
                shape_rows: super::super::SHAPE_ROWS,
                phone_metadata_version: String::new(),
                supported_regions: vec![],
                experimental_regions: vec![],
                training_snapshot: String::new(),
                report_url: String::new(),
                context96_rms: false,
            },
            data: &[],
            entries: vec![],
        };
        bundle.entries.push(Entry {
            name: "detector.block6.conv.weight".into(),
            dtype: Dtype::I8,
            shape: vec![96, 96, 3],
            begin: 0,
            end: 0,
        });
        assert!(bundle.check_block_count("detector", 6).is_err());
        bundle.check_block_count("detector", 7).unwrap();
        bundle.entries.last_mut().unwrap().name = "detector.block7.conv.weight".into();
        assert!(bundle.check_block_count("detector", 7).is_err());
    }
}
