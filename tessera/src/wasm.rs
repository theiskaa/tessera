//! Feature `wasm`: raw `wasm-bindgen` exports with UTF-16 offsets and
//! hand-built JS objects. The npm package's async wrapper, worker, and bundle
//! fetching sit on top of this module in JavaScript and are not here.

use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

use crate::{Config, Entity, Error, Kind, KindSet, Query, Tessera, token};

/// The JavaScript `Tessera` class: a loaded extractor reporting UTF-16 offsets.
#[wasm_bindgen(js_name = Tessera)]
pub struct JsTessera {
    inner: Tessera,
}

#[wasm_bindgen(js_class = Tessera)]
impl JsTessera {
    /// `new Tessera(bundle, kinds, expectedChecksum?)`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        bundle: &[u8],
        kinds: Vec<String>,
        expected_checksum: Option<String>,
    ) -> Result<JsTessera, JsValue> {
        let mut set = KindSet::EMPTY;
        for k in &kinds {
            match Kind::from_str_label(k) {
                Some(kind) => set = set | kind,
                None => {
                    return Err(js_error(
                        "INVALID_KIND",
                        &format!("unknown kind `{k}`"),
                        None,
                    ));
                }
            }
        }
        let inner = Tessera::load(
            bundle,
            Config {
                kinds: set,
                expected_checksum: expected_checksum.as_deref(),
            },
        )
        .map_err(to_js)?;
        Ok(JsTessera { inner })
    }

    /// Entities with UTF-16 offsets and a `text` field.
    pub fn detect(
        &self,
        text: &str,
        country_hint: Vec<String>,
        include_uncertain: bool,
    ) -> Result<JsValue, JsValue> {
        let hints: Vec<&str> = country_hint.iter().map(String::as_str).collect();
        let entities = self
            .inner
            .detect(
                text,
                &Query {
                    country_hint: &hints,
                    include_uncertain,
                },
            )
            .map_err(to_js)?;
        // One pass over the text for every offset; a table for the whole text would cost four
        // bytes per input byte, and wasm memory does not shrink.
        let offsets = entities.iter().flat_map(|e| {
            [e.start, e.end]
                .into_iter()
                .chain(e.components.iter().flat_map(|c| [c.start, c.end]))
        });
        let units = token::utf16_at(text, offsets);
        let arr = Array::new();
        let mut at = 0;
        for e in &entities {
            let n = 2 + 2 * e.components.len();
            arr.push(&entity_to_js(
                e,
                text,
                units.get(at..at + n).unwrap_or_default(),
            ));
            at += n;
        }
        Ok(arr.into())
    }

    /// The kinds this instance was loaded for, as strings.
    pub fn kinds(&self) -> Vec<String> {
        Kind::ALL
            .iter()
            .filter(|k| self.inner.kinds().contains(**k))
            .map(|k| k.as_str().to_string())
            .collect()
    }
}

fn set(obj: &Object, key: &str, val: JsValue) {
    // Reflect::set only fails on non-objects; `obj` is always an Object here.
    let _ = Reflect::set(obj, &JsValue::from_str(key), &val);
}

/// `units` holds the UTF-16 offsets of `e.start`, `e.end`, then each component's start and end.
fn entity_to_js(e: &Entity, text: &str, units: &[u32]) -> JsValue {
    let u16 = |k: usize| units.get(k).copied().unwrap_or(0);
    let obj = Object::new();
    set(&obj, "kind", e.kind.as_str().into());
    set(&obj, "text", e.text(text).into());
    set(&obj, "start", u16(0).into());
    set(&obj, "end", u16(1).into());
    set(
        &obj,
        "confidence",
        crate::policy::display_confidence(e.confidence).into(),
    );
    set(&obj, "source", e.source.as_str().into());
    set(&obj, "reviewRecommended", e.review_recommended.into());
    if let Some(n) = &e.normalized {
        set(&obj, "normalized", n.as_str().into());
    }
    if let Some(r) = &e.region {
        set(&obj, "region", r.as_str().into());
    }
    if !e.components.is_empty() {
        let comps = Array::new();
        for (k, c) in e.components.iter().enumerate() {
            let co = Object::new();
            set(&co, "label", c.label.as_str().into());
            set(&co, "text", c.text(text).into());
            set(&co, "start", u16(2 + 2 * k).into());
            set(&co, "end", u16(3 + 2 * k).into());
            set(
                &co,
                "confidence",
                crate::policy::display_confidence(c.confidence).into(),
            );
            comps.push(&co);
        }
        set(&obj, "components", comps.into());
    }
    obj.into()
}

fn js_error(code: &str, message: &str, stage: Option<&str>) -> JsValue {
    let err = js_sys::Error::new(message);
    set(&err, "code", code.into());
    if let Some(s) = stage {
        set(&err, "stage", s.into());
    }
    err.into()
}

fn to_js(e: Error) -> JsValue {
    let (code, stage) = match &e {
        Error::BundleInvalid => ("BUNDLE_INVALID", None),
        Error::ChecksumMismatch => ("CHECKSUM_MISMATCH", None),
        Error::UnsupportedVersion => ("UNSUPPORTED_VERSION", None),
        Error::InputTooLarge => ("INPUT_TOO_LARGE", None),
        Error::Inference { stage } => ("INFERENCE", Some(*stage)),
    };
    js_error(code, &e.to_string(), stage)
}
