//! Feature `wasm`: the JavaScript `Tessera` class. Offsets cross this boundary as UTF-16 code
//! units, result objects carry a `text` field, and every library failure is a JS `Error` named
//! `TesseraError` with a stable `code`. Calls return a `Promise`, so a worker-backed instance
//! can keep the shape of this in-thread one. Fetching the bundle and the worker are not here.

use js_sys::{Array, ArrayBuffer, Object, Promise, Reflect, TypeError, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::policy::display_confidence;
use crate::{Config, Entity, Error, Kind, KindSet, Query, Tessera, token};

/// The JavaScript `Tessera` class: a loaded extractor reporting UTF-16 offsets.
#[wasm_bindgen(js_name = Tessera)]
pub struct JsTessera {
    inner: Option<Tessera>,
}

#[wasm_bindgen(js_class = Tessera)]
impl JsTessera {
    /// `Tessera.load(bytes, options?)`: parse and verify a bundle already in memory.
    ///
    /// `bytes` is a `Uint8Array` or an `ArrayBuffer`. `options.kinds` is a non-empty `string[]`
    /// of kinds to serve, every kind when absent; a set of only `"email"` and `"phone"` needs no
    /// bundle, so `bytes` may be empty. `options.integrity` is a `sha256-` digest the bundle must
    /// match. Throws a `TypeError` for a malformed argument and a `TesseraError` when the bundle
    /// cannot be loaded.
    pub fn load(
        #[wasm_bindgen(unchecked_param_type = "Uint8Array | ArrayBuffer")] bytes: &JsValue,
        options: Option<JsValue>,
    ) -> Result<JsTessera, JsValue> {
        let options = options.unwrap_or_default();
        let bytes = bundle_bytes(bytes)?;
        let kinds = kinds_option(&options)?;
        let integrity = string_option(&options, "integrity")?;
        let inner = Tessera::load(
            &bytes,
            Config {
                kinds,
                expected_checksum: integrity.as_deref(),
            },
        )?;
        Ok(JsTessera { inner: Some(inner) })
    }

    /// The kinds this instance was loaded for, as labels in taxonomy order; empty once disposed.
    #[wasm_bindgen(getter)]
    pub fn kinds(&self) -> Array {
        let set = self.inner.as_ref().map_or(KindSet::EMPTY, Tessera::kinds);
        Kind::ALL
            .iter()
            .filter(|k| set.contains(**k))
            .map(|k| JsValue::from_str(k.as_str()))
            .collect()
    }

    /// `parseAddress(text, options?)`: split `text`, known to be one address, into components.
    ///
    /// Resolves to an address entity whose offsets are relative to `text`. Of `options`, only
    /// `includeUncertain` applies.
    #[wasm_bindgen(js_name = parseAddress)]
    pub fn parse_address(
        &self,
        #[wasm_bindgen(unchecked_param_type = "string")] text: &JsValue,
        options: Option<JsValue>,
    ) -> Promise {
        settle(self.local().and_then(|t| {
            let text = text_argument(text)?;
            let entity = with_query(&options.unwrap_or_default(), |q| t.parse_address(&text, q))?;
            Ok(entity_objects(&text, std::slice::from_ref(&entity)).get(0))
        }))
    }

    /// `detect(text, options?)`: every supported entity in `text`, in document order.
    ///
    /// `options.countryHint` is a `string[]` of regions for numbers written without a country
    /// code; `options.includeUncertain` also returns low-confidence entities.
    pub fn detect(
        &self,
        #[wasm_bindgen(unchecked_param_type = "string")] text: &JsValue,
        options: Option<JsValue>,
    ) -> Promise {
        settle(self.local().and_then(|t| {
            let text = text_argument(text)?;
            let entities = with_query(&options.unwrap_or_default(), |q| t.detect(&text, q))?;
            Ok(entity_objects(&text, &entities).into())
        }))
    }

    /// Release the model. Later calls reject with code `DISPOSED`; disposing twice is harmless.
    /// After `free()`, which releases the Rust object itself, every call throws synchronously,
    /// as with any wasm-bindgen class.
    pub fn dispose(&mut self) {
        self.inner = None;
    }

    fn local(&self) -> Result<&Tessera, JsValue> {
        self.inner.as_ref().ok_or_else(|| {
            tessera_error("DISPOSED", "this Tessera instance has been disposed", None)
        })
    }
}

fn settle(outcome: Result<JsValue, JsValue>) -> Promise {
    match outcome {
        Ok(value) => Promise::resolve(&value),
        Err(err) => Promise::reject(&err),
    }
}

fn with_query<T>(
    options: &JsValue,
    call: impl FnOnce(&Query<'_>) -> Result<T, Error>,
) -> Result<T, JsValue> {
    let hints = string_array_option(options, "countryHint")?.unwrap_or_default();
    let hints: Vec<&str> = hints.iter().map(String::as_str).collect();
    let query = Query {
        country_hint: &hints,
        include_uncertain: bool_option(options, "includeUncertain")?.unwrap_or(false),
    };
    call(&query).map_err(JsValue::from)
}

/// The entities as JS objects, with every offset converted in one pass over `text`.
fn entity_objects(text: &str, entities: &[Entity]) -> Array {
    // A byte-to-unit table for the whole text would cost four bytes per input byte, and wasm
    // memory never shrinks; only the offsets the results carry are converted.
    let bytes = entities.iter().flat_map(|e| {
        [e.start, e.end]
            .into_iter()
            .chain(e.components.iter().flat_map(|c| [c.start, c.end]))
    });
    // Consumed in the order gathered: an entity's start and end, then its components' pairs.
    let mut units = token::utf16_at(text, bytes).into_iter();
    let mut next = move || units.next().unwrap_or_default();
    entities
        .iter()
        .map(|e| entity_object(text, e, &mut next))
        .collect()
}

fn entity_object(text: &str, e: &Entity, next: &mut impl FnMut() -> u32) -> JsValue {
    let obj = Object::new();
    set(&obj, "kind", e.kind.as_str());
    set(&obj, "text", slice(text, e.start, e.end));
    set(&obj, "start", next());
    set(&obj, "end", next());
    set(&obj, "confidence", display_confidence(e.confidence));
    set(&obj, "source", e.source.as_str());
    set(&obj, "reviewRecommended", e.review_recommended);
    let components: Array = e
        .components
        .iter()
        .map(|c| {
            let co = Object::new();
            set(&co, "label", c.label.as_str());
            set(&co, "text", slice(text, c.start, c.end));
            set(&co, "start", next());
            set(&co, "end", next());
            set(&co, "confidence", display_confidence(c.confidence));
            JsValue::from(co)
        })
        .collect();
    // Absent rather than empty on other kinds, so `"components" in entity` is a kind check.
    if e.kind == Kind::Address {
        set(&obj, "components", components);
    }
    if let Some(normalized) = &e.normalized {
        set(&obj, "normalized", normalized.as_str());
    }
    if let Some(region) = &e.region {
        set(&obj, "region", region.as_str());
    }
    obj.into()
}

fn slice(text: &str, start: usize, end: usize) -> &str {
    text.get(start..end).unwrap_or_default()
}

fn set(obj: &Object, key: &str, value: impl Into<JsValue>) {
    // Reflect.set fails only on frozen or exotic targets; every target here is a fresh object.
    let _ = Reflect::set(obj, &JsValue::from_str(key), &value.into());
}

/// A `TesseraError`: a JS `Error` with a `code` and, for `INFERENCE`, the failing `stage`.
fn tessera_error(code: &str, message: &str, stage: Option<&str>) -> JsValue {
    let err = js_sys::Error::new(message);
    err.set_name("TesseraError");
    set(&err, "code", code);
    if let Some(stage) = stage {
        set(&err, "stage", stage);
    }
    err.into()
}

/// The stable JavaScript code of each library error. Conditions that exist only in JavaScript,
/// such as `DISPOSED`, have codes but no `Error` variant and are raised where they arise.
fn error_code(err: &Error) -> &'static str {
    match err {
        Error::BundleInvalid => "BUNDLE_INVALID",
        Error::ChecksumMismatch => "CHECKSUM_MISMATCH",
        Error::UnsupportedVersion => "UNSUPPORTED_VERSION",
        Error::InputTooLarge => "INPUT_TOO_LARGE",
        Error::Inference { .. } => "INFERENCE",
    }
}

impl From<Error> for JsValue {
    /// The error as a `TesseraError`, carrying its code and, for `INFERENCE`, its stage.
    fn from(err: Error) -> JsValue {
        let stage = match &err {
            Error::Inference { stage } => Some(*stage),
            _ => None,
        };
        tessera_error(error_code(&err), &err.to_string(), stage)
    }
}

/// `options[key]`, with `undefined` and `null` both meaning absent.
fn option(options: &JsValue, key: &str) -> Result<Option<JsValue>, JsValue> {
    if options.is_undefined() || options.is_null() {
        return Ok(None);
    }
    if !options.is_object() {
        return Err(TypeError::new("options must be an object").into());
    }
    let value = Reflect::get(options, &JsValue::from_str(key))?;
    Ok((!value.is_undefined() && !value.is_null()).then_some(value))
}

fn type_error(key: &str, expected: &str) -> JsValue {
    TypeError::new(&format!("option `{key}` must be {expected}")).into()
}

fn string_option(options: &JsValue, key: &str) -> Result<Option<String>, JsValue> {
    option(options, key)?
        .map(|v| v.as_string().ok_or_else(|| type_error(key, "a string")))
        .transpose()
}

fn bool_option(options: &JsValue, key: &str) -> Result<Option<bool>, JsValue> {
    option(options, key)?
        .map(|v| v.as_bool().ok_or_else(|| type_error(key, "a boolean")))
        .transpose()
}

fn string_array_option(options: &JsValue, key: &str) -> Result<Option<Vec<String>>, JsValue> {
    let Some(value) = option(options, key)? else {
        return Ok(None);
    };
    let not_strings = || type_error(key, "an array of strings");
    // Read in place and stop at the first bad item: a sparse `new Array(1e9)` fails at once
    // instead of being copied.
    let array = value.dyn_into::<Array>().map_err(|_| not_strings())?;
    array
        .iter()
        .map(|item| item.as_string().ok_or_else(not_strings))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn kinds_option(options: &JsValue) -> Result<KindSet, JsValue> {
    let Some(labels) = string_array_option(options, "kinds")? else {
        return Ok(Kind::all());
    };
    if labels.is_empty() {
        return Err(type_error("kinds", "a non-empty array of kinds"));
    }
    labels.iter().try_fold(KindSet::EMPTY, |set, label| {
        Kind::from_str_label(label)
            .map(|k| set | k)
            .ok_or_else(|| TypeError::new(&format!("unknown kind `{label}`")).into())
    })
}

/// The bundle as bytes, from the `Uint8Array` or `ArrayBuffer` a caller is likely to hold.
fn bundle_bytes(bytes: &JsValue) -> Result<Vec<u8>, JsValue> {
    if let Some(array) = bytes.dyn_ref::<Uint8Array>() {
        return Ok(array.to_vec());
    }
    if let Some(buffer) = bytes.dyn_ref::<ArrayBuffer>() {
        return Ok(Uint8Array::new(buffer).to_vec());
    }
    Err(TypeError::new("bundle bytes must be a Uint8Array or an ArrayBuffer").into())
}

/// The text argument, checked rather than coerced, so a non-string rejects the returned promise.
fn text_argument(text: &JsValue) -> Result<String, JsValue> {
    text.as_string()
        .ok_or_else(|| TypeError::new("text must be a string").into())
}
