//! Feature `wasm`: the JavaScript `Tessera` class and `createInstance`, which fetches and
//! verifies the bundle. Offsets cross this boundary as UTF-16 code units, result objects carry a
//! `text` field, and every library failure is a JS `Error` named `TesseraError` with a stable
//! `code`. Calls return a `Promise`, so a worker-backed instance can keep the shape of this
//! in-thread one. Instantiating the module and the worker are not here.

use js_sys::{Array, ArrayBuffer, Function, Object, Promise, Reflect, TypeError, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::Response;

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
        let bytes = byte_view(bytes)
            .ok_or_else(|| TypeError::new("bundle bytes must be a Uint8Array or an ArrayBuffer"))?;
        let kinds = kinds_option(&options)?;
        let integrity = string_option(&options, "integrity")?;
        construct(&bytes.to_vec(), kinds, integrity.as_deref())
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

/// `createInstance(options)`: build an instance once the module is instantiated. Called by the
/// package's `createTessera`.
///
/// Takes the options of [`JsTessera::load`] plus `modelBytes`, a `Uint8Array` or `ArrayBuffer`,
/// and `modelUrl`, fetched with the global `fetch` when `modelBytes` is absent. A rules-only
/// `kinds` set needs neither and fetches nothing. Rejects with `BUNDLE_INVALID` when a model
/// kind has no bundle source, `UNSUPPORTED_RUNTIME` when the bundle must be fetched and the
/// runtime has no `fetch`, and `MODEL_FETCH_FAILED` when the request, its status, or its body
/// fails; otherwise rejects as `Tessera.load` throws (`TypeError`, `CHECKSUM_MISMATCH`, …).
#[wasm_bindgen(js_name = createInstance)]
pub async fn create_instance(
    #[wasm_bindgen(
        unchecked_param_type = "{ kinds?: string[]; integrity?: string; modelUrl?: string; modelBytes?: Uint8Array | ArrayBuffer }"
    )]
    options: JsValue,
) -> Result<JsTessera, JsValue> {
    let kinds = kinds_option(&options)?;
    let integrity = string_option(&options, "integrity")?;
    let given = option(&options, "modelBytes")?
        .map(|v| {
            byte_view(&v).ok_or_else(|| type_error("modelBytes", "a Uint8Array or an ArrayBuffer"))
        })
        .transpose()?;
    let url = string_option(&options, "modelUrl")?;
    let bytes = match (given, url) {
        _ if kinds.is_rules_only() => Vec::new(),
        (Some(given), _) => given.to_vec(),
        (None, Some(url)) => fetch_bundle(&url).await?,
        (None, None) => {
            let labels: Vec<&str> = Kind::ALL
                .iter()
                .filter(|k| kinds.contains(**k))
                .map(|k| k.as_str())
                .collect();
            let message = format!(
                "modelUrl or modelBytes is required for kinds [{}]",
                labels.join(", ")
            );
            return Err(tessera_error("BUNDLE_INVALID", &message, None));
        }
    };
    construct(&bytes, kinds, integrity.as_deref())
}

fn construct(bytes: &[u8], kinds: KindSet, integrity: Option<&str>) -> Result<JsTessera, JsValue> {
    let inner = Tessera::load(
        bytes,
        Config {
            kinds,
            expected_checksum: integrity,
        },
    )?;
    Ok(JsTessera { inner: Some(inner) })
}

/// The bundle at `url`, fetched through the global object's `fetch` rather than `window`'s,
/// because a worker, Node, and Bun have no `window`.
async fn fetch_bundle(url: &str) -> Result<Vec<u8>, JsValue> {
    let global = js_sys::global();
    let fetch = Reflect::get(&global, &JsValue::from_str("fetch"))
        .ok()
        .and_then(|f| f.dyn_into::<Function>().ok())
        .ok_or_else(|| {
            tessera_error(
                "UNSUPPORTED_RUNTIME",
                "this runtime has no fetch; pass modelBytes instead",
                None,
            )
        })?;
    let failed =
        |detail: &str| tessera_error("MODEL_FETCH_FAILED", &format!("{url}: {detail}"), None);
    let pending = fetch
        .call1(&global, &JsValue::from_str(url))
        .map_err(|e| failed(&describe(&e)))?;
    let response: Response = JsFuture::from(Promise::resolve(&pending))
        .await
        .map_err(|e| failed(&describe(&e)))?
        .dyn_into()
        .map_err(|_| failed("fetch did not return a Response"))?;
    if !response.ok() {
        return Err(failed(&format!("http {}", response.status())));
    }
    let body = response.array_buffer().map_err(|e| failed(&describe(&e)))?;
    let buffer = JsFuture::from(body)
        .await
        .map_err(|e| failed(&describe(&e)))?;
    Ok(Uint8Array::new(&buffer).to_vec())
}

/// The message of a thrown or rejected value, which need not be an `Error`.
fn describe(value: &JsValue) -> String {
    value
        .dyn_ref::<js_sys::Error>()
        .map(|e| String::from(e.message()))
        .or_else(|| value.as_string())
        .unwrap_or_else(|| "unknown error".into())
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

/// A view of the bytes in the `Uint8Array` or `ArrayBuffer` a caller is likely to hold, without
/// copying them; `None` for anything else.
fn byte_view(bytes: &JsValue) -> Option<Uint8Array> {
    if let Some(array) = bytes.dyn_ref::<Uint8Array>() {
        return Some(array.clone());
    }
    bytes.dyn_ref::<ArrayBuffer>().map(|b| Uint8Array::new(b))
}

/// The text argument, checked rather than coerced, so a non-string rejects the returned promise.
fn text_argument(text: &JsValue) -> Result<String, JsValue> {
    text.as_string()
        .ok_or_else(|| TypeError::new("text must be a string").into())
}
