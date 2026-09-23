//! Feature `wasm`: the JavaScript `Tessera` class and `createInstance`, which fetches and
//! verifies the bundle and can move inference into a Web Worker. Offsets cross this boundary as
//! UTF-16 code units, result objects carry a `text` field, and every library failure is a JS
//! `Error` named `TesseraError` with a stable `code`. Calls return a `Promise` whether the
//! instance runs in this thread or in a worker. Instantiating the module is the package entry's
//! job, in `js/index.js`, and `js/worker.js` only instantiates and relays messages.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use js_sys::{Array, ArrayBuffer, Function, Object, Promise, Reflect, TypeError, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{MessageEvent, Response, Worker, WorkerOptions, WorkerType};

use crate::policy::display_confidence;
use crate::{Config, Entity, Error, Kind, KindSet, Query, Tessera, token};

/// The JavaScript `Tessera` class: a loaded extractor reporting UTF-16 offsets.
///
/// Keep a reference to an instance while its calls are in flight. The generated glue frees an
/// unreachable instance through a `FinalizationRegistry`, which for a worker-backed instance
/// terminates the worker and rejects those calls with `DISPOSED`.
#[wasm_bindgen(js_name = Tessera)]
pub struct JsTessera {
    backend: Option<Backend>,
}

enum Backend {
    Local(Box<Tessera>),
    Remote(Remote),
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
        let local = load_local(&bytes.to_vec(), kinds, integrity.as_deref())?;
        Ok(JsTessera {
            backend: Some(Backend::Local(Box::new(local))),
        })
    }

    /// The kinds this instance was loaded for, as labels in taxonomy order; empty once disposed.
    #[wasm_bindgen(getter)]
    pub fn kinds(&self) -> Array {
        kind_labels(match &self.backend {
            Some(Backend::Local(t)) => t.kinds(),
            Some(Backend::Remote(r)) => r.kinds,
            None => KindSet::EMPTY,
        })
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
        self.call("parseAddress", text, options, |t, text, opts| {
            let entity = opts.run(|q| t.parse_address(text, q))?;
            Ok(entity_objects(text, std::slice::from_ref(&entity)).get(0))
        })
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
        self.call("detect", text, options, |t, text, opts| {
            Ok(entity_objects(text, &opts.run(|q| t.detect(text, q))?).into())
        })
    }

    /// Release the model, and terminate the worker of a worker-backed instance. Calls already in
    /// flight and every later call reject with code `DISPOSED`; disposing twice is harmless. After
    /// `free()`, which also terminates the worker, every call throws synchronously, as with any
    /// wasm-bindgen class.
    pub fn dispose(&mut self) {
        self.backend = None;
    }

    /// Validates the arguments here, then runs `local` in this thread or posts `op` to the worker.
    fn call(
        &self,
        op: &str,
        text: &JsValue,
        options: Option<JsValue>,
        local: impl FnOnce(&Tessera, &str, &CallOptions) -> Result<JsValue, JsValue>,
    ) -> Promise {
        let outcome = self
            .backend
            .as_ref()
            .ok_or_else(disposed)
            .and_then(|backend| {
                if !text.is_string() {
                    return Err(TypeError::new("text must be a string").into());
                }
                let options = CallOptions::read(options)?;
                Ok(match backend {
                    Backend::Local(t) => {
                        let text = text.as_string().unwrap_or_default();
                        Promise::resolve(&local(t, &text, &options)?)
                    }
                    // The string goes to the worker as it is, never copied into this thread's
                    // wasm memory, which would otherwise grow to the largest document seen.
                    Backend::Remote(r) => r.request(op, text, &options),
                })
            });
        outcome.unwrap_or_else(|err| Promise::reject(&err))
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
///
/// `worker: true` on a browser main thread loads the bundle into a module worker started from
/// `workerUrl`, which instantiates the module from `wasmUrl`; the package entry supplies both, and
/// their absence is `UNSUPPORTED_RUNTIME`. Elsewhere, in Node, Bun, or a worker, it is ignored.
#[wasm_bindgen(js_name = createInstance)]
pub async fn create_instance(
    #[wasm_bindgen(
        unchecked_param_type = "{ kinds?: string[]; integrity?: string; modelUrl?: string; modelBytes?: Uint8Array | ArrayBuffer; worker?: boolean; workerUrl?: string; wasmUrl?: string }"
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
    let worker = worker_urls(&options)?;
    // `owned` bytes are this function's own copy, so a worker can take them without copying.
    let (bytes, owned) = match (given, url) {
        _ if kinds.is_rules_only() => (Uint8Array::new_with_length(0), true),
        (Some(given), _) => (given, false),
        (None, Some(url)) => (fetch_bundle(&url).await?, true),
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
    let integrity = integrity.as_deref();
    let backend = match worker {
        Some((worker_url, wasm_url)) => {
            let bundle = Bundle {
                bytes,
                owned,
                kinds,
                integrity,
            };
            Backend::Remote(Remote::start(&worker_url, &wasm_url, bundle).await?)
        }
        None => Backend::Local(Box::new(load_local(&bytes.to_vec(), kinds, integrity)?)),
    };
    Ok(JsTessera {
        backend: Some(backend),
    })
}

fn load_local(bytes: &[u8], kinds: KindSet, integrity: Option<&str>) -> Result<Tessera, JsValue> {
    Ok(Tessera::load(
        bytes,
        Config {
            kinds,
            expected_checksum: integrity,
        },
    )?)
}

/// The worker and wasm URLs when `options.worker` asks for a worker on a browser main thread,
/// the one place a worker helps: only there are both `Worker` and `document` defined.
fn worker_urls(options: &JsValue) -> Result<Option<(String, String)>, JsValue> {
    let global = js_sys::global();
    let has = |key: &str| Reflect::has(&global, &JsValue::from_str(key)).unwrap_or(false);
    if !bool_option(options, "worker")?.unwrap_or(false) || !has("document") || !has("Worker") {
        return Ok(None);
    }
    let required = |key: &str| {
        string_option(options, key)?.ok_or_else(|| {
            let message = format!("{key} is missing; call createTessera from the package entry");
            tessera_error("UNSUPPORTED_RUNTIME", &message, None)
        })
    };
    Ok(Some((required("workerUrl")?, required("wasmUrl")?)))
}

/// What a worker needs to load its instance.
struct Bundle<'a> {
    bytes: Uint8Array,
    owned: bool,
    kinds: KindSet,
    integrity: Option<&'a str>,
}

/// A call waiting for a worker reply.
struct Waiting {
    id: u32,
    resolve: Function,
    reject: Function,
    /// The code and stage for a failure the worker reports without a code: for `load` the module
    /// could not start, for anything else the worker broke.
    uncoded: (&'static str, Option<&'static str>),
}

const LOAD_FAILED: (&str, Option<&str>) = ("UNSUPPORTED_RUNTIME", None);
const CALL_FAILED: (&str, Option<&str>) = ("INFERENCE", Some("worker"));

/// A few calls wait at a time, so a list costs less code than a hash map for the same speed.
type Pending = Rc<RefCell<Vec<Waiting>>>;

/// A module worker holding its own instance. Each call is one message with an id, answered by
/// `{ id, ok: true, result }` or `{ id, ok: false, error: { code, message, stage } }`.
struct Remote {
    worker: Worker,
    pending: Pending,
    /// Set by the worker's `error` event, after which the worker answers nothing.
    stopped: Rc<RefCell<Option<String>>>,
    next_id: Cell<u32>,
    kinds: KindSet,
    _onmessage: Closure<dyn FnMut(MessageEvent)>,
    messageerror_listener: Closure<dyn FnMut(JsValue)>,
    _onerror: Closure<dyn FnMut(JsValue)>,
}

impl Remote {
    /// Starts the worker at `worker_url` and waits for it to load `bundle`.
    async fn start(
        worker_url: &str,
        wasm_url: &str,
        bundle: Bundle<'_>,
    ) -> Result<Remote, JsValue> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options(worker_url, &options).map_err(|e| {
            let message = format!("could not start the worker: {}", describe(&e));
            tessera_error("UNSUPPORTED_RUNTIME", &message, None)
        })?;
        let pending: Pending = Rc::default();
        let stopped: Rc<RefCell<Option<String>>> = Rc::default();
        let onmessage = {
            let pending = Rc::clone(&pending);
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                settle_reply(&pending, &event.data());
            })
        };
        // A reply that cannot be deserialized carries no id, so every waiting call is rejected
        // rather than one of them left waiting forever.
        let onmessageerror = {
            let pending = Rc::clone(&pending);
            Closure::<dyn FnMut(JsValue)>::new(move |_: JsValue| {
                let err = tessera_error(
                    "INFERENCE",
                    "a worker reply could not be read",
                    Some("worker"),
                );
                reject_all(&pending, &err);
            })
        };
        // A module that fails to import, or a worker that dies, reports here and never replies,
        // so the waiting calls are rejected and later calls fail at once instead of hanging.
        let onerror = {
            let pending = Rc::clone(&pending);
            let stopped = Rc::clone(&stopped);
            Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
                let message = field(&event, "message")
                    .as_string()
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| "the worker stopped".into());
                reject_all(
                    &pending,
                    &tessera_error("UNSUPPORTED_RUNTIME", &message, None),
                );
                *stopped.borrow_mut() = Some(message);
            })
        };
        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        // Chrome's `Worker` has no `onmessageerror` attribute, only the event.
        let _ = worker.add_event_listener_with_callback(
            "messageerror",
            onmessageerror.as_ref().unchecked_ref(),
        );
        let remote = Remote {
            worker,
            pending,
            stopped,
            next_id: Cell::new(0),
            kinds: bundle.kinds,
            _onmessage: onmessage,
            messageerror_listener: onmessageerror,
            _onerror: onerror,
        };
        let load = Object::new();
        set(&load, "kinds", kind_labels(bundle.kinds));
        if let Some(integrity) = bundle.integrity {
            set(&load, "integrity", integrity);
        }
        let message = Object::new();
        set(&message, "op", "load");
        set(&message, "wasmUrl", wasm_url);
        set(&message, "bytes", bundle.bytes.clone());
        set(&message, "options", load);
        // A caller's `modelBytes` are copied by structured cloning; bytes of our own are moved.
        let transfer = bundle.owned.then(|| bundle.bytes.buffer());
        JsFuture::from(remote.post(&message, transfer, LOAD_FAILED)).await?;
        Ok(remote)
    }

    fn request(&self, op: &str, text: &JsValue, options: &CallOptions) -> Promise {
        let message = Object::new();
        set(&message, "op", op);
        set(&message, "text", text);
        set(&message, "options", options.to_js());
        self.post(&message, None, CALL_FAILED)
    }

    /// Posts `message` under a fresh id and returns a promise the worker's reply settles.
    fn post(
        &self,
        message: &Object,
        transfer: Option<ArrayBuffer>,
        uncoded: (&'static str, Option<&'static str>),
    ) -> Promise {
        if let Some(message) = self.stopped.borrow().as_deref() {
            return Promise::reject(&tessera_error("UNSUPPORTED_RUNTIME", message, None));
        }
        let id = self.next_id.get();
        self.next_id.set(id.wrapping_add(1));
        set(message, "id", id);
        let promise = Promise::new(&mut |resolve, reject| {
            self.pending.borrow_mut().push(Waiting {
                id,
                resolve,
                reject,
                uncoded,
            });
        });
        let posted = match transfer {
            Some(buffer) => self
                .worker
                .post_message_with_transfer(message, &Array::of1(&buffer)),
            None => self.worker.post_message(message),
        };
        if let Err(e) = posted
            && let Some(waiting) = take(&self.pending, id)
        {
            let err = tessera_error("INFERENCE", &describe(&e), Some("worker"));
            let _ = waiting.reject.call1(&JsValue::UNDEFINED, &err);
        }
        promise
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.worker.terminate();
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        let _ = self.worker.remove_event_listener_with_callback(
            "messageerror",
            self.messageerror_listener.as_ref().unchecked_ref(),
        );
        reject_all(&self.pending, &disposed());
    }
}

/// Settles the call a worker reply answers. The error arrives as a plain object because
/// structured cloning keeps only an `Error`'s name and message, so the `TesseraError` is rebuilt.
fn settle_reply(pending: &Pending, reply: &JsValue) {
    let Some(id) = field(reply, "id").as_f64() else {
        return;
    };
    let Some(waiting) = take(pending, id as u32) else {
        return;
    };
    if field(reply, "ok").as_bool() == Some(true) {
        let _ = waiting
            .resolve
            .call1(&JsValue::UNDEFINED, &field(reply, "result"));
        return;
    }
    let error = field(reply, "error");
    let message = field(&error, "message")
        .as_string()
        .unwrap_or_else(|| "the worker failed".into());
    let err = match field(&error, "code").as_string() {
        Some(code) => tessera_error(
            &code,
            &message,
            field(&error, "stage").as_string().as_deref(),
        ),
        None => tessera_error(waiting.uncoded.0, &message, waiting.uncoded.1),
    };
    let _ = waiting.reject.call1(&JsValue::UNDEFINED, &err);
}

fn take(pending: &Pending, id: u32) -> Option<Waiting> {
    let mut pending = pending.borrow_mut();
    let at = pending.iter().position(|w| w.id == id)?;
    Some(pending.swap_remove(at))
}

fn reject_all(pending: &Pending, err: &JsValue) {
    let waiting = std::mem::take(&mut *pending.borrow_mut());
    for w in waiting {
        let _ = w.reject.call1(&JsValue::UNDEFINED, err);
    }
}

fn disposed() -> JsValue {
    tessera_error("DISPOSED", "this Tessera instance has been disposed", None)
}

/// The bundle at `url`, fetched through the global object's `fetch` rather than `window`'s,
/// because a worker, Node, and Bun have no `window`.
async fn fetch_bundle(url: &str) -> Result<Uint8Array, JsValue> {
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
    Ok(Uint8Array::new(&buffer))
}

/// The message of a thrown or rejected value, which need not be an `Error`.
fn describe(value: &JsValue) -> String {
    value
        .dyn_ref::<js_sys::Error>()
        .map(|e| String::from(e.message()))
        .or_else(|| value.as_string())
        .unwrap_or_else(|| "unknown error".into())
}

/// Validated per-call options, owned so they can run here or be posted to a worker as plain data.
struct CallOptions {
    country_hint: Vec<String>,
    include_uncertain: bool,
}

impl CallOptions {
    fn read(options: Option<JsValue>) -> Result<CallOptions, JsValue> {
        let options = options.unwrap_or_default();
        Ok(CallOptions {
            country_hint: string_array_option(&options, "countryHint")?.unwrap_or_default(),
            include_uncertain: bool_option(&options, "includeUncertain")?.unwrap_or(false),
        })
    }

    fn run<T>(&self, call: impl FnOnce(&Query<'_>) -> Result<T, Error>) -> Result<T, JsValue> {
        let hints: Vec<&str> = self.country_hint.iter().map(String::as_str).collect();
        let query = Query {
            country_hint: &hints,
            include_uncertain: self.include_uncertain,
        };
        call(&query).map_err(JsValue::from)
    }

    fn to_js(&self) -> Object {
        let obj = Object::new();
        let hints: Array = self
            .country_hint
            .iter()
            .map(|h| JsValue::from_str(h))
            .collect();
        set(&obj, "countryHint", hints);
        set(&obj, "includeUncertain", self.include_uncertain);
        obj
    }
}

fn kind_labels(set: KindSet) -> Array {
    Kind::ALL
        .iter()
        .filter(|k| set.contains(**k))
        .map(|k| JsValue::from_str(k.as_str()))
        .collect()
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

/// `value[key]`, or `undefined` when `value` is not an object or the read throws.
fn field(value: &JsValue, key: &str) -> JsValue {
    Reflect::get(value, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
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
