//! The page: header, the live demo run by the inference worker, the measured sections, footer.

mod demo;
mod document;
mod figures;
mod format;
mod highlight;
mod json;
mod output;
mod parse;
mod stream;

use gloo_worker::{Spawnable, WorkerBridge};
use leptos::ev;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::{JsCast, JsValue};

use crate::inference::Inference;
use crate::protocol::FoundKind;
use crate::samples::SAMPLES;
use crate::stats;
use demo::{Demo, DemoState, Failure};
use figures::Figures;
use parse::ParseBox;

/// The worker's loader, `site/worker_loader.js`, which Trunk copies beside the worker binary.
const WORKER_URL: &str = "./worker_loader.js";

const SOURCE_URL: &str = "https://github.com/theiskaa/tessera";

/// The JavaScript API as `tessera/js/index.d.ts` declares it. The import path is only an example
/// of a relative import: this page runs the Rust crate in its own worker and does not serve the
/// package. The field comments name the main fields, not all of them.
const SNIPPET: &str = "import { createTessera } from './tessera/index.js'

const tessera = await createTessera({ modelUrl, integrity })
const found = await tessera.detect(text, { countryHint: ['GB'] })
// [{ kind: 'person' | 'org' | 'address' | 'email' | 'phone',
//    text, start, end, confidence, source, ... }]
// addresses also carry components: [{ label, text, start, end, confidence }]

const address = await tessera.parseAddress('Flat 4, 221B Baker Street, London NW1 6XE')
// offsets are UTF-16 code units";

/// How the headline names each kind.
fn plural(kind: FoundKind) -> &'static str {
    match kind {
        FoundKind::Person => "people",
        FoundKind::Org => "organizations",
        FoundKind::Address => "addresses",
        FoundKind::Phone => "phones",
        FoundKind::Email => "emails",
    }
}

/// The whole page. Spawns the worker and sends it the first document; the worker answers once
/// its bundle is loaded.
#[component]
pub fn App() -> impl IntoView {
    let state = DemoState::new();
    if let Some(bridge) = start_worker(state) {
        state.connect(bridge);
    }

    view! {
        <main class="page">
            <Header />
            <Intro />
            <Demo state=state />
            <ParseBox state=state />
            <Figures />
            <Footer state=state />
        </main>
    }
}

/// Spawns the inference worker, or records why it cannot run.
///
/// gloo's spawner panics when `new Worker` throws, as it does on a `file:` page or in a
/// sandboxed frame, so the constructor is tried once first. After that, a worker that fails
/// shows up in one of two places: an uncaught error inside it (a missing script, a wasm load the
/// loader rethrows, a panic) is re-raised on this window, and a loader that cannot be fetched at
/// all is caught by requesting it here.
fn start_worker(state: DemoState) -> Option<WorkerBridge<Inference>> {
    match web_sys::Worker::new(WORKER_URL) {
        Ok(probe) => probe.terminate(),
        Err(e) => {
            state.fail(Failure::Worker(format!(
                "this page cannot start a web worker ({})",
                describe(&e)
            )));
            return None;
        }
    }
    let worker_files: Vec<String> = ["worker_loader.js", "worker.js", "worker_bg.wasm"]
        .iter()
        .filter_map(|name| {
            let base = document().base_uri().ok().flatten()?;
            web_sys::Url::new_with_base(name, &base)
                .ok()
                .map(|u| u.href())
        })
        .collect();
    window_event_listener(ev::error, move |e| {
        if worker_files.contains(&e.filename()) {
            state.fail(Failure::Worker(e.message()));
        }
    });
    spawn_local(async move {
        let failure = match gloo_net::http::Request::get(WORKER_URL).send().await {
            Ok(response) if response.ok() => return,
            Ok(response) => format!("{WORKER_URL}: http {}", response.status()),
            Err(e) => format!("{WORKER_URL}: {e}"),
        };
        state.fail(Failure::Worker(failure));
    });
    Some(
        Inference::spawner()
            .callback(move |response| state.receive(response))
            .with_loader(true)
            .as_module(false)
            .spawn(WORKER_URL),
    )
}

/// A thrown JavaScript value as text.
fn describe(value: &JsValue) -> String {
    value
        .dyn_ref::<js_sys::Error>()
        .map(|e| String::from(e.message()))
        .or_else(|| value.as_string())
        .unwrap_or_else(|| "unknown error".to_string())
}

#[component]
fn Header() -> impl IntoView {
    view! {
        <header class="top">
            <div class="brand">
                <span class="name">"tessera"</span>
                <span class="meta">
                    {format!("v{}", env!("CARGO_PKG_VERSION"))}
                    <span class="wide-only">
                        {format!(" · {} gzip · on-device", format::bytes(stats::BUNDLE_GZIP))}
                    </span>
                </span>
            </div>
            <nav aria-label="Links">
                <a href=SOURCE_URL>"[ source ]"</a>
                <a href="#demo" class="wide-only">
                    "[ live demo ]"
                </a>
            </nav>
        </header>
    }
}

#[component]
fn Intro() -> impl IntoView {
    let last = demo::KINDS.len() - 1;
    let kinds = demo::KINDS
        .iter()
        .enumerate()
        .map(|(i, &kind)| {
            let joint = match i {
                0 => "",
                i if i == last => " and ",
                _ => ", ",
            };
            view! {
                {joint}
                <span
                    class=format!("hit k-{}", json::kind_name(kind))
                    data-anim=""
                    style=format!("--i: {i}")
                >
                    {plural(kind)}
                </span>
            }
        })
        .collect_view();
    view! {
        <section class="intro">
            <div>
                <h1>"Find " {kinds} " in text"</h1>
                <p class="subtitle">"in the browser, with exact spans"</p>
            </div>
            <pre class="code">{highlight::render(highlight::javascript(SNIPPET))}</pre>
            <p>
                "Emails and phone numbers are found by validating rules, people, organizations and addresses by an int8 network, and each address is split into its parts by a second one, all in WebAssembly. Offsets are UTF-8 bytes from Rust and UTF-16 code units from JavaScript."
            </p>
        </section>
    }
}

#[component]
fn Footer(state: DemoState) -> impl IntoView {
    view! {
        <footer class="bottom">
            <span>
                "[ live demos ] "
                {SAMPLES
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        view! {
                            {(i > 0).then_some(" · ")}
                            <a href="#demo" on:click=move |_| state.pick(i)>
                                {s.name}
                            </a>
                        }
                    })
                    .collect_view()}
            </span>
            <span>
                {format!(
                    "code MIT or Apache-2.0 · model trained on © OpenStreetMap contributors, ODbL · v{}",
                    env!("CARGO_PKG_VERSION"),
                )}
            </span>
        </footer>
    }
}
