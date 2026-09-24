//! The inference worker: fetches and verifies the bundle once, then answers requests off the
//! page's thread. Requests sent while the bundle loads wait for it.

use gloo_worker::{HandlerId, Worker, WorkerScope};
use tessera::{Config, Kind, Query, Tessera};

use crate::protocol::{Found, Request, Response};

/// Resolved against the worker script, which Trunk puts beside the bundle.
const BUNDLE_URL: &str = "tessera-v1.safetensors";
const CHECKSUM: &str = include_str!("../../models/tessera-v1.sha256");

/// The worker's state.
pub struct Inference {
    tessera: Loading,
    waiting: Vec<(HandlerId, Request)>,
}

enum Loading {
    Pending,
    Ready(Box<Tessera>),
    Failed(String),
}

/// Messages the worker sends itself.
pub struct Loaded(Result<Tessera, String>);

impl Worker for Inference {
    type Message = Loaded;
    type Input = Request;
    type Output = Response;

    fn create(scope: &WorkerScope<Self>) -> Self {
        scope.send_future(async { Loaded(load().await) });
        Inference {
            tessera: Loading::Pending,
            waiting: Vec::new(),
        }
    }

    fn update(&mut self, scope: &WorkerScope<Self>, Loaded(tessera): Loaded) {
        self.tessera = match tessera {
            Ok(tessera) => Loading::Ready(Box::new(tessera)),
            Err(e) => Loading::Failed(e),
        };
        for (id, request) in std::mem::take(&mut self.waiting) {
            self.received(scope, request, id);
        }
    }

    fn received(&mut self, scope: &WorkerScope<Self>, request: Request, id: HandlerId) {
        match &self.tessera {
            Loading::Pending => self.waiting.push((id, request)),
            Loading::Failed(message) => scope.respond(id, Response::LoadFailed(message.clone())),
            Loading::Ready(tessera) => scope.respond(id, answer(tessera, request)),
        }
    }
}

fn answer(tessera: &Tessera, request: Request) -> Response {
    match request {
        Request::ParseAddress { id, text } => Response::Parsed {
            id,
            result: tessera
                .parse_address(&text, &Query::default())
                .map(|entity| Found::from_entity(&entity))
                .map_err(|e| format!("{e:?}: {e}")),
        },
        Request::Detect {
            id,
            text,
            country_hint,
        } => {
            let hints: Vec<&str> = country_hint.iter().map(String::as_str).collect();
            let query = Query {
                country_hint: &hints,
                ..Query::default()
            };
            Response::Detected {
                id,
                result: tessera
                    .detect(&text, &query)
                    .map(|found| found.iter().map(Found::from_entity).collect())
                    .map_err(|e| format!("{e:?}: {e}")),
            }
        }
    }
}

async fn load() -> Result<Tessera, String> {
    let response = gloo_net::http::Request::get(BUNDLE_URL)
        .send()
        .await
        .map_err(|e| format!("{BUNDLE_URL}: {e}"))?;
    if !response.ok() {
        return Err(format!("{BUNDLE_URL}: http {}", response.status()));
    }
    let bytes = response
        .binary()
        .await
        .map_err(|e| format!("{BUNDLE_URL}: {e}"))?;
    Tessera::load(
        &bytes,
        Config {
            kinds: Kind::all(),
            expected_checksum: Some(CHECKSUM.trim()),
        },
    )
    .map_err(|e| format!("{BUNDLE_URL}: {e}"))
}
