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
    models: Loading,
    waiting: Vec<(HandlerId, Request)>,
}

enum Loading {
    Pending,
    Ready(Box<Models>),
    Failed(String),
}

/// `detect` needs an instance loaded for emails and phones only, so the rules and the parser are
/// two instances.
struct Models {
    rules: Tessera,
    parser: Tessera,
}

/// Messages the worker sends itself.
pub struct Loaded(Result<Tessera, String>);

impl Worker for Inference {
    type Message = Loaded;
    type Input = Request;
    type Output = Response;

    fn create(scope: &WorkerScope<Self>) -> Self {
        scope.send_future(async { Loaded(load_parser().await) });
        Inference {
            models: Loading::Pending,
            waiting: Vec::new(),
        }
    }

    fn update(&mut self, scope: &WorkerScope<Self>, Loaded(parser): Loaded) {
        let rules = Tessera::load(
            &[],
            Config {
                kinds: Kind::Email | Kind::Phone,
                expected_checksum: None,
            },
        )
        .map_err(|e| e.to_string());
        self.models = match (parser, rules) {
            (Ok(parser), Ok(rules)) => Loading::Ready(Box::new(Models { rules, parser })),
            (Err(e), _) | (_, Err(e)) => Loading::Failed(e),
        };
        for (id, request) in std::mem::take(&mut self.waiting) {
            self.received(scope, request, id);
        }
    }

    fn received(&mut self, scope: &WorkerScope<Self>, request: Request, id: HandlerId) {
        match &self.models {
            Loading::Pending => self.waiting.push((id, request)),
            Loading::Failed(message) => scope.respond(id, Response::LoadFailed(message.clone())),
            Loading::Ready(models) => scope.respond(id, models.answer(request)),
        }
    }
}

impl Models {
    fn answer(&self, request: Request) -> Response {
        match request {
            Request::ParseAddress { id, text } => Response::Parsed {
                id,
                result: self
                    .parser
                    .parse_address(&text, &Query::default())
                    .map_err(|e| format!("{e:?}: {e}"))
                    .and_then(|entity| {
                        Found::from_entity(&entity, 0)
                            .ok_or_else(|| "the parser returned no address".to_string())
                    }),
            },
            Request::Analyze {
                id,
                text,
                country_hint,
                addresses,
            } => Response::Analyzed {
                id,
                result: self.analyze(&text, &country_hint, &addresses),
            },
        }
    }

    fn analyze(
        &self,
        text: &str,
        country_hint: &[String],
        addresses: &[(usize, usize)],
    ) -> Result<Vec<Found>, String> {
        let hints: Vec<&str> = country_hint.iter().map(String::as_str).collect();
        let query = Query {
            country_hint: &hints,
            ..Query::default()
        };
        let mut found: Vec<Found> = self
            .rules
            .detect(text, &query)
            .map_err(|e| e.to_string())?
            .iter()
            .filter_map(|e| Found::from_entity(e, 0))
            .collect();
        // The page sends only spans it located on the text itself, so each one slices it.
        for (start, span) in addresses
            .iter()
            .filter_map(|&(start, end)| Some((start, text.get(start..end)?)))
        {
            let entity = self
                .parser
                .parse_address(span, &Query::default())
                .map_err(|e| format!("{e:?}: {e}"))?;
            found.extend(Found::from_entity(&entity, start));
        }
        found.sort_by_key(|f| (f.start, f.end));
        Ok(found)
    }
}

async fn load_parser() -> Result<Tessera, String> {
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
            kinds: Kind::Address.into(),
            expected_checksum: Some(CHECKSUM.trim()),
        },
    )
    .map_err(|e| format!("{BUNDLE_URL}: {e}"))
}
