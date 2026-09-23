//! Worker binary entry. Trunk builds it with a loader shim, `worker_loader.js`, which the page
//! spawns.

use gloo_worker::Registrable;
use site::inference::Inference;

fn main() {
    console_error_panic_hook::set_once();
    Inference::registrar().register();
}
