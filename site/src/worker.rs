//! Worker binary entry. The page spawns `worker_loader.js`, a hand-written loader that Trunk
//! copies beside this binary's glue and wasm, and the loader instantiates it.

use gloo_worker::Registrable;
use site::inference::Inference;

fn main() {
    console_error_panic_hook::set_once();
    Inference::registrar().register();
}
