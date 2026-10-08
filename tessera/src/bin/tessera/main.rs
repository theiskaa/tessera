//! Feature `cli`: text in, JSON out. Native only.
//!
//! `tessera [flags] [FILE]` detects entities in a file or stdin. `tessera json --bundle DIR`
//! answers one JSON request read from stdin, for programs that drive the binary as a subprocess.

mod bundle;
mod flags;
mod json;
mod render;

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_some_and(|a| a == "json") {
        args.next();
        return json::main(args);
    }
    flags::main(args)
}
