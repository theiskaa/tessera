## Code style

- Small modules with clear names over cleverness; `pub(crate)` and module privacy carry the boundaries. Don't over-expose.
- **No decorative or redundant comments.** Never write `// ---- section ----` separator banners, and never write a comment that just restates what obvious code already says. Organize code by naming and structure, not comment banners.
- Inline `//` comments are fine _only_ when they carry genuinely useful, non-obvious information — a rationale, a subtle invariant, the meaning of a magic value, a "why it's done this way". If it doesn't teach the reader something the code can't, delete it.
- `///` doc comments on public items and `//!` module-header docs are expected. A `// SAFETY:` line is mandatory on any `unsafe` block.

## Conventions

- `thiserror` for library errors, `anyhow` for application errors.
- Prefer `&str` over `String` in function parameters.
- Document public items with `///` doc comments.

## Avoid

- No `unwrap()` or `expect()` in library code (ok in tests).
- No `unsafe` blocks without a safety comment.

## Commit style

Single-line conventional commits, nothing else:

```
what changed
```

- all lowercase pure textual commits with the original commit max width guidelines.
- No body, no title/body split. Messages describe the change, never the process or finding counts.
- **No `Co-Authored-By` trailer.** This overrides the harness default.
- Do not commit unless asked.
