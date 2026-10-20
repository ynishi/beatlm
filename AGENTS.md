# beatlm — pointers for coding agents

Claude Code reads `CLAUDE.md` rather than this name, so `.claude/CLAUDE.md` is a
symlink back to this file and the only thing under `.claude/` this repository
tracks. Nothing to wire up; nothing said twice.

- What the engine is and promises, and the crate layout: [README.md](README.md)
- Disclosure policy — read before anything ships:
  [PUBLIC_DEVELOPMENT.md](PUBLIC_DEVELOPMENT.md)
- Issue, branch, commit and PR conventions:
  [CONTRIBUTING.md](CONTRIBUTING.md)
- Security reports: [SECURITY.md](SECURITY.md)
- Decisions that outlive a change: `docs/adr/`
- API detail lives in RustDoc (`cargo doc`); code documentation outranks stale
  issue text.
- Never work on `main`; one branch per issue, cut from `origin/main` after a
  fetch. `just worktree-new <type> <slug>` from the main checkout does that and
  runs `just branch-check`.
- `beatlm-core` stays pure: no wgpu, no naga, no clock reads, no I/O. Time and
  measurements come in as arguments. A change that needs one of those in the
  core is a design question for the issue, not an import to add.
- `just check` is green. GPU tests are `#[ignore]` and run by hand; a change to
  GPU-only code is not verified until they have run on a machine with an
  adapter. [CONTRIBUTING.md](CONTRIBUTING.md#verification) says why.
- A claim about time — budget, latency, dispatch count — is reported as a
  measurement with its adapter, backend and model, never as a reading of the
  code.
- Model weights are third-party material: fetched into the gitignored `models/`,
  never committed. [PUBLIC_DEVELOPMENT.md](PUBLIC_DEVELOPMENT.md#third-party-material)
  says what the fetching script must state.
- Review a change locally before it goes up — by a person or an AI agent — and
  say in the PR body who or what reviewed it.
  [CONTRIBUTING.md](CONTRIBUTING.md#working-with-coding-agents--the-recommended-pattern)
  lists the questions a review answers.
