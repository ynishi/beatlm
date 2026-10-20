# Contributing

Shared conventions for changes to this repository, for humans and coding agents
alike. The disclosure policy in [PUBLIC_DEVELOPMENT.md](PUBLIC_DEVELOPMENT.md)
outranks this file.

Three kinds of statement share this file, and the boundary between them is a
rule of its own. What the repository decides binds everyone who contributes:
never on `main`, the commit format, what a gate is and when it runs. What
installed tooling decides travels with the tool: a plugin's agents behave the
same in every repository that installs them, and this file does not restate
them. What one setup decides — which directory holds a second
checkout — binds nobody, and a file contributors read as policy does not state
it as policy. Where a recipe here fixes one of those choices, this file says so
as a fact about the recipe, next to the rule the recipe is serving. What belongs
to your own setup goes somewhere the tree does not track — for Claude Code, a
`CLAUDE*.md` of your own, which `.gitignore` keeps out of the tree.

## Issue conventions

### Labels

Four categories. Assign at least one when you open an issue.

| Label         | The change                                           |
| ------------- | ---------------------------------------------------- |
| `bug`         | behaviour contradicts what it promises               |
| `enhancement` | behaviour that does not exist yet                    |
| `refactor`    | behaviour unchanged                                  |
| `chore`       | production code untouched — CI, tests, tooling, docs |

A performance regression is a `bug` when it breaks a promise the engine makes —
a frame budget, a delivery latency — and an `enhancement` otherwise. The promise
is the line, not the size of the number.

## Branches

Never work on `main`. One branch per issue, cut from `origin/main` after a fetch
and named `<type>/<slug>`, where `<type>` is `ci`, `fix`, `feat` or `docs`.

Whether that branch lives in a second worktree or in the checkout you already
have is your setup's decision. If the second checkout sits inside this tree, it
sits where `.gitignore` already covers, so the checkout containing it stays
clean.

Here a recipe makes that choice, and this is what it chooses:

```bash
just worktree-new <type> <slug>
```

That is `git fetch origin`, then
`git worktree add .worktrees/<slug> -b <type>/<slug> origin/main` — one worktree
per issue, under the gitignored `.worktrees/` — then `just branch-check` to
verify the base before you build on it. Run it from the main checkout, and
remove the worktree once the branch is merged.

## Verification

`just check` is the definition of green, and CI is where it runs. It formats,
lints with warnings denied, tests, lints the umbrella crate for
`wasm32-unknown-unknown`, and runs `cargo deny`. The browser build is part of
green on every change: a target that is only checked when someone gets round to
the demo is a target that is broken when they do.

**Tests that need a real GPU adapter are not part of `check`.** A hosted runner
does not guarantee an adapter, and a GPU test that skips itself when none is
found contributes a green step that checked nothing. Such tests are marked
`#[ignore]` with a reason naming the adapter they need, and run by hand with
`cargo test -- --ignored`. A change to code that only runs on the GPU is not
verified until they have run on a machine that has one.

**A claim about time is a measurement, not a reading of the code.** Frame
budgets, dispatch counts and delivery latency are the engine's promises; a
change that touches them reports the numbers it measured, on which adapter and
backend, and with what model. A number without that context is not evidence
for or against a promise.

Report what was actually run; "I did not verify X" is a usable report, a green
claim resting on a recipe nobody ran is not.

## Documentation

API detail lives in the rustdoc. `cargo doc` is where a reader goes for what a
type is and what a function promises. When two texts disagree the code wins, and
after that the doc comment: a comment contradicting an issue is a finding about
the issue.

Four rules for writing that prose.

**State a rule once, where the thing it constrains is defined.** Everywhere
else, link to that statement and write only what this site adds — why it is an
instance of the rule, and what follows here. A second full statement is the copy
nobody edits the day the rule moves.

**Do not write the state of the tree where nothing maintains it.** "Not yet",
"neither has one", "the only caller" are true when written and go false in
silence. Keep the rule and cut the clause about today.

**A number describing a list belongs in the file holding the list, or nowhere.**
The list is the answer and the number is a copy of it. Point at the list
instead.

**A sentence recording that a rule changed stays.** It is a constraint written
in the past tense, and it is what stands between the next reader and undoing the
rule. The test is whether deleting it lets somebody repeat the mistake. If
deleting it costs nothing it is a report on how a change went, and that belongs
in the commit message.

Decisions that outlive a single change — a concurrency model, a sync promise —
are recorded under `docs/adr/`, one decision per file, with the measurement that
settled it.

## Preferred commit format

```text
<subject: what changed, one line>

<prose: the problem, why this fix and not the alternative, what it
cost. This is where the reasoning lives.>

Verified: <the recipes actually run, and their outcome>

Refs #<issue>
```

- No AI attribution of any kind — no `Co-Authored-By`, no "Generated with", no
  `Signed-off-by`. `just commit-msg-check` looks for them.
- `cargo fmt` output and clippy fixes go in their own commits, separate from
  behaviour changes.
- Update `CHANGELOG.md` under `## [Unreleased]` as its own commit.
- Never commit working notes, another checkout, local agent state, or model
  weights; `.gitignore` names them. If a commit needs `git add -f`, stop:
  something is filed wrong.
- A pull request title reaches `main`'s merge commit, and GitHub reads a CI skip
  keyword there the same way it reads one in a message: the run is skipped, its
  checks stay _pending_ rather than failing, and nothing looks wrong. Write "the
  skip keyword" instead, or name the mechanism.

## Working with coding agents — the recommended pattern

What this repository tells an agent is `AGENTS.md`, at the root, in the open and
under the name other coding agents already read. Claude Code reads `CLAUDE.md`
rather than that name, so `.claude/CLAUDE.md` is committed as a symlink back to
it — the instructions sit in one file, and a clone loads them with nothing to
set up. That symlink is the only thing under `.claude/` this repository tracks;
`.gitignore` takes everything else there.

If you develop here with a coding agent, the loop that works is:

```text
issue -> just worktree-new -> implement -> just check
      -> GPU tests by hand when the change runs on the GPU
      -> review locally
      -> commit
      -> git fetch origin -> just pre-push
      -> write the PR body to a file -> hand over push/PR
```

Review the change locally before it goes up, by a person or by an AI agent —
whichever you have. Three questions are worth asking of every diff: does it do
what its issue asked, may everything in it be published
([PUBLIC_DEVELOPMENT.md](PUBLIC_DEVELOPMENT.md#agent-publication-check)), and
are the comments beside the code still true. Say in the PR body who or what
reviewed it, or that nobody did.

Releases and publishes (crates, the repository's own settings) are performed by
a human hand.

## Pull requests

An agent's part ends with everything that does not write to anything remote, and
that includes the gate:

1. `git fetch origin`, then `just pre-push` — the agent runs both. `pre-push` is
   `branch-check`, `commit-msg-check` and `check`. None of them writes to
   anything remote, so being denied `git push` is no reason to skip it. Report
   the result.
2. **Write the PR body to a file** outside the tree, and say where it is — so
   that the command handed over can be `--body-file <path>` and the human reads
   the same bytes that will be posted. Where the file lives is your setup's
   choice; that it is a file, and not a summary in the chat, is the rule.
3. Hand over the literal commands. These two are the only ones that write to
   anything remote:

   ```bash
   git push -u origin <branch>
   gh pr create --base main --head <branch> \
     --title "<subject>" --body-file <path>
   ```

The PR body records what changed, what was verified — the GPU tests and
measurements included, or a statement that they were not run — and what it
deliberately does not cover, under the same disclosure policy as everything
else.
