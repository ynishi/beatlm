# `check` is the definition of green. CI runs it and nothing else, so a gate
# added here is picked up there without a second edit.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Everything a change owes before it is handed over.
check: rust-fmt-check rust-clippy rust-test wasm-check deny

rust-fmt-check:
    cargo fmt --all -- --check

rust-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

rust-test:
    cargo test --workspace

# The browser is a first-class target: the umbrella must build for it on
# every change, not when someone gets round to the demo.
wasm-check:
    cargo clippy -p beatlm --target wasm32-unknown-unknown -- -D warnings

deny:
    cargo deny check

# One worktree per issue, cut from a freshly fetched origin/main.
worktree-new type slug:
    git fetch origin
    git worktree add .worktrees/{{slug}} -b {{type}}/{{slug}} origin/main
    cd .worktrees/{{slug}} && just branch-check

# Refuses main, and a branch that does not contain origin/main.
branch-check:
    #!/usr/bin/env bash
    set -euo pipefail
    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$branch" = "main" ]; then
        echo "branch-check: on main; cut a branch with 'just worktree-new'" >&2
        exit 1
    fi
    if git rev-parse --verify --quiet origin/main >/dev/null; then
        git merge-base --is-ancestor origin/main HEAD || {
            echo "branch-check: $branch does not contain origin/main; rebase or recut" >&2
            exit 1
        }
    fi
    echo "branch-check: $branch ok"

# Commit messages on this branch carry no AI attribution (CONTRIBUTING.md).
commit-msg-check:
    #!/usr/bin/env bash
    set -euo pipefail
    range="origin/main..HEAD"
    git rev-parse --verify --quiet origin/main >/dev/null || range="HEAD"
    if git log --format=%B "$range" | grep -inE '^(co-authored-by|signed-off-by):|generated with'; then
        echo "commit-msg-check: attribution lines found in $range" >&2
        exit 1
    fi
    echo "commit-msg-check: $range ok"

# The gate before a hand-over: run it after the last commit, over the tree
# that is actually handed over.
pre-push: branch-check commit-msg-check check
