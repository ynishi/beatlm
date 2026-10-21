#!/usr/bin/env bash
#
# Fetches the Depth Anything V2 Small weights into models/depth-anything-v2-small/.
# Run it with `just setup-depth-anything`. It is never run by `just check`, by
# the build, or by a test; the tests that need its output are #[ignore]d and
# say so.
#
# Source
#   Hugging Face repository depth-anything/Depth-Anything-V2-Small-hf, the
#   `transformers` conversion of the Depth Anything V2 Small model by Lihe
#   Yang et al. (https://arxiv.org/abs/2406.09414), fetched at the commit
#   pinned below:
#     https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
#   Files fetched: model.safetensors (the weights) and config.json (the
#   architecture), nothing else.
#
# Terms
#   Apache License, Version 2.0, as declared in the model card's metadata
#   (`license: apache-2.0`) at the pinned commit:
#     https://www.apache.org/licenses/LICENSE-2.0
#   Only this Small model is Apache-2.0. The Base, Large and Giant models of
#   the same family are CC-BY-NC-4.0 (non-commercial); this script must never
#   be pointed at them, and the repository name below is the check.
#   The upstream repository carries no LICENSE or NOTICE file at the pinned
#   commit, so there is no upstream NOTICE text to preserve.
#
# Redistribution
#   This repository does not redistribute these files: they are downloaded
#   into the gitignored models/ directory on request and never committed, and
#   this script holds only their address and checksum. Fetching them does not
#   remove the licence's terms on using them.
#   Whoever does redistribute the weights (or a work derived from them) must,
#   under section 4 of the Apache License 2.0: give every recipient a copy of
#   the licence; mark any file they changed with a prominent notice of the
#   change; keep, in any derivative work, the copyright, patent, trademark and
#   attribution notices of the source; and, if the work ships a NOTICE file,
#   carry its attribution notices along. The licence grants no right to the
#   authors' trade names or marks (section 6) and provides the work AS IS,
#   without warranty (section 7).
#
# Pinned values (obtained 2026-10-21 from the Hugging Face API, read-only)
#   REVISION      the commit of the repository's main branch:
#                   GET https://huggingface.co/api/models/<repo>
#                   (field "sha"); the tree at that commit:
#                   GET https://huggingface.co/api/models/<repo>/tree/<REVISION>
#                   (field lfs.oid of model.safetensors is its sha256).
#   MODEL_SHA256  sha256 of model.safetensors, 99173660 bytes.
#   CONFIG_SHA256 sha256 of config.json, 950 bytes; computed from the file at
#                 that commit, since the tree lists only a git blob id for it.
#   To move to a newer commit, change these three together and re-read the
#   licence on the model card.
#
# The script refuses to run while any pinned value is empty, verifies every
# file against its sha256 before it is put in place, and skips a file that is
# already there and verified, so it can be run again at no cost.

set -euo pipefail

REPO="depth-anything/Depth-Anything-V2-Small-hf"
REVISION="5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_SHA256="3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
CONFIG_SHA256="c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75"

for pinned in REPO REVISION MODEL_SHA256 CONFIG_SHA256; do
    if [ -z "${!pinned}" ]; then
        echo "setup-depth-anything: $pinned is empty; set the pinned values at the top of this script" >&2
        exit 1
    fi
done

# The Apache-2.0 repository is the only one this script may fetch.
if [ "$REPO" != "depth-anything/Depth-Anything-V2-Small-hf" ]; then
    echo "setup-depth-anything: $REPO is not the Apache-2.0 Small model; other sizes are CC-BY-NC-4.0" >&2
    exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/depth-anything-v2-small"
base="https://huggingface.co/$REPO/resolve/$REVISION"

# Checked before anything is fetched, so a missing tool does not cost a download.
if command -v sha256sum >/dev/null 2>&1; then
    sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    echo "setup-depth-anything: neither sha256sum nor shasum is installed" >&2
    exit 1
fi

# fetch <file name> <expected sha256>
fetch() {
    local name="$1" expected="$2" target="$dest/$1" part="$dest/$1.part"
    if [ -f "$target" ] && [ "$(sha256_of "$target")" = "$expected" ]; then
        echo "setup-depth-anything: $name already present and verified"
        return 0
    fi
    echo "setup-depth-anything: fetching $name at $REVISION"
    rm -f "$part"
    curl --fail --location --silent --show-error --retry 3 \
        --output "$part" "$base/$name"
    local actual
    actual="$(sha256_of "$part")"
    if [ "$actual" != "$expected" ]; then
        rm -f "$part"
        echo "setup-depth-anything: $name has sha256 $actual, expected $expected; nothing was kept" >&2
        return 1
    fi
    mv "$part" "$target"
    echo "setup-depth-anything: $name verified"
}

mkdir -p "$dest"
fetch config.json "$CONFIG_SHA256"
fetch model.safetensors "$MODEL_SHA256"
echo "setup-depth-anything: ready in $dest (Apache-2.0; not redistributed by this repository)"
