#!/usr/bin/env python3
"""Serves the browser host's page and the models directory, read-only, on 127.0.0.1.

Run it with `just serve-web`, after `just build-web`. Two roots, each under its
own URL prefix, from one origin:

    /          the built page: --page (target/beatlm-web/), index.html for "/"
    /models/   the models directory: --models (BEATLM_MODELS_DIR, or models/
               at the workspace root, the rule the native tests read it by)

One origin because the page fetches the weights and the references with
`fetch`, and a second origin would need cross-origin headers for nothing. The
models are served from where `just setup-depth-anything` and `just
reference-depth-anything` put them, never copied into the build output: they
are third-party material that is not redistributed (PUBLIC_DEVELOPMENT.md,
"Third-party material"), and a page that bundled them would be redistribution
the moment it was published.

What it refuses, so that only those two trees are ever read:

- every method but GET and HEAD (the standard library answers 501);
- a path with a "..", ".", or empty segment, a backslash or a NUL, before or
  after percent-decoding, and a segment starting with "." (no hidden files);
- anything that is not a regular file: a directory is 404, and there is no
  listing of one;
- a file whose real path, symbolic links resolved, is not inside its root.

It binds 127.0.0.1 and nothing else: the page is for the machine it runs on.
The standard library only, so it adds no dependency to the repository; the
page needs no cross-origin isolation headers (the crate documentation of
`beatlm-web` says why), so a plain static server is enough.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HOST = "127.0.0.1"
MODELS_PREFIX = "/models/"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    # `WebAssembly.instantiateStreaming` requires exactly this type.
    ".wasm": "application/wasm",
    ".json": "application/json",
}


def resolve(url_path: str, page: Path, models: Path) -> Path | None:
    """The file a request path names, or None when it names none it may serve.

    `url_path` is the request target as it arrived; the query and fragment are
    ignored. `page` and `models` are the two roots.
    """
    path = urlsplit(url_path).path
    for text in (path, unquote(path)):
        if "\\" in text or "\x00" in text:
            return None
    path = unquote(path)
    if not path.startswith("/"):
        return None
    if path == "/":
        root, rest = page, "index.html"
    elif path.startswith(MODELS_PREFIX):
        root, rest = models, path[len(MODELS_PREFIX):]
    else:
        root, rest = page, path[1:]
    segments = rest.split("/")
    if any(s in ("", ".", "..") or s.startswith(".") for s in segments):
        return None
    try:
        real_root = root.resolve(strict=True)
        real = root.joinpath(*segments).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if real_root != real and real_root not in real.parents:
        return None
    if not real.is_file():
        return None
    return real


def content_type(path: Path) -> str:
    """The Content-Type of a served file, by its extension."""
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def handler(page: Path, models: Path) -> type[BaseHTTPRequestHandler]:
    """A request handler serving `page` at / and `models` at /models/."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "beatlm-serve-web"

        def do_GET(self) -> None:  # noqa: N802 (the standard library's name)
            self.send(body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self.send(body=False)

        def send(self, body: bool) -> None:
            path = resolve(self.path, page, models)
            if path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                file = path.open("rb")
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            with file:
                size = path.stat().st_size
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type(path))
                self.send_header("Content-Length", str(size))
                # A rebuilt module must not be served from a stale cache.
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if body:
                    try:
                        shutil.copyfileobj(file, self.wfile)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

    return Handler


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", type=Path, required=True, help="the built page's directory")
    parser.add_argument("--models", type=Path, required=True, help="the models directory")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not (args.page / "index.html").is_file():
        print(f"serve-web: no index.html in {args.page}; run 'just build-web'", file=sys.stderr)
        return 1
    if not args.models.is_dir():
        print(
            f"serve-web: no models directory at {args.models}; the synthetic workload runs, the "
            "depth model does not. Run 'just setup-depth-anything', or set BEATLM_MODELS_DIR.",
            file=sys.stderr,
        )
    server = ThreadingHTTPServer((HOST, args.port), handler(args.page, args.models))
    print(f"serve-web: {args.page} at http://{HOST}:{args.port}/, {args.models} at {MODELS_PREFIX}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
