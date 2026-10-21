"""Tests of scripts/serve_web.py: what a path maps to, and what is refused.

Run with `just scripts-test`. They build their own two roots in a temporary
directory, so they read no model and need no build.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import serve_web


class Roots(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        base = Path(self.scratch.name)
        self.page = base / "page"
        self.models = base / "models"
        self.outside = base / "secret.txt"
        (self.page / "sub").mkdir(parents=True)
        (self.models / "m" / "reference" / "266").mkdir(parents=True)
        (self.page / "index.html").write_text("<p>page</p>")
        (self.page / "beatlm_web.js").write_text("export {}")
        (self.page / "beatlm_web_bg.wasm").write_bytes(b"\0asm")
        (self.page / ".hidden").write_text("no")
        (self.models / "m" / "model.safetensors").write_bytes(b"weights")
        (self.models / "m" / "reference" / "266" / "manifest.json").write_text("{}")
        self.outside.write_text("outside")
        (self.page / "escape").symlink_to(self.outside)
        (self.models / "m" / "escape").symlink_to(self.page / "index.html")

    def tearDown(self) -> None:
        self.scratch.cleanup()

    def resolve(self, path: str) -> Path | None:
        return serve_web.resolve(path, self.page, self.models)


class ResolveTests(Roots):
    def test_the_page_is_at_the_root_and_index_html_is_its_slash(self) -> None:
        index = (self.page / "index.html").resolve()
        self.assertEqual(self.resolve("/"), index)
        self.assertEqual(self.resolve("/index.html"), index)
        self.assertEqual(self.resolve("/index.html?frames=600&model=depth"), index)
        self.assertEqual(self.resolve("/?model=depth&side=266"), index)
        self.assertEqual(
            self.resolve("/beatlm_web_bg.wasm"), (self.page / "beatlm_web_bg.wasm").resolve()
        )

    def test_the_models_are_under_their_prefix(self) -> None:
        self.assertEqual(
            self.resolve("/models/m/model.safetensors"),
            (self.models / "m" / "model.safetensors").resolve(),
        )
        self.assertEqual(
            self.resolve("/models/m/reference/266/manifest.json"),
            (self.models / "m" / "reference" / "266" / "manifest.json").resolve(),
        )
        # The page's tree has no models, and the models' tree no page.
        self.assertIsNone(self.resolve("/m/model.safetensors"))
        self.assertIsNone(self.resolve("/models/index.html"))

    def test_a_traversal_is_refused_however_it_is_spelled(self) -> None:
        for path in [
            "/../secret.txt",
            "/models/../page/index.html",
            "/models/m/../../page/index.html",
            "/%2e%2e/secret.txt",
            "/%2E%2E%2Fsecret.txt",
            "/models/%2e%2e/page/index.html",
            "/sub/..%2f..%2fsecret.txt",
            "/./index.html",
            "/sub/%2e/../index.html",
            "/..\\secret.txt",
            "/%5c..%5csecret.txt",
            "/index.html%00.js",
            "index.html",
            "//index.html",
        ]:
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_a_link_out_of_its_root_is_refused(self) -> None:
        self.assertIsNone(self.resolve("/escape"))
        self.assertIsNone(self.resolve("/models/m/escape"))

    def test_directories_hidden_files_and_missing_files_are_not_served(self) -> None:
        for path in ["/models/", "/models", "/models/m", "/models/m/", "/sub", "/sub/",
                     "/.hidden", "/missing.js", "/models/m/missing.f32"]:
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_the_module_has_the_type_streaming_compilation_needs(self) -> None:
        self.assertEqual(serve_web.content_type(Path("a.wasm")), "application/wasm")
        self.assertTrue(serve_web.content_type(Path("a.js")).startswith("text/javascript"))
        self.assertEqual(
            serve_web.content_type(Path("model.safetensors")), "application/octet-stream"
        )


class ServerTests(Roots):
    def setUp(self) -> None:
        super().setUp()
        handler = serve_web.handler(self.page, self.models)
        # The request log is the server's to the person running it, not the
        # test's output.
        handler.log_message = lambda *args: None
        self.server = ThreadingHTTPServer((serve_web.HOST, 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://{serve_web.HOST}:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        super().tearDown()

    def status(self, path: str, method: str = "GET") -> int:
        request = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_files_are_served_and_the_rest_is_404(self) -> None:
        with urllib.request.urlopen(self.base + "/models/m/model.safetensors") as response:
            self.assertEqual(response.read(), b"weights")
            self.assertEqual(response.headers["Content-Type"], "application/octet-stream")
        self.assertEqual(self.status("/"), 200)
        self.assertEqual(self.status("/", method="HEAD"), 200)
        self.assertEqual(self.status("/models/"), 404)
        self.assertEqual(self.status("/%2e%2e/secret.txt"), 404)

    def test_it_is_read_only(self) -> None:
        for method in ["POST", "PUT", "DELETE"]:
            with self.subTest(method=method):
                self.assertEqual(self.status("/index.html", method=method), 501)


if __name__ == "__main__":
    unittest.main()
