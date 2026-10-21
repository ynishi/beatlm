//! The page the host writes into: the canvas the render pass draws on, and a
//! `<pre>` the report goes into as plain text, one fact per line.
//!
//! Everything written to the page is also written to the console, so a run
//! can be read from either. The page is written at the start of a run and at
//! its end, never while frames run: changing the document inside a frame would
//! put layout work into the time being measured.

use anyhow::{Context, Result};
use wasm_bindgen::{JsCast, JsValue};

pub use crate::report::one_line;

/// The `id` of the canvas `index.html` provides.
const CANVAS_ID: &str = "beatlm-canvas";
/// The `id` of the `<pre>` the report goes into.
const REPORT_ID: &str = "beatlm-report";

/// The page's window.
pub fn window() -> Result<web_sys::Window> {
    web_sys::window().context("no window: the host runs on a page's main thread")
}

/// The canvas the render pass draws on.
pub fn canvas() -> Result<web_sys::HtmlCanvasElement> {
    window()?
        .document()
        .context("the window has no document")?
        .get_element_by_id(CANVAS_ID)
        .with_context(|| format!("the page has no element with id {CANVAS_ID:?}"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .ok()
        .with_context(|| format!("the element {CANVAS_ID:?} is not a canvas"))
}

/// Replaces the report with `text`, and logs it.
pub fn show(text: &str) {
    web_sys::console::log_1(&JsValue::from_str(text));
    if let Some(report) = report() {
        report.set_text_content(Some(text));
    }
}

/// Logs `line` as an error, without touching the page.
pub fn log_error(line: &str) {
    web_sys::console::error_1(&JsValue::from_str(line));
}

/// Appends `line` to the report, and logs it as an error.
pub fn append_error(line: &str) {
    log_error(line);
    if let Some(report) = report() {
        let mut text = report.text_content().unwrap_or_default();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(line);
        text.push('\n');
        report.set_text_content(Some(&text));
    }
}

/// Reports that the run stopped on `error`: an `error:` line, then
/// `status: failed`, so a reader waiting for a status line sees one.
pub fn fail(error: &anyhow::Error) {
    append_error(&format!("error: {}", one_line(&format!("{error:#}"))));
    append_error("status: failed");
}

/// The `<pre>` the report goes into, if the page has one.
fn report() -> Option<web_sys::Element> {
    web_sys::window()?.document()?.get_element_by_id(REPORT_ID)
}
