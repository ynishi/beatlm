//! Files from the page's server, by `fetch`.
//!
//! The weights and the references are served by `just serve-web` under
//! [`MODELS_URL`](crate::depth::MODELS_URL) from the models directory, never
//! from the build output. A file the server does not have is `None`, so the
//! caller decides whether its absence fails the run (the weights) or is
//! reported and skipped (a reference).

use anyhow::{Result, anyhow, bail};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::page;

/// The response for `url`; `None` when the server answers 404.
async fn response(url: &str) -> Result<Option<web_sys::Response>> {
    let value = JsFuture::from(page::window()?.fetch_with_str(url))
        .await
        .map_err(|error| anyhow!("cannot fetch {url}: {error:?}"))?;
    let response: web_sys::Response = value
        .dyn_into()
        .map_err(|_| anyhow!("fetching {url} returned something that is not a response"))?;
    if response.status() == 404 {
        return Ok(None);
    }
    if !response.ok() {
        bail!("fetching {url}: the server answered {}", response.status());
    }
    Ok(Some(response))
}

/// The bytes of `url`; `None` when the server has no such file.
///
/// The body arrives as one `ArrayBuffer` and is copied once into the module's
/// memory, which is the buffer [`beatlm::load::Archive::parse`] borrows from.
pub async fn bytes(url: &str) -> Result<Option<Vec<u8>>> {
    let Some(response) = response(url).await? else {
        return Ok(None);
    };
    let promise = response
        .array_buffer()
        .map_err(|error| anyhow!("cannot read the body of {url}: {error:?}"))?;
    let buffer = JsFuture::from(promise)
        .await
        .map_err(|error| anyhow!("cannot read the body of {url}: {error:?}"))?;
    Ok(Some(js_sys::Uint8Array::new(&buffer).to_vec()))
}

/// The text of `url`; `None` when the server has no such file.
pub async fn text(url: &str) -> Result<Option<String>> {
    let Some(response) = response(url).await? else {
        return Ok(None);
    };
    let promise = response
        .text()
        .map_err(|error| anyhow!("cannot read the body of {url}: {error:?}"))?;
    let text = JsFuture::from(promise)
        .await
        .map_err(|error| anyhow!("cannot read the body of {url}: {error:?}"))?;
    text.as_string()
        .map(Some)
        .ok_or_else(|| anyhow!("the body of {url} is not text"))
}
