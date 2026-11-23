//! Liquify smart filters (`filter.liquify` on a smart object) in PSD.
//!
//! Photoshop stores its own Liquify as a proprietary mesh, which PhotoCraft neither reads nor
//! writes. Without a mapping the filter used to be left out of the file, so Liquify applied to a
//! smart object was lost on save. It is now written as PhotoCraft's own filter item: the
//! command's params (the strokes and `meshSize`), as UTF-8 JSON, in a `Fltr` of a class no other
//! app uses:
//!
//! ```text
//! filterFX: Nm = "Liquify...", blendOptions, enab, hasoptions = true, FrgC, BckC,
//!   Fltr = photocraftLiquify { params = raw data (JSON) }, filterID = 'pcLq'
//! ```
//!
//! The layer's pixels in the file are already the filtered result, so other apps show the
//! liquified layer; Photoshop lists the filter as one it doesn't have. A `Fltr` that isn't
//! exactly this (a damaged one, or JSON that isn't an object) stays an opaque
//! [`super::UNSUPPORTED_FILTER`] and is written back as it was.

use photocraft_doc::SmartFilter;
use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value};
use serde_json::Value as J;

use super::{blend_options, rgbc};

pub(super) const CLASS: &str = "photocraftLiquify";
pub(super) const COMMAND: &str = "filter.liquify";
const FILTER_ID: i32 = super::code(b"pcLq");
/// `Nm  `, as the menu item.
const NAME: &str = "Liquify...";
const PARAMS: &str = "params";

/// The command's params from a `photocraftLiquify` item; `None` unless it is exactly the
/// written layout with a JSON object as its params.
pub(super) fn import_params(item: &Descriptor) -> Option<J> {
    let Some(Value::Descriptor(f)) = item.get("Fltr") else { return None };
    if !f.class_id.is(CLASS) || f.items.len() != 1 {
        return None;
    }
    let Some(Value::RawData(raw)) = f.get(PARAMS) else { return None };
    serde_json::from_slice::<J>(raw).ok().filter(J::is_object)
}

/// The `filterFXList` item for a Liquify smart filter.
pub(super) fn export_item(f: &SmartFilter) -> Result<Descriptor, String> {
    if !f.params.is_object() {
        return Err("its settings are missing".into());
    }
    let json = serde_json::to_vec(&f.params).map_err(|e| format!("its settings can't be stored ({e})"))?;
    let fltr = Descriptor::new(CLASS).with(PARAMS, Value::RawData(json));
    Ok(Descriptor::new("filterFX")
        .with("Nm  ", Value::Text(UnicodeString::new_nul(NAME)))
        .with("blendOptions", blend_options(f))
        .with("enab", Value::Boolean(f.visible))
        .with("hasoptions", Value::Boolean(true))
        .with("FrgC", rgbc(0.0))
        .with("BckC", rgbc(255.0))
        .with("Fltr", Value::Descriptor(fltr))
        .with("filterID", Value::Integer(FILTER_ID)))
}
