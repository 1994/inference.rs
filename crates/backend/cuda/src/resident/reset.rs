//! In-place recurrent clearing uses the audited device submission boundary.
pub(super) use crate::device::clear::Clear;

#[cfg(test)]
#[path = "../../tests/unit/resident_reset.rs"]
mod tests;
