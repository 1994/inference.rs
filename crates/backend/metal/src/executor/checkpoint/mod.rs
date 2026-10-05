//! Checkpoint responsibilities.
mod capture;
mod restore;
mod validation;
use super::{Checkpoint, MetalBackend, MetalDevice, RestoredBlock, SavedSequence, Sequence};
