use infer_core::{Error, MediaId, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
    Extension(String),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaInput {
    pub id: MediaId,
    pub modality: Modality,
    pub content_digest: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaStage {
    Fetch,
    Decode,
    Preprocess,
    Encode,
    Fusion,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaNode {
    pub id: u32,
    pub stage: MediaStage,
    pub inputs: Vec<u32>,
    pub modality: Modality,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalGraph {
    pub nodes: Vec<MediaNode>,
}
impl MultimodalGraph {
    /// Serialized order is topological; future or duplicate dependencies are invalid.
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            if node.inputs.iter().any(|input| !seen.contains(input)) || !seen.insert(node.id) {
                return Err(Error::invalid(
                    "multimodal graph is not a unique topological ordering",
                ));
            }
        }
        Ok(())
    }
}
