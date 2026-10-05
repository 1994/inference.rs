use crate::{MediaInput, Workload};
use infer_core::{Error, ModelId, RequestId, Result, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestInput {
    Sequence {
        tokens: crate::TokenBuffer,
        #[serde(default)]
        media: Vec<MediaInput>,
    },
    Pairs {
        query: crate::TokenBuffer,
        documents: crate::SharedOutput<crate::TokenBuffer>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Qos {
    pub tenant: String,
    pub weight: u32,
    /// Absolute deadline in the caller-provided monotonic microsecond clock.
    pub deadline_us: Option<u64>,
    pub ttft_slo_us: Option<u64>,
    pub tpot_slo_us: Option<u64>,
}
impl Default for Qos {
    fn default() -> Self {
        Self {
            tenant: "default".into(),
            weight: 1,
            deadline_us: None,
            ttft_slo_us: None,
            tpot_slo_us: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub eos_token: Option<u32>,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            seed: 0,
            eos_token: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalRequest {
    pub id: RequestId,
    pub model: ModelId,
    #[serde(default)]
    pub session: Option<SessionId>,
    pub input: RequestInput,
    pub workload: Workload,
    #[serde(default)]
    pub qos: Qos,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub extensions: BTreeMap<String, Vec<u8>>,
}
impl CanonicalRequest {
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.qos.tenant.is_empty() || self.qos.tenant.len() > 256 || self.qos.weight == 0 {
            return Err(Error::invalid("tenant and positive weight required"));
        }
        if self.qos.ttft_slo_us == Some(0)
            || self.qos.tpot_slo_us == Some(0)
            || (!matches!(self.workload, Workload::Generate { .. })
                && (self.qos.ttft_slo_us.is_some() || self.qos.tpot_slo_us.is_some()))
        {
            return Err(Error::invalid(
                "positive TTFT/TPOT SLOs require a generate workload",
            ));
        }
        if !self.sampling.temperature.is_finite()
            || self.sampling.temperature < 0.0
            || self.sampling.top_k == Some(0)
        {
            return Err(Error::invalid("invalid sampling parameters"));
        }
        if !self.extensions.is_empty() {
            return Err(Error::unsupported(
                "no feature provider registered for extension fields",
            ));
        }
        match &self.input {
            RequestInput::Sequence { tokens, media } => {
                if tokens.is_empty() && media.is_empty() {
                    return Err(Error::invalid("empty input"));
                }
                if matches!(self.workload, Workload::Rerank { .. }) {
                    return Err(Error::invalid("rerank requires query/document pairs"));
                }
            }
            RequestInput::Pairs { query, documents } => {
                if query.is_empty()
                    || documents.is_empty()
                    || documents.iter().any(|doc| doc.is_empty())
                {
                    return Err(Error::invalid("empty query/document"));
                }
                if !matches!(self.workload, Workload::Rerank { .. }) {
                    return Err(Error::invalid("pair input requires rerank workload"));
                }
            }
        }
        self.workload.validate()
    }
}
