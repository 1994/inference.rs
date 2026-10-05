use crate::Head;
use infer_core::{Error, ModelId, ProgramId, RequestId, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pooling {
    Mean,
    Last,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionQuestion {
    Binary {
        negative_token: u32,
        positive_token: u32,
    },
    Categorical {
        options: Vec<u32>,
    },
    Ordinal {
        options: Vec<u32>,
        values: Vec<f32>,
    },
    Continuous {
        token: u32,
        min: f32,
        max: f32,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionSchema {
    pub questions: Vec<DecisionQuestion>,
    pub calibration_temperature: f32,
    pub abstain_below: Option<f32>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Workload {
    Generate {
        max_new_tokens: usize,
    },
    Embed {
        pooling: Pooling,
        dimensions: Option<usize>,
        normalize: bool,
    },
    Rerank {
        top_k: usize,
    },
    Decision(DecisionSchema),
    Classify {
        classes: usize,
    },
    Reward,
    LateInteraction,
    Extension {
        provider: String,
        payload: Vec<u8>,
    },
}
impl Workload {
    #[must_use]
    pub const fn head(&self) -> Option<Head> {
        Some(match self {
            Self::Generate { .. } => Head::LanguageModel,
            Self::Embed { .. } => Head::Embedding,
            Self::Rerank { .. } => Head::Rank,
            Self::Decision(_) => Head::Decision,
            Self::Classify { .. } => Head::Classification,
            Self::Reward => Head::Reward,
            Self::LateInteraction => Head::LateInteraction,
            Self::Extension { .. } => return None,
        })
    }
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Generate { max_new_tokens: 0 }
            | Self::Embed {
                dimensions: Some(0),
                ..
            }
            | Self::Rerank { top_k: 0 }
            | Self::Classify { classes: 0 } => {
                return Err(Error::invalid("workload size must be positive"));
            }
            Self::Decision(schema) => {
                if schema.questions.is_empty()
                    || !schema.calibration_temperature.is_finite()
                    || schema.calibration_temperature <= 0.0
                    || schema
                        .abstain_below
                        .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
                {
                    return Err(Error::invalid("invalid decision calibration/schema"));
                }
                for question in &schema.questions {
                    match question {
                        DecisionQuestion::Binary {
                            negative_token,
                            positive_token,
                        } if negative_token == positive_token => {
                            return Err(Error::invalid("binary answers must differ"));
                        }
                        DecisionQuestion::Categorical { options }
                        | DecisionQuestion::Ordinal { options, .. } => {
                            if options.len() < 2 {
                                return Err(Error::invalid(
                                    "at least two decision options required",
                                ));
                            }
                            let mut unique = options.clone();
                            unique.sort_unstable();
                            unique.dedup();
                            if unique.len() != options.len() {
                                return Err(Error::invalid("duplicate decision options"));
                            }
                        }
                        DecisionQuestion::Continuous { min, max, .. }
                            if !min.is_finite() || !max.is_finite() || min >= max =>
                        {
                            return Err(Error::invalid("invalid continuous decision bounds"));
                        }
                        _ => {}
                    }
                    if let DecisionQuestion::Ordinal { options, values } = question
                        && (options.len() != values.len()
                            || values.iter().any(|v| !v.is_finite())
                            || values.windows(2).any(|w| w[0] >= w[1]))
                    {
                        return Err(Error::invalid(
                            "ordinal values must match options and increase",
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadPlan {
    pub request: RequestId,
    pub model: ModelId,
    pub program: ProgramId,
    pub units: Vec<crate::TokenBuffer>,
    pub reserved_tokens: usize,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeaturePlan {
    pub speculation: Option<SpeculationPlan>,
    pub grammar: Option<String>,
    pub adapter: Option<String>,
    pub prefix_reuse: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationPlan {
    pub proposal: ProgramId,
    pub verification: ProgramId,
    pub hidden_state_taps: Vec<usize>,
    pub max_candidates: usize,
    pub acceptance_policy: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionAnswer {
    pub probabilities: Vec<f32>,
    pub expected: Option<f32>,
    pub abstained: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedDocument {
    pub index: usize,
    pub score: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkloadOutput {
    Tokens(crate::TokenBuffer),
    Embedding(crate::SharedOutput<f32>),
    Ranking(crate::SharedOutput<RankedDocument>),
    Decisions(crate::SharedOutput<DecisionAnswer>),
}

impl WorkloadOutput {
    pub fn attach_credit(&mut self, credit: infer_core::credits::CreditLease) {
        match self {
            Self::Tokens(tokens) => tokens.attach_byte_credit(credit),
            Self::Embedding(values) => values.attach_credit(credit),
            Self::Ranking(values) => values.attach_credit(credit),
            Self::Decisions(values) => values.attach_credit(credit),
        }
    }
}
