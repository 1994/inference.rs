//! Fixtures CLI support.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_core::{ModelId, RequestId, Result};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_ir::{CanonicalRequest, Qos, RequestInput, Sampling, Workload};
#[cfg(feature = "test-backends")]
use infer_ir::{DecisionQuestion, DecisionSchema, Pooling};

/// Token identifiers shared by every example request's input sequence.
const EXAMPLE_TOKENS: [u32; 5] = [1, 2, 3, 4, 5];
/// Request identifier of the rerank example.
#[cfg(feature = "test-backends")]
const RERANK_REQUEST_ID: u64 = 3;
/// Request identifier of the decision example.
#[cfg(feature = "test-backends")]
const DECISION_REQUEST_ID: u64 = 4;
/// Token identifiers of the first rerank example document.
#[cfg(feature = "test-backends")]
const RERANK_DOCUMENT_HEAD_TOKENS: [u32; 3] = [3, 4, 5];
/// Token identifiers of the second rerank example document.
#[cfg(feature = "test-backends")]
const RERANK_DOCUMENT_BODY_TOKENS: [u32; 2] = [6, 7];
/// Token identifiers of the third rerank example document.
#[cfg(feature = "test-backends")]
const RERANK_DOCUMENT_TAIL_TOKENS: [u32; 1] = [8];
/// Generated token count of the generate example.
#[cfg(feature = "test-backends")]
const GENERATE_EXAMPLE_TOKENS: usize = 8;
/// Requested embedding dimensions of the embed example.
#[cfg(feature = "test-backends")]
const EMBED_EXAMPLE_DIMENSIONS: usize = 4;
/// Categorical option tokens of the decision example.
#[cfg(feature = "test-backends")]
const CATEGORICAL_OPTIONS: [u32; 3] = [2, 3, 4];
/// Ordinal option tokens of the decision example.
#[cfg(feature = "test-backends")]
const ORDINAL_OPTIONS: [u32; 3] = [1, 2, 3];
/// Abstention threshold of the decision example.
#[cfg(feature = "test-backends")]
const DECISION_ABSTAIN_BELOW: f32 = 0.8;

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn example(id: u64, workload: Workload) -> Result<CanonicalRequest> {
    Ok(CanonicalRequest {
        id: RequestId::new(id)?,
        model: ModelId::new(1)?,
        session: None,
        input: RequestInput::Sequence {
            tokens: EXAMPLE_TOKENS.to_vec().into(),
            media: vec![],
        },
        workload,
        qos: Qos::default(),
        sampling: Sampling::default(),
        extensions: std::collections::BTreeMap::new(),
    })
}
#[cfg(feature = "test-backends")]
pub fn examples() -> Result<Vec<CanonicalRequest>> {
    let mut rerank = example(RERANK_REQUEST_ID, Workload::Rerank { top_k: 2 })?;
    rerank.input = RequestInput::Pairs {
        query: vec![1, 2].into(),
        documents: vec![
            RERANK_DOCUMENT_HEAD_TOKENS.to_vec().into(),
            RERANK_DOCUMENT_BODY_TOKENS.to_vec().into(),
            RERANK_DOCUMENT_TAIL_TOKENS.to_vec().into(),
        ]
        .into(),
    };
    Ok(vec![
        example(
            1,
            Workload::Generate {
                max_new_tokens: GENERATE_EXAMPLE_TOKENS,
            },
        )?,
        example(
            2,
            Workload::Embed {
                pooling: Pooling::Mean,
                dimensions: Some(EMBED_EXAMPLE_DIMENSIONS),
                normalize: true,
            },
        )?,
        rerank,
        example(
            DECISION_REQUEST_ID,
            Workload::Decision(DecisionSchema {
                questions: vec![
                    DecisionQuestion::Binary {
                        negative_token: 0,
                        positive_token: 1,
                    },
                    DecisionQuestion::Categorical {
                        options: CATEGORICAL_OPTIONS.to_vec(),
                    },
                    DecisionQuestion::Ordinal {
                        options: ORDINAL_OPTIONS.to_vec(),
                        values: vec![0.0, 1.0, 2.0],
                    },
                    DecisionQuestion::Continuous {
                        token: 0,
                        min: 0.0,
                        max: 1.0,
                    },
                ],
                calibration_temperature: 1.0,
                abstain_below: Some(DECISION_ABSTAIN_BELOW),
            }),
        )?,
    ])
}
