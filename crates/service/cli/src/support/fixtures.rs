//! Fixtures CLI support.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_core::{ModelId, RequestId, Result};
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_ir::{CanonicalRequest, Qos, RequestInput, Sampling, Workload};
#[cfg(feature = "test-backends")]
use infer_ir::{DecisionQuestion, DecisionSchema, Pooling};

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn example(id: u64, workload: Workload) -> Result<CanonicalRequest> {
    Ok(CanonicalRequest {
        id: RequestId::new(id)?,
        model: ModelId::new(1)?,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2, 3, 4, 5].into(),
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
    let mut rerank = example(3, Workload::Rerank { top_k: 2 })?;
    rerank.input = RequestInput::Pairs {
        query: vec![1, 2].into(),
        documents: vec![vec![3, 4, 5].into(), vec![6, 7].into(), vec![8].into()].into(),
    };
    Ok(vec![
        example(1, Workload::Generate { max_new_tokens: 8 })?,
        example(
            2,
            Workload::Embed {
                pooling: Pooling::Mean,
                dimensions: Some(4),
                normalize: true,
            },
        )?,
        rerank,
        example(
            4,
            Workload::Decision(DecisionSchema {
                questions: vec![
                    DecisionQuestion::Binary {
                        negative_token: 0,
                        positive_token: 1,
                    },
                    DecisionQuestion::Categorical {
                        options: vec![2, 3, 4],
                    },
                    DecisionQuestion::Ordinal {
                        options: vec![1, 2, 3],
                        values: vec![0.0, 1.0, 2.0],
                    },
                    DecisionQuestion::Continuous {
                        token: 0,
                        min: 0.0,
                        max: 1.0,
                    },
                ],
                calibration_temperature: 1.0,
                abstain_below: Some(0.8),
            }),
        )?,
    ])
}
