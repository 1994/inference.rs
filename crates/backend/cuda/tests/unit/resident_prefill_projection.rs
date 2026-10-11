use super::{FULL_LOGITS, gemm, last_row};
use crate::{device::CudaDevice, resident::ActivationArena};
use cutile::{half::bf16, prelude::*};
use infer_core::{ModelId, OpId, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};

fn graph(vocabulary: usize) -> infer_core::Result<DataflowGraph> {
    let mut graph = DataflowGraph::default();
    for (id, shape, weight) in [
        (1, vec![32, 32], true),
        (2, vec![32], false),
        (3, vec![32, 32], true),
        (4, vec![32], false),
        (5, vec![vocabulary, 32], true),
        (6, vec![vocabulary], false),
    ] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape,
            dtype: DType::F32,
            storage: if weight {
                TensorStorage::Weight {
                    slot: format!("w{id}"),
                }
            } else {
                TensorStorage::Activation
            },
        });
    }
    for (id, op, inputs, output) in [
        (1, TensorOp::Embedding, vec![1], 2),
        (2, TensorOp::Linear, vec![2, 3], 4),
        (3, TensorOp::Linear, vec![4, 5], 6),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op,
            inputs: inputs
                .into_iter()
                .map(TensorId::new)
                .collect::<infer_core::Result<Vec<_>>>()?,
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    graph.hidden = Some(TensorId::new(4)?);
    graph.logits = Some(TensorId::new(6)?);
    Ok(graph)
}

#[test]
fn compact_logits_storage_preserves_earlier_aliases() -> infer_core::Result<()> {
    let width = 128;
    let compact = Some(TensorId::new(6)?);
    let distinct = graph(320)?;
    assert_eq!(
        ActivationArena::required_bytes(&distinct, width)?,
        (32 + 32 + 320) * width * 4
    );
    assert_eq!(
        ActivationArena::required_prompt_bytes(&distinct, width, compact)?,
        (64 * width + 320) * 4
    );
    // The old logits slot could reuse the first activation. Compacting it must keep that
    // earlier activation full-width and allocate a separate terminal result.
    let aliased = graph(32)?;
    assert_eq!(
        ActivationArena::required_bytes(&aliased, width)?,
        64 * width * 4
    );
    assert_eq!(
        ActivationArena::required_prompt_bytes(&aliased, width, compact)?,
        (64 * width + 32) * 4
    );
    assert!(ActivationArena::required_prompt_bytes(&aliased, 0, compact).is_err());
    assert!(ActivationArena::required_prompt_bytes(&aliased, usize::MAX, compact).is_err());
    Ok(())
}

fn sample(index: usize, seed: usize) -> f32 {
    let value = u16::try_from((index * 37 + seed) % 997).expect("bounded fixture");
    (f32::from(value) - 498.0) / 997.0 + 0.000_031
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn last_valid_prompt_projection_matches_full_mma_bitwise() -> Result<(), Box<dyn std::error::Error>>
{
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (width, columns, rows) in [(64, 130, 130), (128, 256, 256), (256, 5376, 256)] {
        let input = device.upload(
            (0..width * columns).map(|i| sample(i, 5)).collect(),
            &[width, columns],
        )?;
        let weight = device.upload(
            (0..rows * columns)
                .map(|i| bf16::from_f32(sample(i, 13)))
                .collect(),
            &[rows, columns],
        )?;
        let mut full = api::zeros::<f32>(&[width, rows]).sync_on(&device.stream)?;
        gemm::dense((&mut full).partition([32, 64]), &input, &weight)
            .generics(vec![columns.to_string()])
            .sync_on(&device.stream)?;
        let expected = full.to_host_vec().sync_on(&device.stream)?;
        let mut scratch = api::zeros::<f32>(&[columns]).sync_on(&device.stream)?;
        let mut output = api::zeros::<f32>(&[1, rows]).sync_on(&device.stream)?;
        for count in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]
            .into_iter()
            .filter(|&c| c <= width)
        {
            let metadata = device.upload(vec![0, i32::try_from(count)?, 0, 0], &[4])?;
            last_row::gather((&mut scratch).partition([128]), &input, &metadata)
                .sync_on(&device.stream)?;
            gemm::dense(
                (&mut output).partition([32, 64]),
                &scratch.view(&[1, columns])?,
                &weight,
            )
            .generics(vec![columns.to_string()])
            .sync_on(&device.stream)?;
            let actual = device.read_borrowed(&output)?;
            for (col, &value) in actual.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected[(count - 1) * rows + col].to_bits(),
                    "width={width} K={columns} N={rows} count={count} col={col}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and INFER_TEST_TOKENS (JSON token array)"]
fn checkpoint_compact_prompt_logits_match_full_capture() -> infer_core::Result<()> {
    use crate::loading::{LoadOptions, LoadedModel};
    use infer_core::Error;
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path =
        std::env::var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> = serde_json::from_slice(
        &std::fs::read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty checkpoint prompt"));
    }
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 128,
            verification_width: 3,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let capacity = (tokens.len() + 32).next_power_of_two();
    let mut candidate = loaded.sequence(capacity)?;
    FULL_LOGITS.set(true);
    let reference = loaded.sequence(capacity);
    FULL_LOGITS.set(false);
    let mut reference = reference?;
    assert!(
        candidate.reclaimable_bytes < reference.reclaimable_bytes,
        "fixture must activate compact logits"
    );
    for (chunk, tokens) in tokens.chunks(128).enumerate() {
        let actual = candidate.prefill_batch(tokens, chunk * 128, true)?;
        let expected = reference.prefill_batch(tokens, chunk * 128, true)?;
        assert_eq!(
            actual, expected,
            "full hidden and last logits at chunk {chunk}"
        );
    }
    let position = tokens.len();
    let verify: Vec<_> = tokens.iter().copied().cycle().take(3).collect();
    assert_eq!(
        candidate.step_batch(&verify, position)?,
        reference.step_batch(&verify, position)?
    );
    candidate.commit_batch(1)?;
    reference.commit_batch(1)?;
    for step in 0..8 {
        let token = tokens[step % tokens.len()];
        assert_eq!(
            candidate.step(token, position + 1 + step, position + 1 + step, None, true)?,
            reference.step(token, position + 1 + step, position + 1 + step, None, true)?
        );
    }
    eprintln!("compact/full prompt logits, hidden, verification and decode match bitwise");
    Ok(())
}
