use super::*;
use crate::test_support::fixture_graph;
use crate::{mlp::ProjectionWeight, resident::FusionWeights};
use cutile::half::bf16;
use infer_core::TensorId;
use infer_ir::Sampling;
use std::collections::{BTreeMap, BTreeSet};

fn fixture_weights(device: &CudaDevice, width: usize) -> Result<ProgramWeights> {
    let embedding = TensorId::ONE;
    let embedding_values = (0..width * width)
        .map(|i| bf16::from_f32(f32::from(i16::try_from(i * 7 % 31).unwrap() - 15) / 16.0))
        .collect();
    let fusion_values = (0..width * 2 * width)
        .map(|i| bf16::from_f32(f32::from(i16::try_from(i * 13 % 29).unwrap() - 14) / 32.0))
        .collect();
    Ok(ProgramWeights {
        mrope: false,
        autotune: false,
        resident_tuning: Arc::new(crate::tuning::resident_gemm::ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 0,
        prefill_width: 1,
        narrow_prefill_width: 0,
        attention_profile: crate::AttentionProfile::Canonical,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: Some(FusionWeights {
            projection: ProjectionWeight::Dense(device.upload(fusion_values, &[width, 2 * width])?),
            tiling: crate::strategy::default_tiling()?,
            norms: device.upload(vec![0.0; 2 * width], &[2, width])?,
            epsilon: 1e-6,
            offset: 1.0,
        }),
        fp8_inputs: BTreeSet::default(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        embeddings: BTreeMap::from([(
            embedding,
            device.upload(embedding_values, &[width, width])?,
        )]),
    })
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn draft_device_hidden_reuse_matches_host_roundtrips_bitwise() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let weights = fixture_weights(&device, width)?;
    let mut host = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut resident = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    assert!(resident.has_device_greedy());
    let first: Vec<_> = (0..width)
        .map(|i| f32::from(i16::try_from(i).unwrap() - 24) / 16.0)
        .collect();
    let mut previous = first.clone();
    let mut token = 7;
    for position in 0..32 {
        let (next_hidden, expected_logits) =
            host.step(token, position, position, Some(&previous), true)?;
        let expected =
            infer_workloads::sample(&expected_logits, &Sampling::default(), 0, position + 1)?;
        let actual = resident.step_greedy(
            token,
            position,
            position,
            (position == 0).then_some(first.as_slice()),
        )?;
        assert_eq!(actual, expected, "position={position}");
        let actual_hidden = device.read(&resident.hidden)?;
        let actual_logits = device.read(&resident.logits)?;
        assert_eq!(
            actual_hidden
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            next_hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "hidden position={position}"
        );
        assert_eq!(
            actual_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "logits position={position}"
        );
        assert_eq!(resident.next_position, host.next_position);
        previous = next_hidden;
        token = expected;
    }
    Ok(())
}

fn assert_selected_rows(
    expected: &[(Vec<f32>, Vec<f32>)],
    actual: &crate::resident::batch::BatchReadout,
) -> Result<()> {
    assert_eq!(expected.len(), actual.rows.len());
    let tokens = actual
        .tokens
        .as_ref()
        .ok_or_else(|| Error::invariant("fixture selected tokens"))?;
    for (row, ((hidden, logits), (actual_hidden, actual_logits))) in
        expected.iter().zip(&actual.rows).enumerate()
    {
        assert_eq!(actual_logits.as_slice(), &[] as &[f32]);
        assert_eq!(
            hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            actual_hidden
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let selected = infer_workloads::sample(logits, &Sampling::default(), 0, row)?;
        assert_eq!(tokens[row], Some(selected));
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn target_device_readout_matches_host_for_continuation_and_partial_verification() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    weights.batch_width = 3;
    let mut host = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut selected = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut token = 7;
    for position in 0..16 {
        let expected = host.step(token, position, position, None, true)?;
        let actual = selected.step_selected(token, position, true)?;
        assert_selected_rows(&[expected], &actual)?;
        assert_eq!(host.position(), selected.position());
        token = actual.tokens.as_ref().unwrap()[0].unwrap();
    }
    host.reset()?;
    selected.reset()?;
    let mut position = 0;
    for round in 0..12 {
        let chain = [7, 13, 31];
        let expected = host.step_batch(&chain, position)?;
        let actual = selected.step_batch_readout(&chain, position, true)?;
        assert_selected_rows(&expected, &actual)?;
        let committed = round % 3 + 1;
        host.commit_batch(committed)?;
        selected.commit_batch(committed)?;
        assert_eq!(host.position(), selected.position());
        position += committed;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn pooled_device_readout_matches_host_for_inactive_slots_and_partial_commits() -> Result<()> {
    use crate::resident::slot_batch::SlotPool;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    let mut host = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 1))?;
    let mut selected = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 1))?;
    for position in 0..8 {
        let lanes = [(3, 31, position, position), (1, 13, position, position)];
        let expected = host.run_lanes(&device, &lanes)?;
        let actual = selected.run_lanes_readout(&device, &lanes, true)?;
        assert_selected_rows(&expected, &actual)?;
    }
    let mut host = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 3))?;
    let mut selected = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 3))?;
    let mut positions = [0; 4];
    for round in 0..8 {
        let lanes: Vec<_> = [3, 1]
            .into_iter()
            .flat_map(|slot| {
                (0..3).map(move |lane| {
                    (
                        slot,
                        lane,
                        [7, 13, 31][lane],
                        positions[slot] + lane,
                        positions[slot] + lane,
                    )
                })
            })
            .collect();
        let expected = host.run_verify(&device, &lanes)?;
        let actual = selected.run_verify_readout(&device, &lanes, true)?;
        assert_eq!(actual.rows.len(), 12);
        let tokens = actual.tokens.as_ref().unwrap();
        for slot in 0..4 {
            for lane in 0..3 {
                let row = slot * 3 + lane;
                if [1, 3].contains(&slot) {
                    assert_eq!(
                        tokens[row],
                        Some(infer_workloads::sample(
                            &expected[row].1,
                            &Sampling::default(),
                            0,
                            row
                        )?)
                    );
                    assert_eq!(actual.rows[row].1.as_slice(), &[] as &[f32]);
                    assert_eq!(
                        actual.rows[row]
                            .0
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected[row]
                            .0
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(tokens[row], None);
                    assert!(actual.rows[row].0.is_empty() && actual.rows[row].1.is_empty());
                }
            }
        }
        for slot in [1, 3] {
            let accepted = (round + slot) % 3;
            host.commit_verify(&device, slot, accepted)?;
            selected.commit_verify(&device, slot, accepted)?;
            positions[slot] += accepted + 1;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn fresh_pooled_prompt_and_retained_wide_prompt_match_for_continuation() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    weights.batch_width = 3;
    weights.prefill_width = 64;
    weights.narrow_prefill_width = 32;
    let mut retained = DeviceProgram::new(&device, &graph, &weights, 256, width, width)?;
    let mut pooled = DeviceProgram::new_pooled(&device, &graph, &weights, 256, width, width)?;
    assert_eq!(retained.prefill_width(), 64);
    assert_eq!(pooled.prefill_width(), 32);
    assert!(pooled.batch.is_none() && pooled.prompt_batch.is_none());
    retained.forfeit_verification();
    assert!(retained.batch.is_none());
    assert_eq!(retained.prefill_width(), 64);
    assert!(retained.reclaimable_bytes > pooled.reclaimable_bytes);
    let prompt: Vec<_> = (0..63)
        .map(|i| u32::try_from(i * 7 % width).expect("fixture token fits"))
        .collect();
    let expected = retained.prefill_batch(&prompt, 0, true)?;
    let mut actual = pooled.prefill_batch(&prompt[..32], 0, false)?;
    actual.extend(pooled.prefill_batch(&prompt[32..], 32, true)?);
    let bits = |rows: &crate::resident::BatchOutput| {
        rows.iter()
            .map(|(h, l)| {
                (
                    h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&expected), bits(&actual));
    for position in 63..79 {
        let token = u32::try_from(position % width).expect("fixture token fits");
        let expected = retained.step(token, position, position, None, true)?;
        let actual = pooled.step(token, position, position, None, true)?;
        assert_eq!(
            bits(&vec![expected]),
            bits(&vec![actual]),
            "position={position}"
        );
    }
    Ok(())
}
