use super::{Draft, LoadOptions, Projection, floats, phase};
use crate::mlp::ProjectionWeight;
use crate::tuning::{TilingSource, TuningReport, resolve_table, shipped_table, table_path};
use crate::{
    device::CudaDevice,
    resident::{FusionWeights, ProgramWeights},
};
use cutile::half::bf16;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, TensorOp, TensorStorage};
use infer_models::{QuantizedPackage, TensorDtype};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Read budget for one BF16 embedding tensor.
const MAX_EMBEDDING_READ_BYTES: u64 = 4 * crate::constants::GIB as u64;

pub(super) fn load(
    device: &CudaDevice,
    profile: &crate::device::DeviceProfile,
    package: &mut QuantizedPackage,
    options: &LoadOptions,
) -> Result<(ProgramWeights, Option<Draft>, TuningReport)> {
    let path = table_path();
    let table = resolve_table(path.as_deref())?;
    let shipped = shipped_table()?;
    let tuning_source = TilingSource {
        local: table.as_ref(),
        shipped: Some(shipped),
        path,
        autotune: options.autotune,
    };
    let mut tuning = TuningReport::default();
    let mut weights = empty_weights(options);
    let weights_phase = phase::Phase::start(
        "weights",
        format!("binding {} checkpoint weights", package.weights.len()),
    );
    bind_tensors(
        device,
        profile,
        package,
        &tuning_source,
        &mut tuning,
        &mut weights,
    )
    .map_err(|e| weights_phase.fail(&e))?;
    weights_phase.finish();
    add_rope(
        device,
        &package.graph,
        &mut weights,
        &package.imported.model.position,
    )?;
    if options.fp8_kv == Some(true) {
        let kv_phase = phase::Phase::start("kv_scales", "uploading KV scales");
        weights.kv_scales = super::kv_scales::load(package).map_err(|e| kv_phase.fail(&e))?;
        kv_phase.finish();
    }
    let draft = if options.mtp_depth > 0 {
        let draft_phase = phase::Phase::start("draft", "binding the MTP draft head");
        let draft = draft(
            device,
            profile,
            package,
            &weights,
            &tuning_source,
            &mut tuning,
        )
        .map_err(|e| draft_phase.fail(&e))?;
        draft_phase.finish();
        Some(draft)
    } else {
        None
    };
    Ok((weights, draft, tuning))
}

/// Weights with their graph-independent fields set; graph tensors are bound by [`bind_tensors`].
fn empty_weights(options: &LoadOptions) -> ProgramWeights {
    ProgramWeights {
        batch_width: options.verification_width.max(
            if options.prefill_width == crate::constants::FUSED_VERIFY_LANES {
                crate::constants::FUSED_VERIFY_LANES
            } else {
                0
            },
        ),
        prefill_width: options.prefill_width,
        narrow_prefill_width: 0,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: None,
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
    }
}

/// Bind every graph weight: projections choose a tile, embeddings and constants get uploaded.
fn bind_tensors(
    device: &CudaDevice,
    profile: &crate::device::DeviceProfile,
    package: &mut QuantizedPackage,
    tuning_source: &TilingSource<'_>,
    tuning: &mut TuningReport,
    weights: &mut ProgramWeights,
) -> Result<()> {
    let projections: BTreeSet<_> = package
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Linear)
        .map(|n| n.inputs[1])
        .collect();
    let embeddings: BTreeSet<_> = package
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Embedding)
        .map(|n| n.inputs[0])
        .collect();
    let total = package
        .graph
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.storage, TensorStorage::Weight { .. }))
        .count();
    let mut done = 0;
    for tensor in package.graph.tensors.clone() {
        let TensorStorage::Weight { slot } = tensor.storage else {
            continue;
        };
        let source = package
            .weights
            .get(&slot)
            .ok_or_else(|| Error::invalid("missing graph weight"))?
            .clone();
        if projections.contains(&tensor.id) {
            let projection = Projection::load(device, package, &source)?.resident();
            bind_activation(package, &source, weights, tensor.id)?;
            let tiling =
                crate::tuning::select_tiling(device, &projection, profile, tuning_source, tuning)?;
            weights.tiling.insert(tensor.id, tiling);
            // A tied output projection shares its storage with the input embedding, so the very
            // same weight must also be reachable by the embedding lookup.
            if embeddings.contains(&tensor.id) {
                let ProjectionWeight::Dense(shared) = &projection else {
                    return Err(Error::unsupported(
                        "a tied input embedding requires a dense output projection",
                    ));
                };
                weights.embeddings.insert(tensor.id, Arc::clone(shared));
            }
            weights.projections.insert(tensor.id, projection);
        } else if embeddings.contains(&tensor.id) {
            if source.data.dtype != TensorDtype::BF16 {
                return Err(Error::unsupported("CUDA embedding requires BF16"));
            }
            let bytes = package.read(&source.data, MAX_EMBEDDING_READ_BYTES)?;
            let values = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
                .collect();
            drop(bytes);
            weights
                .embeddings
                .insert(tensor.id, device.upload(values, &source.shape)?);
        } else {
            let values = floats(package, &source.data)?;
            let length = values.len();
            weights
                .constants
                .insert(tensor.id, device.upload(values, &[length])?);
        }
        done += 1;
        tracing::debug!(
            target: "infer::load",
            done,
            total,
            bytes = source.data.bytes,
            item = source.data.name.as_str(),
            "weight bound"
        );
    }
    Ok(())
}

/// Lower the single attention block of the MTP head into a standalone draft graph.
fn draft(
    device: &CudaDevice,
    profile: &crate::device::DeviceProfile,
    package: &mut QuantizedPackage,
    target: &ProgramWeights,
    source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<Draft> {
    // The draft head's prefix, depth and fusion slots come from the model provider.
    let plan = package
        .imported
        .speculation
        .clone()
        .ok_or_else(|| Error::unsupported("MTP draft head is not declared"))?;
    if plan.layers != 1 {
        return Err(Error::unsupported("MTP requires exactly one MTP layer"));
    }
    let (block, graph) = package.provider.draft_graph(&package.imported.model)?;
    let mut weights = draft_weights(device, profile, package, target, &graph, source, report)?;
    add_rope(device, &graph, &mut weights, &block.position)?;
    Ok(Draft { graph, weights })
}

/// Bind draft weights, sharing the target embedding and vocabulary projection by slot.
fn draft_weights(
    device: &CudaDevice,
    profile: &crate::device::DeviceProfile,
    package: &mut QuantizedPackage,
    target: &ProgramWeights,
    graph: &DataflowGraph,
    tiling_source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<ProgramWeights> {
    let plan = package
        .imported
        .speculation
        .clone()
        .ok_or_else(|| Error::unsupported("MTP draft head is not declared"))?;
    let linear_inputs: BTreeSet<_> = graph
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Linear)
        .map(|n| n.inputs[1])
        .collect();
    let mut weights = ProgramWeights {
        // The draft captures the widest prompt graph so priming runs one pass per chunk instead
        // of one single-token pass per prompt token; a 32-lane graph turned a 511-token prompt
        // into sixteen replays. It has no verify batch: the target verifies, the draft only
        // proposes.
        batch_width: 0,
        prefill_width: crate::constants::MAX_PREFILL_LANES,
        narrow_prefill_width: 0,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: Some(draft_fusion(
            device,
            profile,
            package,
            tiling_source,
            &plan,
            report,
        )?),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
    };
    for tensor in &graph.tensors {
        let TensorStorage::Weight { slot } = &tensor.storage else {
            continue;
        };
        if slot == "embed_tokens.weight" {
            let id = weight_id(&package.graph, slot)?;
            let embedding = target
                .embeddings
                .get(&id)
                .cloned()
                .ok_or_else(|| Error::invalid("missing MTP target embedding"))?;
            weights.embeddings.insert(tensor.id, embedding);
            continue;
        }
        if slot == "lm_head.weight" {
            let id = weight_id(&package.graph, slot)?;
            let projection = target
                .projections
                .get(&id)
                .cloned()
                .ok_or_else(|| Error::invalid("missing MTP target projection"))?;
            let tiling = target
                .tiling
                .get(&id)
                .copied()
                .ok_or_else(|| Error::invalid("missing MTP target tiling"))?;
            weights.projections.insert(tensor.id, projection);
            if target.fp8_inputs.contains(&id) {
                weights.fp8_inputs.insert(tensor.id);
            }
            if let Some(scale) = target.input_scales.get(&id) {
                weights.input_scales.insert(tensor.id, *scale);
            }
            weights.tiling.insert(tensor.id, tiling);
            continue;
        }
        let source = package
            .mtp
            .get(&format!("{}{slot}", plan.prefix))
            .cloned()
            .ok_or_else(|| Error::invalid("missing MTP binding"))?;
        if source.shape != tensor.shape {
            return Err(Error::invalid("MTP weight shapes"));
        }
        if linear_inputs.contains(&tensor.id) {
            let projection = Projection::load(device, package, &source)?.resident();
            bind_activation(package, &source, &mut weights, tensor.id)?;
            weights.tiling.insert(
                tensor.id,
                crate::tuning::select_tiling(device, &projection, profile, tiling_source, report)?,
            );
            weights.projections.insert(tensor.id, projection);
        } else {
            let values = floats(package, &source.data)?;
            let length = values.len();
            weights
                .constants
                .insert(tensor.id, device.upload(values, &[length])?);
        }
    }
    Ok(weights)
}

/// Load the MTP fusion projection and its two input normalization vectors.
fn draft_fusion(
    device: &CudaDevice,
    profile: &crate::device::DeviceProfile,
    package: &mut QuantizedPackage,
    tiling_source: &TilingSource<'_>,
    plan: &infer_spi::SpeculationPlan,
    report: &mut TuningReport,
) -> Result<FusionWeights> {
    let fusion = plan
        .fusion
        .as_ref()
        .ok_or_else(|| Error::unsupported("MTP fusion projection is not declared"))?;
    let fc = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.projection))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let embedding_norm = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.norms[0]))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let hidden_norm = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.norms[1]))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let hidden = package.imported.model.hidden_size;
    if fc.shape != [hidden, hidden * 2]
        || embedding_norm.shape != [hidden]
        || hidden_norm.shape != [hidden]
    {
        return Err(Error::invalid("MTP fusion weight shapes"));
    }
    let fc = Projection::load(device, package, &fc)?.resident();
    // The fusion projection is tuned exactly like the target's own projections.
    let tiling = crate::tuning::select_tiling(device, &fc, profile, tiling_source, report)?;
    let mut norms = floats(package, &embedding_norm.data)?;
    norms.extend_from_slice(&floats(package, &hidden_norm.data)?);
    Ok(FusionWeights {
        projection: fc,
        tiling,
        norms: device.upload(norms, &[2, hidden])?,
        epsilon: package.imported.model.norm_epsilon,
        offset: package.imported.model.norm_weight_offset,
    })
}

fn weight_id(graph: &DataflowGraph, slot: &str) -> Result<TensorId> {
    graph
        .tensors
        .iter()
        .find_map(|spec| match &spec.storage {
            TensorStorage::Weight { slot: name } if name == slot => Some(spec.id),
            _ => None,
        })
        .ok_or_else(|| Error::invalid("missing MTP target weight"))
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Bounded rotary dimensions are intentionally rounded to the F32 device frequency contract"
)]
fn add_rope(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &mut ProgramWeights,
    position: &infer_ir::PositionSpec,
) -> Result<()> {
    for node in &graph.nodes {
        if let TensorOp::Rope {
            rotary_dim, theta, ..
        } = node.op
        {
            let half = rotary_dim / 2;
            let frequencies = (0..half)
                .map(|i| theta.powf(-((2 * i) as f64) / rotary_dim as f64) as f32)
                .collect();
            let axes = infer_models::mrope::axes(position, half)?;
            weights
                .rope_axes
                .insert(node.id, device.upload(axes, &[half])?);
            weights
                .rope_frequencies
                .insert(node.id, device.upload(frequencies, &[half])?);
        }
    }
    Ok(())
}

/// Bind checkpoint activation encoding separately from weight storage and runtime buffers.
fn bind_activation(
    package: &mut QuantizedPackage,
    source: &infer_models::DeviceWeight,
    weights: &mut ProgramWeights,
    id: TensorId,
) -> Result<()> {
    if source.fp8_token_input {
        weights.fp8_inputs.insert(id);
    }
    if source.encoding == infer_models::WeightEncoding::Nvfp4
        && let Some(scale) = source.input_global_scale.as_ref()
    {
        let values = floats(package, scale)?;
        let [value] = values.as_slice() else {
            return Err(Error::invalid("input global scale must be scalar"));
        };
        if !value.is_finite() || *value <= 0.0 {
            return Err(Error::invalid("invalid activation scale"));
        }
        weights.input_scales.insert(id, *value);
    }
    Ok(())
}
