//! Native model loading independent of diagnostic runners and CPU reference execution.
mod kv_scales;
mod weights;
pub use kv_scales::load as load_kv_scales;
mod bindings;
mod budget;
mod phase;
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, ProgramWeights},
    tuning::TuningReport,
};
use infer_core::{Error, ModelId, Result};
use infer_ir::{DataflowGraph, ModelIr};
use infer_models::QuantizedPackage;
use std::path::Path;
pub use weights::{Projection, floats};

/// Load-time execution policy; each request receives independent mutable device state.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub prefill_width: usize,
    pub verification_width: usize,
    /// None follows the checkpoint declaration; Some overrides KV storage.
    pub fp8_kv: Option<bool>,
    pub mtp_depth: usize,
    /// Measure GEMV tiles for geometries the local or shipped tables do not cover. On by default:
    /// tile size is a property of the machine, so the first load measures and caches its winners
    /// instead of asking the operator to choose them.
    pub autotune: bool,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            prefill_width: 1,
            verification_width: 0,
            fp8_kv: None,
            mtp_depth: 0,
            autotune: true,
        }
    }
}
impl LoadOptions {
    fn resolve_verification(&mut self) -> Result<()> {
        self.validate()?;
        if self.mtp_depth > 0 {
            let width = self.mtp_depth + 1;
            if self.verification_width == 0 {
                self.verification_width = width;
            }
            if self.verification_width != width {
                return Err(Error::invalid(
                    "verification width must equal MTP depth plus one",
                ));
            }
        }
        self.validate()
    }
    fn validate(&self) -> Result<()> {
        if ![
            0,
            1,
            crate::constants::FUSED_VERIFY_LANES,
            crate::constants::PREFILL_LANES,
            crate::constants::MID_PREFILL_LANES,
            crate::constants::MAX_PREFILL_LANES,
            crate::constants::WIDE_PREFILL_LANES,
        ]
        .contains(&self.prefill_width)
            || self.verification_width > crate::constants::MAX_VERIFICATION_WIDTH
            || self.verification_width == 1
            || (self.prefill_width == crate::constants::FUSED_VERIFY_LANES
                && ![0, crate::constants::FUSED_VERIFY_LANES].contains(&self.verification_width))
        {
            return Err(Error::invalid("CUDA model prefill/verification widths"));
        }
        if self.mtp_depth > crate::constants::MAX_MTP_DEPTH {
            return Err(Error::invalid("CUDA MTP draft depth"));
        }
        Ok(())
    }
}

/// Immutable draft graph and weights for MTP speculation.
struct Draft {
    graph: DataflowGraph,
    weights: ProgramWeights,
}

/// Immutable shared model resources. CUDA graphs retain weights independently of this factory.
/// Mutable Conv/GDN/KV and activations are allocated separately for each request.
pub struct LoadedModel {
    device: CudaDevice,
    profile: crate::device::DeviceProfile,
    model: ModelIr,
    graph: DataflowGraph,
    weights: ProgramWeights,
    draft: Option<Draft>,
    vision: Option<LoadedVision>,
    imported: infer_spi::ImportedModel,
    mtp_depth: usize,
    /// Whether the prompt graph width was chosen automatically at load time.
    automatic_prefill: bool,
    tuning: TuningReport,
    requirements: infer_ir::CapabilityRequirements,
}

/// Vision tower bound alongside the text model.
///
/// Encoding runs the real kernels; a family without an image encoder simply has no `LoadedVision`,
/// so a caller holding one can never silently skip encoding.
pub struct LoadedVision {
    weights: crate::vision::VisionWeights,
    encoder: infer_spi::ModalityEncoder,
}

impl LoadedVision {
    /// Bind the declared encoder, or nothing when the family has no image modality.
    /// # Errors
    /// Rejects declared encoders whose weights or geometry are unusable.
    fn bind(device: &CudaDevice, package: &mut QuantizedPackage) -> Result<Option<Self>> {
        let declared = package
            .imported
            .modalities
            .iter()
            .find(|plan| plan.modality == infer_ir::Modality::Image)
            .and_then(|plan| plan.encoder.clone());
        let Some(encoder) = declared else {
            return Ok(None);
        };
        let weights = crate::vision::VisionWeights::load(device, package, &encoder)?;
        Ok(Some(Self { weights, encoder }))
    }

    /// Declared geometry of the bound tower.
    #[must_use]
    pub const fn encoder(&self) -> &infer_spi::ModalityEncoder {
        &self.encoder
    }

    /// Encode one preprocessed image into `merged_tokens × text_hidden` embeddings.
    /// # Errors
    /// Rejects a patch grid that disagrees with the pixel buffer or failed CUDA execution.
    pub fn encode(
        &self,
        device: &CudaDevice,
        image: &infer_models::PromptImage,
    ) -> Result<Vec<f32>> {
        let (temporal, height, width) = image.grid;
        let patches = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| Error::invalid("image patch grid overflow"))?;
        if patches == 0 {
            return Err(Error::invalid("image patch grid is empty"));
        }
        let width = crate::vision::patch_width(&self.encoder)?;
        if image.pixels.len() != patches * width {
            return Err(Error::invalid("image pixels do not match the patch grid"));
        }
        // The position contribution is a host-side gather of the learned table, then the tower
        // consumes patch embeddings and the merger folds each merge group back into text width.
        let taps = infer_models::vision::position_taps(image.grid, &self.encoder)?;
        let positions = infer_models::vision::gather_positions(
            self.weights.position_table(),
            &self.encoder,
            &taps,
        )?;
        let embedded = crate::vision::patch_embed(
            device,
            &self.encoder,
            &self.weights,
            &image.pixels,
            &positions,
            patches,
        )?;
        let hidden = crate::vision::tower(
            device,
            &self.weights,
            &self.encoder,
            &embedded,
            image.grid,
            patches,
        )?;
        crate::vision::merger(device, &self.weights, &self.encoder, &hidden, patches)
    }
}

impl LoadedModel {
    pub(crate) fn draft_graph(&self) -> Option<&DataflowGraph> {
        self.draft.as_ref().map(|draft| &draft.graph)
    }

    /// Effective graph widths and speculation depth after automatic resolution.
    #[must_use]
    pub fn execution_profile(&self) -> infer_ir::ExecutionProfileInspection {
        infer_ir::ExecutionProfileInspection {
            prefill_width: self.weights.prefill_width,
            batch_width: self.weights.batch_width,
            mtp_depth: self.mtp_depth,
            automatic_prefill: self.automatic_prefill,
            arena_budget_bytes: self.profile.arena_budget_bytes(),
        }
    }
    /// # Errors
    /// Rejects unsupported formats, invalid policies, budgets or CUDA loading errors.
    pub fn open(
        device: CudaDevice,
        root: impl AsRef<Path>,
        id: ModelId,
        mut options: LoadOptions,
    ) -> Result<Self> {
        options.resolve_verification()?;
        // Query hardware once: the profile keys machine-local tuning artifacts and derives the
        // device budget policy, so no per-model or per-board table needs maintaining here.
        let profile = device.profile()?.clone();
        let package_phase = phase::Phase::start(
            "package",
            format!(
                "reading checkpoint metadata from {}",
                root.as_ref().display()
            ),
        );
        let mut package = QuantizedPackage::open(root, id).map_err(|e| package_phase.fail(&e))?;
        package_phase.finish();
        options.fp8_kv =
            Some(options.fp8_kv.unwrap_or_else(|| {
                package.kv_cache_dtype == Some(infer_models::TensorDtype::F8E4m3)
            }));
        let automatic_prefill = options.prefill_width == 0;
        if automatic_prefill {
            options.prefill_width = crate::constants::PREFILL_LANES;
        }
        let mtp_depth = options.mtp_depth;
        let (mut weights, draft, tuning) =
            bindings::load(&device, &profile, &mut package, &options)?;
        if automatic_prefill
            && !weights.projections.iter().any(|(id, weight)| {
                matches!(weight, crate::mlp::ProjectionWeight::Fp4(..))
                    && !weights.input_scales.contains_key(id)
            })
        {
            let geometry_phase =
                phase::Phase::start("geometry", "resolving the prompt graph width");
            // One request owns one arena, so the width decision is bounded by the profiled
            // activation-arena budget, not by a fixed share of total memory.
            //
            // Both this budget and the recurrent cap below are load-bearing: letting live
            // free memory fund the 128-lane 27B prompt graph measured worse on three of four
            // cases (short TTFT 0.058 -> 0.084 s, batch4 0.247 -> 0.291 s, hot-prefix
            // 0.090 -> 0.115 s) while the long case barely moved (0.436 -> 0.424 s). Do not
            // widen a quantized recurrent prompt graph without an end-to-end measurement.
            // Quantized recurrent graphs keep the scalar Delta path for numerical
            // stability. Limit their prompt graph size while sharing projection loads.
            // The arena is the real bound; this is only a cap for the ladder below. Quantized
            // recurrent graphs stay at the established width, dense ones may take the wide rung
            // when the arena has room for it.
            let quantized_recurrent = package
                .graph
                .nodes
                .iter()
                .any(|node| matches!(node.op, infer_ir::TensorOp::Delta { .. }))
                && (!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty());
            // One request owns one arena, so the width decision is bounded by the profiled
            // activation-arena budget, not by a fixed share of total memory.
            select_prompt_width(
                &package.graph,
                quantized_recurrent,
                profile.arena_budget_bytes(),
                &mut weights,
            )?;
            tracing::info!(
                target: "infer::load",
                prefill_width = weights.prefill_width,
                narrow_prefill_width = weights.narrow_prefill_width,
                "prompt graph width selected"
            );
            geometry_phase.finish();
        }
        let requirements = package.imported.requirements.clone();
        let vision_phase = phase::Phase::start("vision", "binding the vision tower");
        let vision =
            LoadedVision::bind(&device, &mut package).map_err(|e| vision_phase.fail(&e))?;
        vision_phase.finish();
        let imported = package.imported.clone();
        Ok(Self {
            device,
            profile,
            model: package.imported.model,
            graph: package.graph,
            weights,
            draft,
            vision,
            imported,
            mtp_depth,
            automatic_prefill,
            tuning,
            requirements,
        })
    }
    /// Provider-imported description of this model, including its modalities.
    #[must_use]
    pub const fn imported(&self) -> &infer_spi::ImportedModel {
        &self.imported
    }

    /// Bound vision tower, absent for text-only families.
    #[must_use]
    pub const fn vision(&self) -> Option<&LoadedVision> {
        self.vision.as_ref()
    }
    /// Device capabilities this model's provider declared as required.
    #[must_use]
    pub const fn requirements(&self) -> &infer_ir::CapabilityRequirements {
        &self.requirements
    }
    /// Speculative draft depth configured at load time; zero disables speculation.
    #[must_use]
    pub const fn mtp_depth(&self) -> usize {
        self.mtp_depth
    }
    /// Automatic tiling decisions from this load: what was measured, and what fell back.
    #[must_use]
    pub const fn tuning(&self) -> &TuningReport {
        &self.tuning
    }
    #[must_use]
    pub const fn model(&self) -> &ModelIr {
        &self.model
    }
    #[must_use]
    pub const fn graph(&self) -> &DataflowGraph {
        &self.graph
    }
    #[must_use]
    pub const fn profile(&self) -> &crate::device::DeviceProfile {
        &self.profile
    }
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }
    /// # Errors
    /// Rejects invalid capacity, exceeded state budgets or CUDA graph capture failures.
    pub fn sequence(&self, capacity: usize) -> Result<DeviceProgram> {
        DeviceProgram::new(
            &self.device,
            &self.graph,
            &self.weights,
            capacity,
            self.model.hidden_size,
            self.model.vocab_size,
        )
    }
    pub(crate) fn ensure_verification(&self, program: &mut DeviceProgram) -> Result<()> {
        program.ensure_verification(&self.graph, &self.weights)
    }
    /// # Errors
    /// Rejects invalid capacity, missing draft configuration or capture failures.
    pub fn draft(&self, capacity: usize) -> Result<Option<DeviceProgram>> {
        let Some(draft) = &self.draft else {
            return Ok(None);
        };
        DeviceProgram::new(
            &self.device,
            &draft.graph,
            &draft.weights,
            capacity,
            self.model.hidden_size,
            self.model.vocab_size,
        )
        .map(Some)
    }

    pub(crate) fn draft_slot_pool(
        &self,
        width: usize,
        capacity: usize,
    ) -> Result<Option<crate::resident::slot_batch::SlotPool>> {
        let Some(draft) = &self.draft else {
            return Ok(None);
        };
        if draft.weights.fusion.is_none()
            || draft.graph.tensors.iter().any(|tensor| {
                matches!(
                    tensor.storage,
                    infer_ir::TensorStorage::State {
                        kind: infer_ir::StateKind::Conv | infer_ir::StateKind::LinearAttention,
                        ..
                    }
                )
            })
        {
            return Ok(None);
        }
        crate::resident::slot_batch::SlotPool::new(
            &self.device,
            &draft.graph,
            &draft.weights,
            width,
            capacity,
            self.model.hidden_size,
            (self.model.vocab_size, 0),
        )
        .map(Some)
    }

    /// Continuous-batching slot pool over the target weights: `width` slots of `capacity`
    /// tokens each, plus one shared graph captured against them. With a draft loaded the
    /// graph is the pooled speculation graph of `mtp_depth + 1` candidate lanes per slot.
    /// # Errors
    /// Rejects invalid geometry, exceeded device budgets or CUDA capture failures.
    pub(crate) fn slot_pool(
        &self,
        width: usize,
        capacity: usize,
    ) -> Result<crate::resident::slot_batch::SlotPool> {
        crate::resident::slot_batch::SlotPool::new(
            &self.device,
            &self.graph,
            &self.weights,
            width,
            capacity,
            self.model.hidden_size,
            (
                self.model.vocab_size,
                if self.mtp_depth == 0 {
                    0
                } else {
                    self.mtp_depth + 1
                },
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mtp_depth_controls_the_actual_verification_width() -> Result<()> {
        for depth in [1, 2, 4, 8] {
            let mut options = LoadOptions {
                mtp_depth: depth,
                prefill_width: 32,
                ..LoadOptions::default()
            };
            options.resolve_verification()?;
            assert_eq!(options.verification_width, depth + 1);
        }
        let mut mismatch = LoadOptions {
            mtp_depth: 4,
            verification_width: 3,
            ..LoadOptions::default()
        };
        assert!(mismatch.resolve_verification().is_err());
        Ok(())
    }
    #[test]
    fn mtp_depth_bounds_are_validated() {
        assert!(
            LoadOptions {
                mtp_depth: 0,
                ..LoadOptions::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            LoadOptions {
                mtp_depth: crate::constants::MAX_MTP_DEPTH,
                ..LoadOptions::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            LoadOptions {
                mtp_depth: crate::constants::MAX_MTP_DEPTH + 1,
                ..LoadOptions::default()
            }
            .validate()
            .is_err()
        );
    }
}

/// Pick the prompt graph width from the arena budget.
///
/// A wider prompt graph is arithmetically identical to several narrower chunks: keys and values
/// always pass through the cache, and the prompt GEMM accumulates per element over K. It is also
/// fewer replays, which is where the measured gains come from. The 256-lane rung is only offered
/// to quantized recurrent models, where it measured -4.4% long-prompt TTFT with identical tokens;
/// on the dense models it measured neutral on the long case and 1.3% worse on the short one.
fn select_prompt_width(
    graph: &DataflowGraph,
    quantized_recurrent: bool,
    arena_budget: u64,
    weights: &mut ProgramWeights,
) -> Result<()> {
    let recurrent_limit = if quantized_recurrent {
        crate::constants::MID_PREFILL_LANES
    } else {
        crate::constants::MAX_PREFILL_LANES
    };
    let mut selected = 0usize;
    for width in [
        crate::constants::MAX_PREFILL_LANES,
        crate::constants::MID_PREFILL_LANES,
    ]
    .into_iter()
    .filter(|&width| width <= recurrent_limit)
    {
        let needed = crate::resident::arena::ActivationArena::required_bytes(graph, width)?;
        if needed as u64 <= arena_budget {
            selected = width;
            break;
        }
    }
    if selected != 0 && selected < crate::constants::MAX_PREFILL_LANES {
        let rungs: &[usize] = if quantized_recurrent {
            &[
                crate::constants::WIDE_PREFILL_LANES,
                crate::constants::MAX_PREFILL_LANES,
            ]
        } else {
            &[crate::constants::MAX_PREFILL_LANES]
        };
        for &wide in rungs {
            let needed = crate::resident::arena::ActivationArena::required_bytes(graph, wide)?;
            if needed as u64 <= arena_budget {
                weights.narrow_prefill_width = selected;
                selected = wide;
                break;
            }
        }
    }
    if selected != 0 {
        weights.prefill_width = selected;
    }
    Ok(())
}
