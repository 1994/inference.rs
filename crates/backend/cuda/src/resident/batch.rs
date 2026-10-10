//! Node-major verification: consecutive projections reuse weights in cache.
//! Recurrent checkpoints stay on the device; KV rollback is a logical prefix.
//! Prefill width 32 keeps one batched arena: auxiliaries record once per node.
use super::{
    ActivationArena, ProgramWeights,
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
    profile,
};
use crate::device::{CudaDevice, ReadbackSource, device_error};
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorNode, TensorOp, TensorStorage};
use std::collections::BTreeMap;

pub type States = BTreeMap<TensorId, Vec<Tensor<f32>>>;
pub type BatchOutput = Vec<(Vec<f32>, Vec<f32>)>;

pub(super) struct Lane {
    pub arena: ActivationArena,
    metadata: Tensor<i32>,
    checkpoints: States,
}

pub(super) struct BatchGraph {
    graph: Option<CudaGraph<()>>,
    readbacks: crate::device::Readbacks,
    prefill: CudaGraph<()>,
    prefill_last: CudaGraph<()>,
    restore: Vec<CudaGraph<()>>,
    /// Split-K partial windows the captured fp4 prompt GEMMs accumulate into;
    /// declared after the graphs so their executables drop first.
    _partials: BTreeMap<usize, Tensor<f32>>,
    _recurrent: super::recurrent_prefill::Workspace,
    metadata: Vec<Tensor<i32>>,
    prefill_info: Option<Tensor<i32>>,
    hidden: Vec<Arc<Tensor<f32>>>,
    logits: Vec<Arc<Tensor<f32>>>,
    pub width: usize,
    /// Token capacity of the KV arena this graph's kernels address. Positions are rows in it,
    /// so a chunk that would write past the end is an internal inconsistency, not a request
    /// the engine may reject later.
    pub capacity: usize,
    /// Positions the state writes lag the `RoPE` position by, matching the MTP KV offset.
    state_offset: i32,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graphs so their executables are destroyed before the events they record.
    profile: Option<profile::PrefillProfile>,
    // Batched width-32 arena anchors every slot allocation past graph replays;
    // declared last so it drops after the graphs that reference its rows.
    _arena: Option<ActivationArena>,
}

pub(super) struct BatchBuilder<'a> {
    pub device: &'a CudaDevice,
    pub graph: &'a DataflowGraph,
    pub weights: &'a ProgramWeights,
    pub nvfp4: &'a mut super::nvfp4_gemm::Workspace,
    pub delegated: Option<&'a mut crate::device::cublaslt::Support>,
    pub attention: &'a mut super::attention_decode::Workspace,
    pub states: &'a mut States,
    pub fp8_states: &'a mut Fp8Caches,
    pub external: &'a Tensor<f32>,
    pub lane_external: &'a mut Vec<Tensor<f32>>,
    pub lane_fusion: &'a mut Vec<Option<super::program::FusionWorkspace>>,
    pub capacity: usize,
    pub width: usize,
}

/// Decode-only slot graph: lane `i` reads and writes the slot pool's `i`-th state set,
/// so one replay advances every leased sequence through the same weight pass.
///
/// Inactive lanes are masked with a `-1` state position (the state kernels gate writes on
/// it); slot tenants never rewind, so no rollback checkpoints are captured.
pub(super) struct SlotDecodeGraph {
    graph: CudaGraph<()>,
    state_only: Option<CudaGraph<()>>,
    readbacks: crate::device::Readbacks,
    metadata: Vec<Tensor<i32>>,
    // Batched hidden rows stay allocated for the graph's writes but are never read back.
    hidden: Arc<Tensor<f32>>,
    pub(super) external_hidden: bool,
    logits: Arc<Tensor<f32>>,
    pub width: usize,
    /// Host-side state position each slot's next replay must continue at.
    cursors: Vec<usize>,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graph so its executable is destroyed before the events it records.
    profile: Option<profile::SlotProfile>,
    // The shared batched arena anchors the graph's activations; declared last so the
    // graph drops first.
    _arena: ActivationArena,
}

impl BatchBuilder<'_> {
    /// Batch graph plus every prompt graph the program captures, widest last.
    ///
    /// A prompt chunk costs one fixed-width replay whatever its token count, so a program may
    /// capture a second, narrower prompt graph and route short chunks to it.
    /// Capture the verification graph (unless the sequence will speculate through the
    /// shared slot pool) plus whichever prompt graphs the configured widths select.
    /// `wide_prompt` skips the widest prompt graph: pool-bound sequences prefill short
    /// prompts through the narrow graph, and a per-sequence wide prompt arena costs more
    /// physical memory than four pooled residents have room for.
    pub fn build_pair(
        &mut self,
        verification: bool,
        wide_prompt: bool,
    ) -> Result<(Option<BatchGraph>, Option<BatchGraph>, Option<BatchGraph>)> {
        if self.weights.batch_width > crate::constants::MAX_VERIFICATION_WIDTH {
            return Err(Error::invalid("verification width exceeds 9"));
        }
        let batch = if verification && self.width > 1 {
            Some(self.build()?)
        } else {
            None
        };
        let wide = self.weights.prefill_width;
        let narrow = self.weights.narrow_prefill_width;
        if wide < crate::constants::PREFILL_LANES && wide > 1 && wide != self.weights.batch_width {
            return Err(Error::invalid(
                "prefill width must match verification width or be 32",
            ));
        }
        let prompt_narrow =
            if narrow >= crate::constants::PREFILL_LANES && (narrow < wide || !wide_prompt) {
                self.width = narrow;
                Some(self.build()?)
            } else {
                None
            };
        let prompt = if wide_prompt && wide >= crate::constants::PREFILL_LANES {
            self.width = wide;
            Some(self.build()?)
        } else {
            None
        };
        Ok((batch, prompt, prompt_narrow))
    }

    /// State positions a fused program writes, relative to its `RoPE` positions.
    fn state_offset(weights: &ProgramWeights) -> i32 {
        if weights.fusion.is_some() {
            -i32::try_from(crate::constants::MTP_KV_OFFSET).unwrap_or(0)
        } else {
            0
        }
    }

    /// Decode-only graph over per-slot state sets on one shared batched arena: Linear
    /// nodes read and write their `[width, size]` slots in place (no pack/unpack), pure
    /// activation nodes record once over every row, and state or metadata kernels record
    /// per lane against `lane_states[lane]`/`lane_fp8[lane]`. The builder's own
    /// `states`/`fp8_states` are placeholders
    /// and stay untouched.
    /// # Errors
    /// Rejects mismatched lane slices, a width outside `2..=CB_DECODE_SLOTS` or CUDA
    /// capture failures.
    pub fn build_slots(
        &mut self,
        lane_states: &mut [States],
        lane_fp8: &mut [Fp8Caches],
    ) -> Result<SlotDecodeGraph> {
        let width = lane_states.len();
        if !(2..=crate::constants::CB_DECODE_SLOTS).contains(&width)
            || lane_fp8.len() != width
            || self.lane_external.len() < width
            || self.lane_fusion.len() < width
        {
            return Err(Error::invalid("slot decode lanes"));
        }
        let mut arena = slot_arena(self.device, self.graph, width)?;
        let metadata = slot_metadata(self.device, width)?;
        let mut profile = profile::SlotProfile::from_env(self.device);
        let mut capture = |read_logits: bool| {
            CudaGraph::scope(&self.device.stream, |scope| {
                let mut no_fusion = None;
                for (index, node) in self.graph.nodes.iter().enumerate() {
                    if !read_logits && node.outputs.iter().any(|id| Some(*id) == self.graph.logits)
                    {
                        continue;
                    }
                    if read_logits && let Some(profile) = profile.as_mut() {
                        profile.graph().boundary(scope, index, node)?;
                    }
                    if node.op == TensorOp::Linear {
                        super::batch_projection::record_slots(
                            scope,
                            node,
                            &mut arena,
                            self.weights,
                            self.nvfp4,
                        )?;
                        continue;
                    }
                    match dispatch32(node, self.weights) {
                        Dispatch32::Linear => return Err(error("slot decode linear dispatch")),
                        Dispatch32::Batched => Capture {
                            scope,
                            arena: &mut arena,
                            weights: self.weights,
                            nvfp4: self.nvfp4,
                            attention: self.attention,
                            states: self.states,
                            fp8_states: self.fp8_states,
                            metadata: &metadata[0],
                            external: self.external,
                            capacity: self.capacity,
                            fusion: &mut no_fusion,
                            mode: CaptureMode::Batched,
                        }
                        .record(node)?,
                        Dispatch32::Row => {
                            for lane in 0..width {
                                Capture {
                                    scope,
                                    arena: &mut arena,
                                    weights: self.weights,
                                    nvfp4: self.nvfp4,
                                    attention: self.attention,
                                    states: &mut lane_states[lane],
                                    fp8_states: &mut lane_fp8[lane],
                                    metadata: &metadata[lane],
                                    external: &self.lane_external[lane],
                                    capacity: self.capacity,
                                    fusion: &mut self.lane_fusion[lane],
                                    mode: CaptureMode::Row(lane),
                                }
                                .record_row(node, lane)?;
                            }
                        }
                    }
                }
                if read_logits && let Some(profile) = profile.as_mut() {
                    profile.graph().finish(scope)?;
                }
                Ok(())
            })
            .map_err(device_error)
        };
        let graph = capture(true)?;
        let state_only = if self.weights.fusion.is_some() {
            Some(capture(false)?)
        } else {
            None
        };
        let hidden = take_result(&mut arena, self.graph.hidden)?;
        let logits = take_result(&mut arena, self.graph.logits)?;
        Ok(SlotDecodeGraph {
            graph,
            state_only,
            readbacks: crate::device::Readbacks::default(),
            metadata,
            hidden,
            external_hidden: self.weights.fusion.is_some(),
            logits,
            width,
            cursors: vec![0; width],
            profile,
            _arena: arena,
        })
    }

    pub fn build(&mut self) -> Result<BatchGraph> {
        let width = self.width;
        if !(2..=crate::constants::MAX_VERIFICATION_WIDTH).contains(&width)
            && ![
                crate::constants::PREFILL_LANES,
                crate::constants::MID_PREFILL_LANES,
                crate::constants::MAX_PREFILL_LANES,
                crate::constants::WIDE_PREFILL_LANES,
            ]
            .contains(&width)
        {
            return Err(Error::invalid("verification width"));
        }
        if width >= crate::constants::PREFILL_LANES {
            return self.build32();
        }
        let mut lanes = self.allocate_lanes(width)?;
        let mut projections = super::batch_projection::allocate(self.device, self.graph, width)?;
        let graph = Some(self.capture(&mut lanes, &mut projections, None)?);
        let prefill = self.capture(&mut lanes, &mut projections, Some(false))?;
        let prefill_last = self.capture(&mut lanes, &mut projections, Some(true))?;
        let mut restore = Vec::new();
        let mut metadata = Vec::new();
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        for mut lane in lanes {
            restore.push(
                CudaGraph::scope(&self.device.stream, |scope| {
                    for (id, tensors) in &lane.checkpoints {
                        for (output, input) in self
                            .states
                            .get_mut(id)
                            .ok_or_else(|| error("checkpoint state"))?
                            .iter_mut()
                            .zip(tensors)
                        {
                            scope.record(api::memcpy(output, input))?;
                        }
                    }
                    Ok(())
                })
                .map_err(device_error)?,
            );
            hidden.push(take_result(&mut lane.arena, self.graph.hidden)?);
            logits.push(take_result(&mut lane.arena, self.graph.logits)?);
            metadata.push(lane.metadata);
        }
        Ok(BatchGraph {
            graph,
            state_offset: Self::state_offset(self.weights),
            readbacks: crate::device::Readbacks::default(),
            prefill,
            prefill_last,
            restore,
            _partials: BTreeMap::new(),
            _recurrent: super::recurrent_prefill::Workspace::default(),
            prefill_info: None,
            metadata,
            hidden,
            logits,
            width,
            capacity: self.capacity,
            profile: None,
            _arena: None,
        })
    }

    /// Prefill width 32: one batched arena, one metadata tensor per lane.
    fn build32(&mut self) -> Result<BatchGraph> {
        let mut arena = ActivationArena::new_batched(
            self.device,
            self.graph,
            self.width,
            usize::try_from(self.device.profile()?.arena_budget_bytes()).map_err(device_error)?,
        )?;
        let mut metadata = Vec::with_capacity(self.width);
        for _ in 0..self.width {
            metadata.push(
                api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
            );
        }
        let mut partials = BTreeMap::new();
        for node in &self.graph.nodes {
            if node.op != TensorOp::Linear
                || self.weights.input_scales.contains_key(&node.inputs[1])
                || !matches!(
                    self.weights.projections.get(&node.inputs[1]),
                    Some(ProjectionWeight::Fp4(..))
                )
            {
                continue;
            }
            let rows = self
                .graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == node.outputs[0])
                .ok_or_else(|| Error::invalid("prefill linear output"))?
                .elements()?;
            if partials.contains_key(&rows) {
                continue;
            }
            partials.insert(
                rows,
                api::zeros::<f32>(&[crate::constants::PREFILL_SPLIT_K * self.width, rows])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
            );
        }
        let prefill_info = api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
            .sync_on(&self.device.stream)
            .map_err(device_error)?;
        let mut recurrent = super::recurrent_prefill::Workspace::new(
            self.device,
            self.graph,
            self.weights,
            self.width,
        )?;
        let mut profile = profile::PrefillProfile::from_env(self.device);
        let prefill = self.capture32(
            &mut arena,
            &metadata,
            &prefill_info,
            false,
            profile.as_mut(),
            (&mut partials, &mut recurrent),
        )?;
        let prefill_last = self.capture32(
            &mut arena,
            &metadata,
            &prefill_info,
            true,
            profile.as_mut(),
            (&mut partials, &mut recurrent),
        )?;
        let hidden = vec![take_result(&mut arena, self.graph.hidden)?];
        let logits = vec![take_result(&mut arena, self.graph.logits)?];
        Ok(BatchGraph {
            graph: None,
            state_offset: Self::state_offset(self.weights),
            readbacks: crate::device::Readbacks::default(),
            prefill,
            prefill_last,
            restore: Vec::new(),
            _partials: partials,
            _recurrent: recurrent,
            prefill_info: Some(prefill_info),
            metadata,
            hidden,
            logits,
            width: self.width,
            capacity: self.capacity,
            profile,
            _arena: Some(arena),
        })
    }

    fn capture(
        &mut self,
        lanes: &mut [Lane],
        projections: &mut super::batch_projection::Workspace,
        prefill: Option<bool>,
    ) -> Result<CudaGraph<()>> {
        let width = lanes.len();
        CudaGraph::scope(&self.device.stream, |scope| {
            for node in &self.graph.nodes {
                let logits = node.outputs.iter().any(|id| Some(*id) == self.graph.logits);
                if logits && prefill == Some(false) {
                    continue;
                }
                if width > 1 && node.op == TensorOp::Linear && !(logits && prefill.is_some()) {
                    super::batch_projection::record(
                        scope,
                        node,
                        lanes,
                        self.weights,
                        projections,
                        self.nvfp4,
                    )?;
                    continue;
                }
                for (index, lane) in lanes.iter_mut().enumerate() {
                    if logits && prefill.is_some() && index + 1 != width {
                        continue;
                    }
                    Capture {
                        scope,
                        arena: &mut lane.arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: &lane.metadata,
                        external: &self.lane_external[index],
                        capacity: self.capacity,
                        fusion: &mut self.lane_fusion[index],
                        mode: CaptureMode::Flat,
                    }
                    .record(node)?;
                    if prefill.is_none()
                        && let Some(id) = node.states.first()
                        && let Some(checkpoint) = lane.checkpoints.get_mut(id)
                    {
                        for (output, input) in checkpoint.iter_mut().zip(&self.states[id]) {
                            scope.record(api::memcpy(output, input))?;
                        }
                    }
                }
            }
            Ok(())
        })
        .map_err(device_error)
    }

    /// One graph covering all 32 lanes: Linear reads batched slots directly,
    /// shape-agnostic auxiliaries record once, state/metadata kernels stay per-lane.
    fn capture32(
        &mut self,
        arena: &mut ActivationArena,
        metadata: &[Tensor<i32>],
        prefill_info: &Tensor<i32>,
        last: bool,
        profile: Option<&mut profile::PrefillProfile>,
        workspaces: (
            &mut BTreeMap<usize, Tensor<f32>>,
            &mut super::recurrent_prefill::Workspace,
        ),
    ) -> Result<CudaGraph<()>> {
        let (partials, recurrent) = workspaces;
        CudaGraph::scope(&self.device.stream, |scope| {
            let mut boundaries = profile.map(|profile| profile.graph(last));
            let mut no_fusion = None;
            for (index, node) in self.graph.nodes.iter().enumerate() {
                let logits = node.outputs.iter().any(|id| Some(*id) == self.graph.logits);
                if logits && !last {
                    continue;
                }
                if let Some(boundaries) = boundaries.as_mut() {
                    boundaries.boundary(scope, index, node)?;
                }
                if recurrent.record(scope, node, arena, self.weights, self.states, prefill_info)? {
                    continue;
                }
                if matches!(node.op, TensorOp::Attention { .. } | TensorOp::Rope { .. })
                    && !node
                        .inputs
                        .iter()
                        .any(|id| self.weights.constants.contains_key(id))
                {
                    Capture {
                        scope,
                        arena: &mut *arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: prefill_info,
                        external: self.external,
                        capacity: self.capacity,
                        fusion: &mut no_fusion,
                        mode: CaptureMode::Prefill,
                    }
                    .record(node)?;
                    continue;
                }
                match dispatch32(node, self.weights) {
                    Dispatch32::Linear => {
                        super::prefill_projection::record(
                            scope,
                            node,
                            arena,
                            self.weights,
                            partials,
                            self.nvfp4,
                            self.delegated.as_deref_mut(),
                        )?;
                    }
                    Dispatch32::Batched => {
                        Capture {
                            scope,
                            arena: &mut *arena,
                            weights: self.weights,
                            nvfp4: self.nvfp4,
                            attention: self.attention,
                            states: self.states,
                            fp8_states: self.fp8_states,
                            metadata: &metadata[0],
                            external: self.external,
                            capacity: self.capacity,
                            fusion: &mut no_fusion,
                            mode: CaptureMode::Batched,
                        }
                        .record(node)?;
                    }
                    Dispatch32::Row => {
                        // Row kernels take scalar per-lane inputs, so a fused program records
                        // its embedding and normalization against that lane's own hidden.
                        for (lane, meta) in metadata.iter().enumerate().take(self.width) {
                            Capture {
                                scope,
                                arena: &mut *arena,
                                weights: self.weights,
                                nvfp4: self.nvfp4,
                                attention: self.attention,
                                states: self.states,
                                fp8_states: self.fp8_states,
                                metadata: meta,
                                external: &self.lane_external[lane],
                                capacity: self.capacity,
                                fusion: &mut self.lane_fusion[lane],
                                mode: CaptureMode::Row(lane),
                            }
                            .record_row(node, lane)?;
                        }
                    }
                }
            }
            if let Some(boundaries) = boundaries.as_mut() {
                boundaries.finish(scope)?;
            }
            Ok(())
        })
        .map_err(device_error)
    }

    fn allocate_lanes(&self, width: usize) -> Result<Vec<Lane>> {
        let mut lanes = Vec::new();
        let mut bytes = 0usize;
        for _ in 0..width {
            let mut checkpoints = BTreeMap::new();
            for spec in &self.graph.tensors {
                if !matches!(
                    spec.storage,
                    TensorStorage::State {
                        kind: StateKind::Conv | StateKind::LinearAttention,
                        ..
                    }
                ) {
                    continue;
                }
                let mut copies = Vec::new();
                for source in &self.states[&spec.id] {
                    bytes = source
                        .size()
                        .checked_mul(crate::constants::F32_BYTES)
                        .and_then(|n| bytes.checked_add(n))
                        .ok_or_else(|| Error::invalid("checkpoint overflow"))?;
                    if u64::try_from(bytes).map_err(device_error)?
                        > self.device.profile()?.checkpoint_budget_bytes()
                    {
                        return Err(Error::invalid("verification checkpoints exceed budget"));
                    }
                    let shape = source
                        .shape()
                        .iter()
                        .map(|v| usize::try_from(*v).map_err(device_error))
                        .collect::<Result<Vec<_>>>()?;
                    copies.push(
                        api::zeros::<f32>(&shape)
                            .sync_on(&self.device.stream)
                            .map_err(device_error)?,
                    );
                }
                checkpoints.insert(spec.id, copies);
            }
            lanes.push(Lane {
                arena: ActivationArena::new(
                    self.device,
                    self.graph,
                    usize::try_from(self.device.profile()?.arena_budget_bytes())
                        .map_err(device_error)?,
                )?,
                metadata: api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
                checkpoints,
            });
        }
        Ok(lanes)
    }
}

/// Node dispatch for prefill width 32 captures.
pub(super) enum Dispatch32 {
    Linear,
    Batched,
    Row,
}

/// Batched recording needs every data input as a `[32, size]` slot; a constant
/// data input is a single row and degrades the node to per-lane recording.
/// Norm weights stay shared constants and never degrade the node.
pub(super) fn dispatch32(node: &TensorNode, weights: &ProgramWeights) -> Dispatch32 {
    let constant = |index: usize| weights.constants.contains_key(&node.inputs[index]);
    let batched = match &node.op {
        TensorOp::Linear => return Dispatch32::Linear,
        TensorOp::Silu | TensorOp::Sigmoid | TensorOp::Add | TensorOp::Multiply => !node
            .inputs
            .iter()
            .any(|id| weights.constants.contains_key(id)),
        TensorOp::Norm { .. } | TensorOp::Split { .. } => !constant(0),
        TensorOp::GatedNorm { .. } => !constant(0) && !constant(1),
        _ => false,
    };
    if batched {
        Dispatch32::Batched
    } else {
        Dispatch32::Row
    }
}

pub(super) fn take_result(
    arena: &mut ActivationArena,
    id: Option<TensorId>,
) -> Result<Arc<Tensor<f32>>> {
    let slot = arena.slot(id.ok_or_else(|| Error::invalid("batch result binding"))?)?;
    Ok(Arc::new(
        arena.buffers[slot]
            .take()
            .ok_or_else(|| Error::invariant("batch result buffer"))?,
    ))
}

impl BatchGraph {
    pub fn run(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        prefill: Option<bool>,
        read_hidden: bool,
    ) -> Result<BatchOutput> {
        if self.width >= crate::constants::PREFILL_LANES {
            return self.run32(device, tokens, position, prefill, read_hidden);
        }
        let graph = match prefill {
            None => self
                .graph
                .as_ref()
                .ok_or_else(|| Error::invalid("prompt-only batch cannot verify"))?,
            Some(false) => &self.prefill,
            Some(true) => &self.prefill_last,
        };
        Self::update_metadata(
            graph,
            &mut self.metadata,
            self.width,
            self.capacity,
            tokens,
            position,
            self.state_offset,
        )?;
        let mut sources = Vec::with_capacity(2 * tokens.len());
        for lane in 0..tokens.len() {
            let logits = prefill.is_none() || prefill == Some(true) && lane + 1 == tokens.len();
            sources
                .push(read_hidden.then(|| ReadbackSource::whole(Arc::clone(&self.hidden[lane]))));
            sources.push(logits.then(|| ReadbackSource::whole(Arc::clone(&self.logits[lane]))));
        }
        let mut rows = self.readbacks.run(device, graph, &sources)?.into_iter();
        (0..tokens.len())
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("batch hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("batch logits readback"))?,
                ))
            })
            .collect()
    }

    /// State positions this graph lags its `RoPE` position by.
    #[must_use]
    pub const fn state_offset(&self) -> i32 {
        self.state_offset
    }

    /// Rebind one lane's external hidden source in the prompt graph.
    /// # Errors
    /// Returns a CUDA error if the graph update fails.
    pub fn bind_lane_external(
        &self,
        slot: &mut Tensor<f32>,
        uploaded: &Arc<Tensor<f32>>,
    ) -> Result<()> {
        self.prefill
            .update(api::memcpy(slot, uploaded))
            .map_err(device_error)
    }

    fn update_metadata(
        graph: &CudaGraph<()>,
        metadata: &mut [Tensor<i32>],
        width: usize,
        capacity: usize,
        tokens: &[u32],
        position: usize,
        state_offset: i32,
    ) -> Result<()> {
        for (lane, metadata) in metadata.iter_mut().enumerate().take(width) {
            let token = tokens.get(lane).copied().unwrap_or(0);
            let pos = i32::try_from(position + lane).map_err(device_error)?;
            let state_pos = if width >= crate::constants::PREFILL_LANES && lane >= tokens.len() {
                -1
            } else {
                pos.saturating_add(state_offset)
            };
            // A KV row past the arena would be written out of bounds by the state kernels, and
            // the earlier kernels only report that on a checked load. Refuse it here, where the
            // numbers are still host-side.
            // `-1` is the inactive sentinel the kernels skip; anything below it, or at or past
            // the arena depth, would address a row the state kernels do not check on store.
            if state_pos < -1 || state_pos >= i32::try_from(capacity).map_err(device_error)? {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    format!(
                        "state position {state_pos} (base {position}, lane {lane}, offset {state_offset}) \
                         exceeds the {capacity}-token KV arena"
                    ),
                ));
            }
            super::metadata::update(
                graph,
                metadata,
                [
                    pos,
                    i32::try_from(token).map_err(device_error)?,
                    state_pos,
                    0,
                ],
            )?;
        }
        Ok(())
    }

    /// Batched slots read back in one copy each; per-lane rows are split on host.
    fn run32(
        &mut self,
        device: &CudaDevice,
        tokens: &[u32],
        position: usize,
        prefill: Option<bool>,
        read_hidden: bool,
    ) -> Result<BatchOutput> {
        let graph = match prefill {
            Some(false) => &self.prefill,
            Some(true) => &self.prefill_last,
            None => return Err(Error::invalid("prompt-only batch cannot verify")),
        };
        Self::update_metadata(
            graph,
            &mut self.metadata,
            self.width,
            self.capacity,
            tokens,
            position,
            self.state_offset,
        )?;
        // The slot-decode path already refuses a state position past the arena
        // (`stage_lanes`); the prompt path has to as well, because the prefill append and
        // decode kernels address that many rows of KV without a checked load.
        let last_row = i64::try_from(position).map_err(device_error)?
            + i64::try_from(tokens.len()).map_err(device_error)?
            - 1
            + i64::from(self.state_offset);
        if last_row >= i64::try_from(self.capacity).map_err(device_error)? {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                format!(
                    "prompt state rows end at {last_row} (base {position}, {} tokens, offset {}), \
                     past the {}-token KV arena",
                    tokens.len(),
                    self.state_offset,
                    self.capacity
                ),
            ));
        }
        super::metadata::update(
            graph,
            self.prefill_info
                .as_mut()
                .ok_or_else(|| Error::invariant("prefill metadata"))?,
            [
                i32::try_from(position).map_err(device_error)?,
                i32::try_from(tokens.len()).map_err(device_error)?,
                self.state_offset,
                0,
            ],
        )?;
        let hidden_dim = self.hidden[0].size() / self.width;
        let vocabulary = self.logits[0].size() / self.width;
        let sources = [
            read_hidden.then(|| ReadbackSource {
                tensor: Arc::clone(&self.hidden[0]),
                skip: 0,
                len: tokens.len() * hidden_dim,
            }),
            (prefill == Some(true)).then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits[0]),
                skip: (tokens.len() - 1) * vocabulary,
                len: vocabulary,
            }),
        ];
        let rows = self.readbacks.run(device, graph, &sources);
        if rows.is_ok()
            && let Some(profile) = &self.profile
        {
            profile.report(prefill == Some(true), tokens.len(), position);
        }
        let mut rows = rows?.into_iter();
        let hidden_rows = rows
            .next()
            .ok_or_else(|| Error::invariant("batch hidden readback"))?;
        let mut logits = rows
            .next()
            .ok_or_else(|| Error::invariant("batch logits readback"))?;
        (0..tokens.len())
            .map(|lane| {
                let hidden = if read_hidden {
                    hidden_rows[lane * hidden_dim..(lane + 1) * hidden_dim].to_vec()
                } else {
                    Vec::new()
                };
                let logits = if lane + 1 == tokens.len() {
                    std::mem::take(&mut logits)
                } else {
                    Vec::new()
                };
                Ok((hidden, logits))
            })
            .collect()
    }

    pub fn restore(&self, device: &CudaDevice, count: usize) -> Result<()> {
        self.restore
            .get(count - 1)
            .ok_or_else(|| Error::invalid("batch commit count"))?
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)
    }
}

/// One active lane of a slot decode replay: slot index, token, `RoPE` position, state position.
pub(super) type SlotLane = (usize, u32, usize, usize);

impl SlotDecodeGraph {
    /// Validate the lane set against the cursors, then stage every slot's metadata for
    /// the next replay. `graph` only retains the metadata writes until that replay.
    #[allow(
        clippy::too_many_arguments,
        reason = "disjoint field borrows of the graph"
    )]
    fn stage_lanes(
        metadata: &mut [Tensor<i32>],
        cursors: &[usize],
        width: usize,
        external_hidden: bool,
        graph: &CudaGraph<()>,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
    ) -> Result<()> {
        if lanes.is_empty() || lanes.len() > width {
            return Err(Error::invalid("slot decode lane count"));
        }
        let mut per_slot = vec![None; width];
        for &(slot, token, rope_pos, state_pos) in lanes {
            if slot >= width
                || per_slot[slot].is_some()
                || usize::try_from(token).map_err(device_error)? >= vocabulary
                || state_pos >= capacity
            {
                return Err(Error::invalid("slot decode slot, token or position"));
            }
            if state_pos != cursors[slot] {
                return Err(Error::invalid("nonsequential slot state position"));
            }
            per_slot[slot] = Some((token, rope_pos, state_pos));
        }
        for (slot, metadata) in metadata.iter_mut().enumerate() {
            let values = match per_slot[slot] {
                Some((token, rope_pos, state_pos)) => [
                    i32::try_from(rope_pos).map_err(device_error)?,
                    i32::try_from(token).map_err(device_error)?,
                    i32::try_from(state_pos).map_err(device_error)?,
                    i32::from(external_hidden),
                ],
                None => [0, 0, crate::constants::INACTIVE_LANE_STATE_POSITION, 0],
            };
            super::metadata::update(graph, metadata, values)?;
        }
        Ok(())
    }

    /// Replay one decode step for the active lanes in `lanes` (output order); every other
    /// lane is masked with a `-1` state position so its slot's state stays untouched.
    ///
    /// Each active lane's state position must equal the slot cursor (same nonsequential
    /// rejection as the single-step path) and advances it by one on success. Returns one
    /// `(hidden, logits)` pair per active lane; external-hidden draft graphs return
    /// hidden rows for their next step, while target decode uses logits-only readout.
    /// # Errors
    /// Rejects empty/oversized lane sets, duplicate or unknown slots, out-of-vocabulary
    /// tokens, out-of-capacity state positions, stale cursors and CUDA replay failures.
    pub fn run_lanes(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
        read_logits: bool,
    ) -> Result<BatchOutput> {
        let graph = if read_logits {
            &self.graph
        } else {
            self.state_only
                .as_ref()
                .ok_or_else(|| Error::invalid("state-only draft graph"))?
        };
        Self::stage_lanes(
            &mut self.metadata,
            &self.cursors,
            self.width,
            self.external_hidden,
            graph,
            lanes,
            capacity,
            vocabulary,
        )?;
        let vocabulary = self.logits.size() / self.width;
        let mut sources = Vec::with_capacity(2 * lanes.len());
        for &(slot, ..) in lanes {
            sources.push(
                (read_logits && self.external_hidden).then(|| ReadbackSource {
                    tensor: Arc::clone(&self.hidden),
                    skip: slot * (self.hidden.size() / self.width),
                    len: self.hidden.size() / self.width,
                }),
            );
            sources.push(read_logits.then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: slot * vocabulary,
                len: vocabulary,
            }));
        }
        let rows = self.readbacks.run(device, graph, &sources)?;
        if read_logits
            && let Some(profile) = &self.profile
            && let Some(&(.., rope_pos, _)) = lanes.first()
        {
            profile.report("slot_decode", lanes.len(), rope_pos);
        }
        let mut rows = rows.into_iter();
        for &(slot, ..) in lanes {
            self.cursors[slot] += 1;
        }
        lanes
            .iter()
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot logits readback"))?,
                ))
            })
            .collect()
    }

    /// Replay the state-only graph for `lanes` without a readback or a stream sync: a
    /// draft catch-up needs the state writes, never the outputs. Completion is carried
    /// by the next readback on this stream or the pool's drop barrier.
    /// # Errors
    /// Rejects the same lane/position errors as [`Self::run_lanes`], a missing state-only
    /// graph, and CUDA launch failures detected at enqueue time.
    #[expect(
        unsafe_code,
        reason = "Audited detached replay: single-stream ordering after staged metadata, \
                  pool-owned buffers, drained by the owning pool's drop barrier"
    )]
    pub fn replay_detached(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
    ) -> Result<()> {
        let graph = self
            .state_only
            .as_ref()
            .ok_or_else(|| Error::invalid("state-only draft graph"))?;
        Self::stage_lanes(
            &mut self.metadata,
            &self.cursors,
            self.width,
            self.external_hidden,
            graph,
            lanes,
            capacity,
            vocabulary,
        )?;
        // SAFETY: the engine is single-stream, so this replay is ordered after the staged
        // metadata and uploads and before any later replay; every buffer it touches is
        // pool-owned, and the pool's drop barrier drains the stream before anything frees.
        unsafe { graph.launch().async_on(&device.stream) }.map_err(device_error)?;
        for &(slot, ..) in lanes {
            self.cursors[slot] += 1;
        }
        Ok(())
    }

    /// Drain the capture stream: the pool's teardown barrier for detached replays.
    /// # Errors
    /// Returns the driver error when the stream cannot be synchronized.
    #[expect(
        unsafe_code,
        reason = "Audited stream drain at teardown: no work is enqueued after the owning \
                  pool starts dropping"
    )]
    pub(super) fn barrier(&self) -> Result<()> {
        // SAFETY: teardown only; no work is enqueued after the owning pool starts dropping.
        unsafe { self.graph.stream().synchronize() }.map_err(device_error)
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        &self.cursors
    }

    /// Seed a slot's cursor after binding a sequence's covered prefix into it.
    /// # Errors
    /// Rejects unknown slot indices.
    pub fn set_cursor(&mut self, slot: usize, position: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("slot decode cursor"))?;
        *cursor = position;
        Ok(())
    }
}

fn slot_metadata(device: &CudaDevice, width: usize) -> Result<Vec<Tensor<i32>>> {
    (0..width)
        .map(|_| {
            api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                .sync_on(&device.stream)
                .map_err(device_error)
        })
        .collect()
}

fn slot_arena(device: &CudaDevice, graph: &DataflowGraph, width: usize) -> Result<ActivationArena> {
    ActivationArena::new_batched(
        device,
        graph,
        width,
        usize::try_from(device.profile()?.arena_budget_bytes()).map_err(device_error)?,
    )
}
