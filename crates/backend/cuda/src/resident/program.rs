use super::{ActivationArena, capture};
use crate::{
    constants::{
        CONV_KERNEL_SIZE, CONV_STATE_TENSORS, F32_BYTES, MAX_CAPACITY_TOKENS, MAX_HIDDEN_SIZE,
        METADATA_FIELDS, PREFILL_LANES,
    },
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use std::collections::BTreeMap;

pub struct FusionWeights {
    pub projection: ProjectionWeight,
    /// Tile for the fusion projection, selected like every other projection weight.
    pub tiling: LinearTiling,
    pub norms: Arc<Tensor<f32>>,
    pub epsilon: f32,
    pub offset: f32,
}

impl FusionWeights {
    fn validate(&self, hidden: usize) -> Result<()> {
        self.projection.validate(hidden, 2 * hidden)?;
        if self.norms.shape() != [2, i32::try_from(hidden).map_err(device_error)?]
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.offset.is_finite()
        {
            return Err(Error::invalid("MTP fusion dimensions or normalization"));
        }
        Ok(())
    }
}

pub(super) struct FusionWorkspace {
    pub embedding: Tensor<f32>,
    pub normalized: Option<Tensor<f32>>,
}

impl FusionWorkspace {
    pub(super) fn allocate(
        device: &CudaDevice,
        hidden: usize,
        enabled: bool,
    ) -> Result<Option<Self>> {
        let fusion = if enabled {
            Some(Self {
                embedding: api::zeros::<f32>(&[hidden])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
                normalized: Some(
                    api::zeros::<f32>(&[2 * hidden])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                ),
            })
        } else {
            None
        };
        Ok(fusion)
    }
}

pub struct ProgramWeights {
    pub batch_width: usize,
    /// Widest captured prompt graph.
    pub prefill_width: usize,
    /// Second, narrower prompt graph captured alongside [`Self::prefill_width`]; zero when the
    /// program captures one prompt graph. A prompt chunk costs one fixed-width replay, so a
    /// 51-token prompt wants the narrow graph while a 511-token prompt wants the wide one.
    pub narrow_prefill_width: usize,
    pub kv_scales: BTreeMap<TensorId, [f32; 2]>,
    pub tiling: BTreeMap<TensorId, LinearTiling>,
    pub fusion: Option<FusionWeights>,
    pub projections: BTreeMap<TensorId, ProjectionWeight>,
    pub input_scales: BTreeMap<TensorId, f32>,
    pub fp8_inputs: std::collections::BTreeSet<TensorId>,
    pub constants: BTreeMap<TensorId, Arc<Tensor<f32>>>,
    pub embeddings: BTreeMap<TensorId, Arc<Tensor<bf16>>>,
    pub rope_axes: BTreeMap<infer_core::OpId, Arc<Tensor<i32>>>,
    pub rope_frequencies: BTreeMap<infer_core::OpId, Arc<Tensor<f32>>>,
}

#[derive(Clone, Copy)]
pub(super) enum ActivationQuantization {
    Fp4(f32),
    Fp8Token,
}
impl ProgramWeights {
    /// Prompt graph widths this program captures, widest first; empty when it captures none.
    ///
    /// A width below `PREFILL_LANES` is not a prompt graph: it is the verification width the
    /// program falls back to.
    pub fn prompt_widths(&self) -> Vec<usize> {
        let mut widths = Vec::with_capacity(2);
        for width in [self.prefill_width, self.narrow_prefill_width] {
            if width >= PREFILL_LANES && !widths.contains(&width) {
                widths.push(width);
            }
        }
        widths
    }

    /// Widest captured prompt graph, or zero when the program captures none.
    pub fn widest_prompt_width(&self) -> usize {
        self.prompt_widths().into_iter().max().unwrap_or(0)
    }

    /// Total prompt lanes across every captured prompt graph.
    pub fn prompt_lane_total(&self) -> usize {
        self.prompt_widths().into_iter().sum()
    }

    pub(super) fn activation_quantization(&self, id: TensorId) -> Option<ActivationQuantization> {
        self.input_scales
            .get(&id)
            .copied()
            .map(ActivationQuantization::Fp4)
            .or_else(|| {
                self.fp8_inputs
                    .contains(&id)
                    .then_some(ActivationQuantization::Fp8Token)
            })
    }
}

/// Captures a complete single-token forward pass; all state and activations stay on CUDA.
pub struct DeviceProgram {
    batch: Option<super::batch::BatchGraph>,
    last_batch: Option<(usize, usize)>,
    prompt_batch: Option<super::batch::BatchGraph>,
    /// Second, narrower prompt graph; chunks at most this wide replay here instead.
    prompt_narrow: Option<super::batch::BatchGraph>,
    graph: CudaGraph<()>,
    readbacks: crate::device::Readbacks,
    prefill: CudaGraph<()>,
    metadata: Tensor<i32>,
    nvfp4: super::nvfp4_gemm::Workspace,
    attention: super::attention_decode::Workspace,
    external: Tensor<f32>,
    // Declared after the graphs: the state tensors outlive every capture that references them,
    // and prefix caching reads them back for snapshots.
    states: super::batch::States,
    fp8_states: super::fp8_cache::Fp8Caches,
    lane_external: Vec<Tensor<f32>>,
    lane_uploads: Vec<Arc<Tensor<f32>>>,
    // Fusion workspaces are referenced by the captured graphs and never touched afterwards.
    lane_fusion: Vec<Option<FusionWorkspace>>,
    reset_graph: CudaGraph<()>,
    hidden: Arc<Tensor<f32>>,
    logits: Arc<Tensor<f32>>,
    device: CudaDevice,
    capacity: usize,
    vocabulary: usize,
    next_position: usize,
    attention_only: bool,
    requires_external: bool,
    pub activation_bytes: usize,
    pub(crate) reclaimable_bytes: u64,
    released_verification_bytes: u64,
}

/// Conservative lower bound on uniquely owned device storage; excludes graph metadata,
/// readback buffers and auxiliary workspaces. Idle-state admission can reclaim this much.
fn retained_tensor_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    states: &super::batch::States,
    fp8: &super::fp8_cache::Fp8Caches,
    activation_bytes: usize,
) -> u64 {
    let verify = if weights.batch_width > 1 {
        weights.batch_width
    } else {
        0
    };
    let prompt = weights.prompt_lane_total();
    let activations = activation_bytes as u64 * (1 + verify + prompt) as u64;
    let state_bytes = states
        .values()
        .flatten()
        .map(|t| t.size() as u64 * F32_BYTES as u64)
        .sum::<u64>();
    let packed_bytes = fp8
        .values()
        .map(|(k, v)| (k.size() + v.size()) as u64)
        .sum::<u64>();
    let recurrent = graph
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                t.storage,
                TensorStorage::State {
                    kind: StateKind::Conv | StateKind::LinearAttention,
                    ..
                }
            )
        })
        .filter_map(|t| states.get(&t.id))
        .flatten()
        .map(|t| t.size() as u64 * F32_BYTES as u64)
        .sum::<u64>();
    activations + state_bytes + packed_bytes + recurrent * verify as u64
}

/// Per-lane external hidden buffers paired with their fusion workspaces.
type LaneFusion = (Vec<Tensor<f32>>, Vec<Option<FusionWorkspace>>);

/// Per-lane external hidden buffers and fusion workspaces for batched captures.
///
/// A fused program consumes one external hidden per lane; the batch capture owns a buffer and
/// a workspace per lane so batched draft priming no longer replays the prompt token by token.
fn allocate_lane_fusion(
    device: &CudaDevice,
    hidden: usize,
    weights: &ProgramWeights,
) -> Result<LaneFusion> {
    let width = weights.prefill_width.max(weights.batch_width).max(1);
    let enabled = weights.fusion.is_some();
    let mut external = Vec::new();
    let mut fusion = Vec::new();
    for _ in 0..width {
        external.push(
            api::zeros::<f32>(&[hidden])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        );
        fusion.push(FusionWorkspace::allocate(device, hidden, enabled)?);
    }
    Ok((external, fusion))
}

/// Activation arena budget for this device, as a host `usize`.
fn arena_budget(device: &CudaDevice) -> Result<usize> {
    usize::try_from(device.profile()?.arena_budget_bytes()).map_err(device_error)
}
/// Validate one program request and return the padded capacity it will be captured at.
fn program_capacity(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    capacity: usize,
    hidden: usize,
    vocabulary: usize,
) -> Result<usize> {
    if capacity == 0
        || capacity > MAX_CAPACITY_TOKENS
        || vocabulary == 0
        || hidden == 0
        || hidden > MAX_HIDDEN_SIZE
    {
        return Err(Error::invalid("resident capacity or vocabulary"));
    }
    if graph
        .nodes
        .iter()
        .any(|node| node.inputs.iter().any(|id| Some(*id) == graph.logits))
    {
        return Err(Error::unsupported("logits must be a terminal graph output"));
    }
    if let Some(fusion) = &weights.fusion {
        fusion.validate(hidden)?;
    }
    Ok(capacity.next_power_of_two())
}

impl DeviceProgram {
    /// A leased sequence verifies through the shared pool. Its private rollback
    /// graph is idle and can be rebuilt if the sequence later returns to serial use.
    pub(crate) fn discard_verification(&mut self) -> bool {
        if self.batch.take().is_none() {
            return false;
        }
        self.last_batch = None;
        let before = self.reclaimable_bytes;
        let state_bytes = self
            .states
            .values()
            .flatten()
            .map(|t| t.size() as u64 * F32_BYTES as u64)
            .sum::<u64>();
        let packed_bytes = self
            .fp8_states
            .values()
            .map(|(k, v)| (k.size() + v.size()) as u64)
            .sum::<u64>();
        let prompt: usize = [self.prompt_batch.as_ref(), self.prompt_narrow.as_ref()]
            .into_iter()
            .flatten()
            .map(|g| g.width)
            .sum();
        self.reclaimable_bytes =
            state_bytes + packed_bytes + self.activation_bytes as u64 * (1 + prompt) as u64;
        self.released_verification_bytes = before - self.reclaimable_bytes;
        true
    }

    pub(crate) const fn released_verification_bytes(&self) -> u64 {
        self.released_verification_bytes
    }

    pub(crate) fn ensure_verification(
        &mut self,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
    ) -> Result<()> {
        if self.batch.is_some() || weights.batch_width < 2 {
            return Ok(());
        }
        let mut builder = super::batch::BatchBuilder {
            device: &self.device,
            graph,
            weights,
            nvfp4: &mut self.nvfp4,
            attention: &mut self.attention,
            states: &mut self.states,
            fp8_states: &mut self.fp8_states,
            external: &self.external,
            lane_external: &mut self.lane_external,
            lane_fusion: &mut self.lane_fusion,
            capacity: self.capacity,
            width: weights.batch_width,
        };
        self.batch = Some(builder.build()?);
        self.reclaimable_bytes = retained_tensor_bytes(
            graph,
            weights,
            &self.states,
            &self.fp8_states,
            self.activation_bytes,
        );
        self.released_verification_bytes = 0;
        Ok(())
    }

    /// # Errors
    /// Rejects unsupported graph shapes, storage budgets or CUDA capture failures.
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        capacity: usize,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        super::validation::validate(graph, weights)?;
        let capacity = program_capacity(graph, weights, capacity, hidden, vocabulary)?;
        let mut arena = ActivationArena::new(device, graph, arena_budget(device)?)?;
        let activation_bytes = arena.bytes();
        let mut attention = super::attention_decode::Workspace::new(device, graph, capacity)?;
        let mut nvfp4 = super::nvfp4_gemm::Workspace::program(device, graph, weights)?;
        let mut states = allocate_states(device, graph, capacity, weights)?;
        let mut fp8_states =
            super::fp8_cache::allocate(device, graph, capacity, &weights.kv_scales)?;
        let reclaimable_bytes =
            retained_tensor_bytes(graph, weights, &states, &fp8_states, activation_bytes);
        let reset_graph = capture_reset(device, &mut states)?;
        let metadata = device.upload(vec![0_i32; METADATA_FIELDS * 2], &[METADATA_FIELDS * 2])?;
        let metadata =
            Arc::try_unwrap(metadata).map_err(|_| Error::invariant("unique metadata"))?;
        let external = api::zeros::<f32>(&[hidden])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut fusion = FusionWorkspace::allocate(device, hidden, weights.fusion.is_some())?;
        let (mut lane_external, mut lane_fusion) = allocate_lane_fusion(device, hidden, weights)?;
        let mut capture_graph = |skip_logits: bool| {
            CudaGraph::scope(&device.stream, |scope| {
                let mut capture = capture::Capture {
                    scope,
                    arena: &mut arena,
                    weights,
                    nvfp4: &mut nvfp4,
                    attention: &mut attention,
                    states: &mut states,
                    fp8_states: &mut fp8_states,
                    metadata: &metadata,
                    external: &external,
                    capacity,
                    fusion: &mut fusion,
                    mode: capture::CaptureMode::Flat,
                };
                for node in &graph.nodes {
                    if !skip_logits || !node.outputs.iter().any(|id| Some(*id) == graph.logits) {
                        capture.record(node)?;
                    }
                }
                Ok(())
            })
            .map_err(device_error)
        };
        let graph_exec = capture_graph(false)?;
        let prefill = capture_graph(true)?;
        let mut builder = super::batch::BatchBuilder {
            device,
            graph,
            weights,
            nvfp4: &mut nvfp4,
            attention: &mut attention,
            states: &mut states,
            fp8_states: &mut fp8_states,
            external: &external,
            lane_external: &mut lane_external,
            lane_fusion: &mut lane_fusion,
            capacity,
            width: weights.batch_width,
        };
        let (batch, prompt_batch, prompt_narrow) = builder.build_pair()?;
        let hidden = super::batch::take_result(&mut arena, graph.hidden)?;
        let logits = super::batch::take_result(&mut arena, graph.logits)?;
        Ok(Self {
            nvfp4,
            attention,
            batch,
            prompt_batch,
            prompt_narrow,
            last_batch: None,
            graph: graph_exec,
            readbacks: crate::device::Readbacks::default(),
            prefill,
            metadata,
            external,
            lane_external,
            lane_uploads: Vec::new(),
            states,
            fp8_states,
            lane_fusion,
            reset_graph,
            hidden,
            logits,
            device: device.clone(),
            capacity,
            vocabulary,
            next_position: 0,
            requires_external: weights.fusion.is_some(),
            attention_only: attention_only(graph),
            activation_bytes,
            reclaimable_bytes,
            released_verification_bytes: u64::default(),
        })
    }

    /// # Errors
    /// Rejects out-of-range positions/tokens, invalid overrides or CUDA failures.
    pub fn step(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.step_readout(token, position, kv_position, external, read_logits, true)
    }

    pub(crate) fn step_readout(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.execute_step(
            token,
            [position; infer_models::mrope::ROTARY_AXES],
            kv_position,
            external,
            read_logits,
            read_hidden,
        )
    }

    /// Execute with independent time/height/width coordinates and a sequential KV index.
    /// # Errors
    /// Rejects invalid tokens, coordinates, overrides or device operations.
    pub fn step_positioned(
        &mut self,
        token: u32,
        position: [usize; infer_models::mrope::ROTARY_AXES],
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.execute_step(token, position, kv_position, external, read_logits, true)
    }

    fn execute_step(
        &mut self,
        token: u32,
        position: [usize; infer_models::mrope::ROTARY_AXES],
        kv_position: usize,
        external: Option<&[f32]>,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if kv_position >= self.capacity
            || usize::try_from(token).map_err(device_error)? >= self.vocabulary
        {
            return Err(Error::invalid(
                "resident token or KV position out of bounds",
            ));
        }
        if self.requires_external && external.is_none() {
            return Err(Error::invalid("MTP requires previous hidden state"));
        }
        if external.is_some_and(|values| values.len() != self.external.size()) {
            return Err(Error::invalid("external hidden dimensions"));
        }
        if kv_position == 0 {
            self.reset()?;
        }
        if kv_position != self.next_position {
            return Err(Error::invalid("nonsequential device state position"));
        }
        let metadata = [
            i32::try_from(position[0]).map_err(device_error)?,
            i32::try_from(token).map_err(device_error)?,
            i32::try_from(kv_position).map_err(device_error)?,
            i32::from(external.is_some()),
        ];
        let graph = if read_logits {
            &self.graph
        } else {
            &self.prefill
        };
        super::metadata::update_rotary(graph, &mut self.metadata, metadata, position)?;
        if let Some(values) = external {
            if values.len() != self.external.size() {
                return Err(Error::invalid("external hidden dimensions"));
            }
            let uploaded = self.device.upload(values.to_vec(), &[values.len()])?;
            graph
                .update(api::memcpy(&mut self.external, &uploaded))
                .map_err(device_error)?;
        }
        let mut rows = self
            .readbacks
            .run(
                &self.device,
                graph,
                &[
                    read_hidden
                        .then(|| crate::device::ReadbackSource::whole(Arc::clone(&self.hidden))),
                    read_logits
                        .then(|| crate::device::ReadbackSource::whole(Arc::clone(&self.logits))),
                ],
            )?
            .into_iter();
        let result = (
            rows.next()
                .ok_or_else(|| Error::invariant("hidden readback"))?,
            rows.next()
                .ok_or_else(|| Error::invariant("logits readback"))?,
        );
        self.last_batch = None;
        self.next_position = kv_position + 1;
        Ok(result)
    }

    #[must_use]
    pub fn batch_width(&self) -> usize {
        self.batch.as_ref().map_or(1, |batch| batch.width)
    }

    /// Execute several target inputs in node order, then retain checkpoints for prefix commit.
    /// # Errors
    /// Rejects invalid positions, unsupported widths or failed device execution.
    pub fn step_batch(&mut self, tokens: &[u32], position: usize) -> Result<super::BatchOutput> {
        let width = self.batch_width();
        if tokens.is_empty()
            || tokens.len() > width
            || self.batch.is_none()
            || position
                .checked_add(width)
                .is_none_or(|end| end > self.capacity)
            || tokens
                .iter()
                .any(|t| usize::try_from(*t).map_or(true, |n| n >= self.vocabulary))
        {
            return Err(Error::invalid("verification tokens, width or capacity"));
        }
        if position == 0 {
            self.reset()?;
        }
        if position != self.next_position {
            return Err(Error::invalid("verification state position"));
        }
        let output = self
            .batch
            .as_mut()
            .ok_or_else(|| Error::invariant("batch graph"))?
            .run(&self.device, tokens, position, None, true)?;
        self.next_position = position + width;
        self.last_batch = Some((position, tokens.len()));
        self.commit_batch(tokens.len())?;
        Ok(output)
    }

    /// Seed the logical position after a prefix restore, so the next prefill resumes there.
    pub const fn set_position(&mut self, position: usize) {
        self.next_position = position;
    }

    /// Resident state tensors of this program, one vector per state id.
    ///
    /// Prefix caching snapshots these with device-to-device copies and restores them into a
    /// later sequence, so the buffers must stay reachable after capture.
    #[must_use]
    pub const fn states(&self) -> &super::batch::States {
        &self.states
    }

    /// Mutable resident state tensors, for prefix restore.
    #[must_use]
    pub const fn states_mut(&mut self) -> &mut super::batch::States {
        &mut self.states
    }

    /// FP8 scale side tables paired with the resident state tensors.
    #[must_use]
    pub const fn fp8_states(&self) -> &super::fp8_cache::Fp8Caches {
        &self.fp8_states
    }

    /// Mutable quantized KV storage for a prefix restore.
    #[must_use]
    pub const fn fp8_states_mut(&mut self) -> &mut super::fp8_cache::Fp8Caches {
        &mut self.fp8_states
    }

    #[must_use]
    pub fn prefill_width(&self) -> usize {
        let widest = self
            .prompt_batch
            .as_ref()
            .map_or(0, |prompt| prompt.width)
            .max(self.prompt_narrow.as_ref().map_or(0, |prompt| prompt.width));
        if widest > 0 {
            widest
        } else {
            self.batch_width()
        }
    }

    /// Whether a chunk of `lanes` tokens replays on the narrow prompt graph.
    ///
    /// Both captured prompt graphs are fixed width, so the narrow one is preferred whenever the
    /// chunk fits it: running a 51-token chunk through a 128-lane graph spends 60% of every
    /// replay on masked rows.
    fn prefers_narrow_prompt(&self, lanes: usize) -> bool {
        self.prompt_narrow
            .as_ref()
            .is_some_and(|prompt| lanes <= prompt.width)
    }

    /// State positions this program writes behind its `RoPE` positions.
    fn batch_state_lag(&self) -> Result<usize> {
        let offset = self
            .prompt_batch
            .as_ref()
            .or(self.prompt_narrow.as_ref())
            .or(self.batch.as_ref())
            .map_or(0, super::batch::BatchGraph::state_offset);
        usize::try_from(offset.unsigned_abs()).map_err(device_error)
    }

    /// Ingest a prompt chunk with one external hidden per lane, as a fused draft needs.
    /// # Errors
    /// Rejects a short hidden slice or a missing prompt graph, then reports CUDA failures.
    pub(crate) fn prime_batch(
        &mut self,
        tokens: &[u32],
        position: usize,
        externals: &[f32],
    ) -> Result<()> {
        let hidden = self.external.size();
        let lanes = tokens
            .len()
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("prompt externals"))?;
        if externals.len() != lanes {
            return Err(Error::invalid("prompt externals"));
        }
        self.lane_uploads.clear();
        let narrow_fits = self.prefers_narrow_prompt(tokens.len());
        let batch = if narrow_fits {
            self.prompt_narrow.take()
        } else {
            self.prompt_batch.take()
        }
        .ok_or_else(|| Error::invariant("prompt batch graph"))?;
        for lane in 0..tokens.len() {
            let slice = externals
                .get(lane * hidden..(lane + 1) * hidden)
                .ok_or_else(|| Error::invalid("prompt externals"))?;
            let uploaded = self.device.upload(slice.to_vec(), &[hidden])?;
            batch.bind_lane_external(&mut self.lane_external[lane], &uploaded)?;
            self.lane_uploads.push(uploaded);
        }
        if narrow_fits {
            self.prompt_narrow = Some(batch);
        } else {
            self.prompt_batch = Some(batch);
        }
        self.prefill_batch_readout(tokens, position, false, false)?;
        Ok(())
    }

    /// Ingest prompt tokens without rollback copies; 32-token graphs mask inactive tail lanes.
    /// # Errors
    /// Rejects invalid widths, positions/tokens or CUDA failures.
    pub fn prefill_batch(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
    ) -> Result<super::BatchOutput> {
        self.prefill_batch_readout(tokens, position, read_logits, true)
    }

    pub(crate) fn prefill_batch_readout(
        &mut self,
        tokens: &[u32],
        position: usize,
        read_logits: bool,
        read_hidden: bool,
    ) -> Result<super::BatchOutput> {
        let width = self.prefill_width();
        if (self.batch.is_none() && self.prompt_batch.is_none() && self.prompt_narrow.is_none())
            || tokens.is_empty()
            || tokens.len() > width
            || (width < PREFILL_LANES && tokens.len() != width)
            || position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.capacity)
            || tokens
                .iter()
                .any(|t| usize::try_from(*t).map_or(true, |n| n >= self.vocabulary))
        {
            return Err(Error::invalid("prefill tokens, width or capacity"));
        }
        let lag = self.batch_state_lag()?;
        if position == 0 {
            self.reset()?;
        }
        if position.saturating_sub(lag) != self.next_position {
            return Err(Error::invalid("prefill state position"));
        }
        let lanes = tokens.len();
        let prompt = if self.prefers_narrow_prompt(lanes) {
            self.prompt_narrow.as_mut()
        } else {
            self.prompt_batch.as_mut()
        };
        let output = prompt
            .or_else(|| self.batch.as_mut())
            .ok_or_else(|| Error::invariant("prefill batch graph"))?
            .run(
                &self.device,
                tokens,
                position,
                Some(read_logits),
                read_hidden,
            )?;
        let end = position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::invalid("prefill position overflow"))?;
        self.next_position = end.saturating_sub(lag);
        self.last_batch = None;
        Ok(output)
    }

    /// Commit a nonempty prefix of the most recent verification batch.
    /// # Errors
    /// Rejects forward commits, stale checkpoints or failed state restoration.
    pub fn commit_batch(&mut self, count: usize) -> Result<()> {
        let (start, available) = self
            .last_batch
            .ok_or_else(|| Error::invalid("no pending verification batch"))?;
        if count == 0 || count > available {
            return Err(Error::invalid("invalid verification prefix"));
        }
        if self.next_position != start + count {
            self.batch
                .as_ref()
                .ok_or_else(|| Error::invariant("batch graph"))?
                .restore(&self.device, count)?;
        }
        self.next_position = start + count;
        self.last_batch = Some((start, count));
        Ok(())
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.next_position
    }

    /// Validate an externally executed attention prefix before copying its state.
    pub(super) fn external_attention_capacity(&self, position: usize) -> Result<usize> {
        if !self.attention_only || position > self.capacity {
            return Err(Error::invalid("external attention prefix is incompatible"));
        }
        Ok(self.capacity)
    }
    /// Called only after slot state has been copied into this program's buffers.
    pub(super) fn adopt_attention_prefix(&mut self, position: usize) -> Result<()> {
        self.external_attention_capacity(position)?;
        self.next_position = position;
        self.last_batch = None;
        Ok(())
    }

    /// Rewind an attention-only draft. Replay overwrites the suffix; attention masks stale rows.
    /// # Errors
    /// Rejects recurrent state and forward skips, which need a different rollback strategy.
    pub fn rewind_attention(&mut self, position: usize) -> Result<()> {
        if !self.attention_only || position > self.next_position {
            return Err(Error::invalid("invalid attention-only rollback"));
        }
        self.next_position = position;
        Ok(())
    }

    /// # Errors
    /// Returns failed CUDA state resets; graph addresses remain unchanged.
    pub fn reset(&mut self) -> Result<()> {
        self.next_position = 0;
        self.last_batch = None;
        self.reset_graph
            .launch()
            .sync_on(&self.device.stream)
            .map_err(device_error)?;
        Ok(())
    }
}

fn capture_reset(
    device: &CudaDevice,
    states: &mut BTreeMap<TensorId, Vec<Tensor<f32>>>,
) -> Result<CudaGraph<()>> {
    CudaGraph::scope(&device.stream, |scope| {
        for tensors in states.values_mut() {
            for _ in 0..tensors.len() {
                let tensor = tensors.remove(0);
                let shape = tensor
                    .shape()
                    .iter()
                    .map(|v| usize::try_from(*v).map_err(capture::error))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let size = tensor.size();
                let mut tensor = tensor.reshape(&[size])?;
                scope.record(super::kernels::aux::zero(
                    (&mut tensor).partition([crate::constants::AUX_KERNEL_TILE]),
                ))?;
                tensors.push(tensor.reshape(&shape)?);
            }
        }
        Ok(())
    })
    .map_err(device_error)
}

pub fn allocate_states(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    weights: &ProgramWeights,
) -> Result<BTreeMap<TensorId, Vec<Tensor<f32>>>> {
    let mut plan = Vec::new();
    let mut total = 0usize;
    let budget = device
        .memory_info()?
        .0
        .saturating_sub(device.profile()?.device_headroom_bytes());
    for spec in &graph.tensors {
        let TensorStorage::State { kind, .. } = &spec.storage else {
            continue;
        };
        if weights.kv_scales.contains_key(&spec.id) {
            continue;
        }
        let shapes = match kind {
            StateKind::Conv if spec.shape.len() == 2 && spec.shape[1] == CONV_KERNEL_SIZE => {
                vec![vec![spec.shape[0]]; CONV_STATE_TENSORS]
            }
            StateKind::LinearAttention => vec![spec.shape.clone()],
            StateKind::AttentionKv if spec.shape.len() == 2 => {
                vec![
                    vec![
                        capacity
                            .checked_mul(spec.shape[1])
                            .ok_or_else(|| Error::invalid("KV allocation overflow"))?
                    ];
                    2
                ]
            }
            _ => return Err(Error::unsupported("resident state recipe")),
        };
        for shape in &shapes {
            let elements = shape.iter().try_fold(1usize, |a, b| {
                a.checked_mul(*b)
                    .ok_or_else(|| Error::invalid("state overflow"))
            })?;
            total = total
                .checked_add(elements)
                .ok_or_else(|| Error::invalid("state budget overflow"))?;
            if total as u64 > budget / F32_BYTES as u64 {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    "resident F32 state exceeds available device memory minus 1 GiB headroom",
                ));
            }
        }
        plan.push((spec.id, shapes));
    }
    let mut states = BTreeMap::new();
    for (id, shapes) in plan {
        let tensors = shapes
            .iter()
            .map(|shape| {
                api::zeros::<f32>(shape)
                    .sync_on(&device.stream)
                    .map_err(device_error)
            })
            .collect::<Result<Vec<_>>>()?;
        states.insert(id, tensors);
    }
    Ok(states)
}

fn attention_only(graph: &DataflowGraph) -> bool {
    graph.tensors.iter().all(|spec| {
        !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::Conv | StateKind::LinearAttention,
                ..
            }
        )
    })
}
