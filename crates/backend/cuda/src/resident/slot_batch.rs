//! Continuous-batching slot pool: `width` pre-allocated state sets and one shared
//! decode graph whose lane `i` is baked onto slot `i`. A sequence binds a slot by
//! device-copying its covered state rows in, then every leased sequence advances
//! through a single graph replay per step.
use super::{
    batch::{BatchOutput, SlotDecodeGraph, SlotLane, States},
    fp8_cache::Fp8Caches,
    program::ProgramWeights,
    slot_verify::{SlotVerifyGraph, SlotVerifyLane},
};
use crate::device::{CudaDevice, device_error};
use cuda_core::{CudaContext, PinnedHostBuffer};
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorStorage};

/// Pinned ring depth per external-hidden lane: one upload per draft or catch-up step,
/// and a lane's chain between stream barriers never exceeds the maximum verify width.
const EXTERNAL_UPLOAD_RING: usize = crate::constants::MAX_VERIFICATION_WIDTH + 1;

/// Per-lane external hidden rows and their pinned upload rings.
type ExternalLanes = (Vec<Tensor<f32>>, Vec<Vec<PinnedHostBuffer<f32>>>);

/// How one state family moves into a slot at bind time: a token-row prefix of a
/// capacity-sized KV cache (f32 or fp8), or the whole tensor (conv/delta state).
/// KV layout is discovered from the captured tensors at bind time, never assumed.
#[derive(Clone, Copy)]
enum SlotCopy {
    Kv,
    KvFp8,
    Whole,
}

/// Dimensions of the captured head-major KV state layout `[kv_heads, capacity, head_dim]`.
const KV_STATE_DIMS: usize = 3;

/// Copy plan over the dataflow graph's state tensors, in declaration order.
fn copy_plan(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<Vec<(TensorId, SlotCopy)>> {
    let mut plan = Vec::new();
    for spec in &graph.tensors {
        let TensorStorage::State { kind, .. } = &spec.storage else {
            continue;
        };
        let fp8 = weights.kv_scales.contains_key(&spec.id);
        let copy = match (kind, fp8) {
            (StateKind::AttentionKv, false) if spec.shape.len() == 2 => SlotCopy::Kv,
            (StateKind::AttentionKv, true) if spec.shape.len() == 2 => SlotCopy::KvFp8,
            (StateKind::Conv | StateKind::LinearAttention, false) => SlotCopy::Whole,
            _ => return Err(Error::unsupported("slot pool state recipe")),
        };
        plan.push((spec.id, copy));
    }
    Ok(plan)
}

/// Device bytes one slot's state tensors occupy: f32 states plus fp8 KV caches.
fn state_bytes(states: &States, fp8: &Fp8Caches) -> u64 {
    let float_elements: u64 = states
        .values()
        .flatten()
        .map(|tensor| tensor.size() as u64)
        .sum();
    let packed_elements: u64 = fp8
        .values()
        .map(|(keys, values)| (keys.size() + values.size()) as u64)
        .sum();
    float_elements.saturating_mul(crate::constants::F32_BYTES as u64) + packed_elements
}

pub struct SlotPool {
    // Exactly one of the two graphs exists: plain decode without a draft, pooled
    // speculation with one. Declared first so the graphs drop before the state sets and
    // arena buffers their kernels reference.
    decode: Option<SlotDecodeGraph>,
    verify: Option<SlotVerifyGraph>,
    states: Vec<States>,
    fp8: Vec<Fp8Caches>,
    /// Per-slot identity block tables; declared with the states so they outlive the captures.
    tables: Vec<Tensor<i32>>,
    plan: Vec<(TensorId, SlotCopy)>,
    free: Vec<usize>,
    slots: usize,
    capacity: usize,
    vocabulary: usize,
    // The captured graph bakes these buffers' device pointers; declared last so the
    // graph drops before anything it references.
    _external: Tensor<f32>,
    _nvfp4: super::nvfp4_gemm::Workspace,
    _attention: super::attention_decode::Workspace,
    lane_external: Vec<Tensor<f32>>,
    /// Whether a lane's external hidden row was uploaded at least once.
    external_uploaded: Vec<bool>,
    /// Pinned staging rings feeding `lane_external`: uploads enqueue without a sync, so a
    /// ring slot must not be rewritten while an earlier copy from it could still be in
    /// flight. The ring outlasts the longest unsynced upload chain (a draft catch-up).
    pinned_external: Vec<Vec<PinnedHostBuffer<f32>>>,
    pinned_cursor: Vec<usize>,
    _lane_fusion: Vec<Option<super::program::FusionWorkspace>>,
}

impl SlotPool {
    /// Allocate `width` slots of `capacity` tokens each and capture the shared graph.
    /// `verify` ≥ 2 captures the pooled speculation graph with `verify` candidate lanes
    /// per slot; `verify` < 2 captures the plain decode graph.
    /// # Errors
    /// Rejects invalid geometry, fused verification graphs, an exceeded device budget
    /// or CUDA capture failures; the caller disables batching on any error.
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
        capacity: usize,
        hidden: usize,
        output: (usize, usize),
    ) -> Result<Self> {
        let (vocabulary, verify) = output;
        if !(2..=crate::constants::MAX_VERIFICATION_WIDTH).contains(&width)
            || capacity == 0
            || capacity > crate::constants::MAX_CAPACITY_TOKENS
            || hidden == 0
            || vocabulary == 0
        {
            return Err(Error::invalid("slot pool width, capacity or geometry"));
        }
        if weights.fusion.is_some() && verify >= 2 {
            return Err(Error::unsupported(
                "pooled verification requires unfused target weights",
            ));
        }
        let plan = copy_plan(graph, weights)?;
        let budget = device
            .memory_info()?
            .0
            .saturating_sub(device.profile()?.device_headroom_bytes());
        let mut states = Vec::with_capacity(width);
        let mut fp8 = Vec::with_capacity(width);
        let mut tables = Vec::with_capacity(width);
        let mut bytes = 0_u64;
        for _ in 0..width {
            let slot_states = super::program::allocate_states(device, graph, capacity, weights)?;
            let slot_fp8 = super::fp8_cache::allocate(device, graph, capacity, &weights.kv_scales)?;
            bytes = bytes
                .checked_add(state_bytes(&slot_states, &slot_fp8))
                .ok_or_else(|| Error::invalid("slot pool budget overflow"))?;
            if bytes > budget {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    "slot decode pool exceeds available device memory minus headroom",
                ));
            }
            states.push(slot_states);
            fp8.push(slot_fp8);
            // Step 1 of the KV refactor: a per-slot identity mapping over that slot's own cache.
            tables.push(super::metadata::identity_table(device, capacity)?);
        }
        let external = api::zeros::<f32>(&[hidden])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let lanes = if verify >= 2 { width * verify } else { width };
        let (mut lane_external, pinned_external) =
            Self::external_lanes(device, weights.fusion.is_some(), lanes, hidden)?;
        let mut lane_fusion: Vec<Option<super::program::FusionWorkspace>> = (0..lanes)
            .map(|_| {
                super::program::FusionWorkspace::allocate(device, hidden, weights.fusion.is_some())
            })
            .collect::<Result<_>>()?;
        let mut attention = super::attention_decode::Workspace::new(device, graph, capacity)?;
        let mut nvfp4 = super::nvfp4_gemm::Workspace::new(device, graph, weights, &[1, lanes])?;
        let mut placeholder_states = States::new();
        let mut placeholder_fp8 = Fp8Caches::new();
        let (decode, verify_graph) = {
            let mut builder = super::batch::BatchBuilder {
                device,
                graph,
                weights,
                nvfp4: &mut nvfp4,
                delegated: None,
                attention: &mut attention,
                states: &mut placeholder_states,
                fp8_states: &mut placeholder_fp8,
                external: &external,
                lane_external: &mut lane_external,
                lane_fusion: &mut lane_fusion,
                tables: &tables,
                capacity,
                width,
            };
            if verify >= 2 {
                (
                    None,
                    Some(builder.build_slot_verify(&mut states, &mut fp8, verify)?),
                )
            } else {
                (Some(builder.build_slots(&mut states, &mut fp8)?), None)
            }
        };
        Ok(Self {
            decode,
            verify: verify_graph,
            states,
            fp8,
            tables,
            plan,
            free: (0..width).rev().collect(),
            slots: width,
            capacity,
            vocabulary,
            _external: external,
            _nvfp4: nvfp4,
            _attention: attention,
            lane_external,
            external_uploaded: vec![false; lanes],
            pinned_external,
            pinned_cursor: vec![0; lanes],
            _lane_fusion: lane_fusion,
        })
    }

    /// Per-lane external hidden rows plus, for an external-hidden pool, their pinned
    /// upload rings. Uploads enqueue without a sync, so a ring slot must not be rewritten
    /// while an earlier copy from it could still be in flight; the ring depth outlasts
    /// the longest unsynced upload chain (a draft catch-up).
    fn external_lanes(
        device: &CudaDevice,
        external_hidden: bool,
        lanes: usize,
        hidden: usize,
    ) -> Result<ExternalLanes> {
        let mut lane_external = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            lane_external.push(
                api::zeros::<f32>(&[hidden])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            );
        }
        let context = external_hidden
            .then(|| CudaContext::new(device.stream.device().ordinal()))
            .transpose()
            .map_err(device_error)?;
        let mut pinned_external = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            let ring = match &context {
                Some(context) => (0..EXTERNAL_UPLOAD_RING)
                    .map(|_| PinnedHostBuffer::zeroed(context, hidden).map_err(device_error))
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            };
            pinned_external.push(ring);
        }
        Ok((lane_external, pinned_external))
    }

    /// Drain in-flight detached work before the pool's buffers free.
    fn barrier(&self) -> Result<()> {
        match (&self.decode, &self.verify) {
            (Some(graph), _) => graph.barrier(),
            (None, Some(graph)) => graph.barrier(),
            (None, None) => Ok(()),
        }
    }

    /// Slots (state sets) in the pool.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.slots
    }

    /// Candidate lanes per slot when the pool speculates (`None` on a plain decode pool).
    #[must_use]
    pub fn verify(&self) -> Option<usize> {
        self.verify.as_ref().map(|graph| graph.verify)
    }

    /// Token rows each slot's KV cache holds.
    #[must_use]
    /// Whether a sequence can still be leased a slot, so that its state is already reserved.
    pub(crate) fn has_free_slot(&self) -> bool {
        !self.free.is_empty()
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        match (&self.decode, &self.verify) {
            (Some(graph), _) => graph.cursors(),
            (None, Some(graph)) => graph.cursors(),
            (None, None) => &[],
        }
    }

    /// Take a free slot, if any remain.
    pub fn take_free(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// Return a slot to the free set; the next bind overwrites its state prefix.
    pub fn release(&mut self, slot: usize) {
        if slot < self.width() && !self.free.contains(&slot) {
            self.free.push(slot);
            if let Some(graph) = self.verify.as_mut() {
                graph.discard(slot);
            }
        }
    }

    /// Copy a sequence's covered state into `slot`: KV token rows up to `rows` (both sides
    /// are `[kv_heads, capacity, head_dim]` with their own capacity, so each head block
    /// copies separately), conv/delta and fp8 tensors whole. The slot cursor is seeded to
    /// `rows` for the next replay.
    /// # Errors
    /// Rejects unknown slots, prefixes beyond the slot capacity, layout mismatches
    /// between source and slot tensors, or driver copy failures.
    pub fn bind(
        &mut self,
        slot: usize,
        source: &States,
        source_fp8: &Fp8Caches,
        rows: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        if slot >= self.width() || rows > self.capacity {
            return Err(Error::invalid("slot bind index or prefix length"));
        }
        // Step 1 of the KV refactor: the kernels resolve KV positions through this table, so it
        // must cover the slot's whole capacity. Step 2 replaces the identity contents with the
        // allocator's blocks and rewrites them here on every bind.
        let blocks = i32::try_from(self.capacity.div_ceil(crate::constants::KV_BLOCK_TOKENS))
            .map_err(|_| Error::invalid("slot block count"))?;
        if self
            .tables
            .get(slot)
            .is_none_or(|table| table.shape() != [blocks])
        {
            return Err(Error::invariant("slot block table shape"));
        }
        for (id, copy) in &self.plan {
            match *copy {
                SlotCopy::Kv => Self::copy_kv_states(
                    self.states
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot states"))?,
                    source,
                    *id,
                    rows,
                    self.capacity,
                    device,
                )?,
                SlotCopy::KvFp8 => Self::copy_kv_fp8(
                    self.fp8
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot fp8 caches"))?,
                    source_fp8,
                    *id,
                    rows,
                    self.capacity,
                    device,
                )?,
                SlotCopy::Whole => Self::copy_whole(
                    self.states
                        .get_mut(slot)
                        .ok_or_else(|| Error::invariant("slot states"))?,
                    source,
                    *id,
                    device,
                )?,
            }
        }
        match (&mut self.decode, &mut self.verify) {
            (Some(graph), _) => graph.set_cursor(slot, rows)?,
            (None, Some(graph)) => graph.set_cursor(slot, rows)?,
            (None, None) => return Err(Error::invariant("slot pool graph")),
        }
        Ok(())
    }

    /// Replay one decode step for the active lanes; see [`SlotDecodeGraph::run_lanes`].
    /// # Errors
    /// Rejects invalid lanes or positions, a speculation pool and CUDA replay failures.
    pub fn run_lanes(&mut self, device: &CudaDevice, lanes: &[SlotLane]) -> Result<BatchOutput> {
        if self
            .decode
            .as_ref()
            .is_some_and(|graph| graph.external_hidden)
            && lanes
                .iter()
                .any(|&(slot, ..)| !self.external_uploaded.get(slot).copied().unwrap_or(false))
        {
            return Err(Error::invalid("draft lane has no external hidden input"));
        }
        self.decode
            .as_mut()
            .ok_or_else(|| Error::invalid("slot pool has no decode graph"))?
            .run_lanes(device, lanes, self.capacity, self.vocabulary, true)
    }

    /// Replay one pooled speculation step; see [`SlotVerifyGraph::run_verify`].
    /// # Errors
    /// Rejects invalid lanes or positions, a decode-only pool and CUDA replay failures.
    pub fn run_verify(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
    ) -> Result<BatchOutput> {
        self.verify
            .as_mut()
            .ok_or_else(|| Error::invalid("slot pool has no verify graph"))?
            .run_verify(device, lanes, self.capacity, self.vocabulary)
    }

    /// Commit one slot's accepted candidates; see [`SlotVerifyGraph::commit`].
    /// # Errors
    /// Rejects unknown slots, out-of-range counts and CUDA restore failures.
    pub fn commit_verify(
        &mut self,
        device: &CudaDevice,
        slot: usize,
        accepted: usize,
    ) -> Result<()> {
        self.verify
            .as_mut()
            .ok_or_else(|| Error::invalid("slot pool has no verify graph"))?
            .commit(device, slot, accepted)
    }

    /// Copy the covered token-row prefix of one KV state family into the slot.
    fn copy_kv_states(
        dst: &mut States,
        src: &States,
        id: TensorId,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        let src_tensors = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source state"))?;
        let dst_tensors = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target state"))?;
        if src_tensors.len() != dst_tensors.len() {
            return Err(Error::invariant("slot bind state tensors"));
        }
        for (dst, src) in dst_tensors.iter_mut().zip(src_tensors) {
            Self::copy_kv_tensor(dst, src, rows, slot_capacity, device)?;
        }
        Ok(())
    }

    /// fp8 twin of [`Self::copy_kv_states`].
    fn copy_kv_fp8(
        dst: &mut Fp8Caches,
        src: &Fp8Caches,
        id: TensorId,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        let (src_keys, src_values) = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source fp8 cache"))?;
        let (dst_keys, dst_values) = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target fp8 cache"))?;
        for (dst, src) in [(dst_keys, src_keys), (dst_values, src_values)] {
            Self::copy_kv_tensor(dst, src, rows, slot_capacity, device)?;
        }
        Ok(())
    }

    /// One KV tensor's covered rows. Captured KV state is head-major
    /// `[kv_heads, capacity, head_dim]` and the two sides differ in capacity, so the
    /// token prefix moves per head block with each side's own stride. Anything but that
    /// exact layout is an invariant violation, never a silent partial copy.
    fn copy_kv_tensor<T: DType>(
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        rows: usize,
        slot_capacity: usize,
        device: &CudaDevice,
    ) -> Result<()> {
        let dims = |tensor: &Tensor<T>| {
            tensor
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(device_error))
                .collect::<Result<Vec<_>>>()
        };
        let dst_dims = dims(dst)?;
        let src_dims = dims(src)?;
        if dst_dims.len() != KV_STATE_DIMS
            || src_dims.len() != KV_STATE_DIMS
            || dst_dims[0] != src_dims[0]
            || dst_dims[2] != src_dims[2]
            || dst_dims[1] != slot_capacity
            || src_dims[1] < rows
        {
            return Err(Error::invariant("slot bind KV layout"));
        }
        let head_dim = dst_dims[2];
        let dst_stride = slot_capacity
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("slot bind copy overflow"))?;
        let src_stride = src_dims[1]
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("slot bind copy overflow"))?;
        let elements = rows
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("slot bind copy overflow"))?;
        for head in 0..dst_dims[0] {
            device.copy_d2d_at(
                dst,
                src,
                head.checked_mul(dst_stride)
                    .ok_or_else(|| Error::invalid("slot bind copy overflow"))?,
                head.checked_mul(src_stride)
                    .ok_or_else(|| Error::invalid("slot bind copy overflow"))?,
                elements,
            )?;
        }
        Ok(())
    }

    /// Copy one conv/delta state family whole; both sides share one capacity-free layout.
    fn copy_whole(dst: &mut States, src: &States, id: TensorId, device: &CudaDevice) -> Result<()> {
        let src_tensors = src
            .get(&id)
            .ok_or_else(|| Error::invariant("slot bind source state"))?;
        let dst_tensors = dst
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("slot bind target state"))?;
        if src_tensors.len() != dst_tensors.len() {
            return Err(Error::invariant("slot bind state tensors"));
        }
        for (dst, src) in dst_tensors.iter_mut().zip(src_tensors) {
            if src.size() != dst.size() {
                return Err(Error::invariant("slot bind tensor size"));
            }
            device.copy_d2d(dst, src, src.size())?;
        }
        Ok(())
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        // Detached draft catch-up replays and verify restores may still be in flight on
        // the graph stream; the pool's states and arenas must not free beneath them.
        if let Err(error) = self.barrier() {
            eprintln!("CUDA slot pool drop barrier failed: {error}");
        }
    }
}

mod draft;

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_batch_draft_tests.rs"]
mod draft_tests;
