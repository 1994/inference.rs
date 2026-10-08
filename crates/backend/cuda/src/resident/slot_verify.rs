//! Cross-sequence target verification with per-slot recurrent rollback.
use super::{
    ActivationArena,
    batch::{BatchBuilder, BatchOutput, Dispatch32, States, dispatch32, take_result},
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
    profile,
};
use crate::constants::MIB;
use crate::device::{CudaDevice, ReadbackSource, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{StateKind, TensorOp, TensorStorage};

/// Pooled speculation graph: `slots` state sets × `verify` contiguous candidate lanes each,
/// flattened into one batched arena so every weight still loads once per replay. Lane
/// `slot * verify + position` verifies position `cursors[slot] + position` of its slot's own
/// state set — the vLLM-style flattening of (sequence × draft depth) into the batch axis.
///
/// Conv/delta snapshots are recorded after every lane but each slot's last, so a partial
/// acceptance rolls one slot back to its longest accepted prefix without touching the other
/// slots; KV rollback is the cursor rewind, exactly like the single-sequence verify graph.
pub(super) struct SlotVerifyGraph {
    graph: CudaGraph<()>,
    /// `restore[slot][accepted]` reinstates the conv/delta snapshot taken after that many
    /// candidate lanes of the slot committed; a fully accepted chain never restores.
    restore: Vec<Vec<CudaGraph<()>>>,
    readbacks: crate::device::Readbacks,
    metadata: Vec<Tensor<i32>>,
    hidden: Arc<Tensor<f32>>,
    logits: Arc<Tensor<f32>>,
    /// Slot state sets the graph interleaves.
    pub slots: usize,
    /// Candidate lanes each slot runs per replay (submitted token plus draft proposals).
    pub verify: usize,
    /// Host-side state position each slot's next replay continues at.
    cursors: Vec<usize>,
    pending: Vec<usize>,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graph so its executable is destroyed before the events it records.
    profile: Option<profile::SlotProfile>,
    // The shared batched arena anchors the graph's activations; declared last so the graph
    // and the restore graphs drop before it.
    _arena: ActivationArena,
}

/// One active lane of a slot verify replay: slot, candidate position inside the slot's
/// chain, token, `RoPE` position and state position.
pub(super) type SlotVerifyLane = (usize, usize, u32, usize, usize);

impl BatchBuilder<'_> {
    fn slot_checkpoints(&self, lane_states: &[States], verify: usize) -> Result<Vec<Vec<States>>> {
        let budget = usize::try_from(self.device.profile()?.checkpoint_budget_bytes())
            .map_err(device_error)?;
        let mut checkpoints: Vec<Vec<States>> = Vec::with_capacity(lane_states.len());
        let mut bytes = 0usize;
        for slot_states in lane_states {
            let mut per_slot = Vec::with_capacity(verify - 1);
            for _ in 0..verify - 1 {
                let mut snapshot = States::new();
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
                    for source in slot_states
                        .get(&spec.id)
                        .ok_or_else(|| Error::invariant("slot verify checkpoint source"))?
                    {
                        bytes = source
                            .size()
                            .checked_mul(crate::constants::F32_BYTES)
                            .and_then(|n| bytes.checked_add(n))
                            .ok_or_else(|| Error::invalid("slot verify checkpoint overflow"))?;
                        if bytes > budget {
                            return Err(Error::new(
                                infer_core::ErrorCode::Capacity,
                                format!(
                                    "slot verify checkpoints exceed budget: need {} MiB, \
                                     budget {} MiB at {} slots x {} lanes",
                                    bytes / MIB,
                                    budget / MIB,
                                    lane_states.len(),
                                    verify
                                ),
                            ));
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
                    snapshot.insert(spec.id, copies);
                }
                per_slot.push(snapshot);
            }
            checkpoints.push(per_slot);
        }
        Ok(checkpoints)
    }

    fn capture_slot_verify(
        &mut self,
        lane_states: &mut [States],
        lane_fp8: &mut [Fp8Caches],
        geometry: (usize, &[Tensor<i32>]),
        arena: &mut ActivationArena,
        checkpoints: &mut [Vec<States>],
        profile: &mut Option<profile::SlotProfile>,
    ) -> Result<CudaGraph<()>> {
        let (verify, metadata) = geometry;
        CudaGraph::scope(&self.device.stream, |scope| {
            let mut no_fusion = None;
            for (index, node) in self.graph.nodes.iter().enumerate() {
                if let Some(profile) = profile.as_mut() {
                    profile.graph().boundary(scope, index, node)?;
                }
                if node.op == TensorOp::Linear {
                    super::batch_projection::record_slots(
                        scope,
                        node,
                        arena,
                        self.weights,
                        self.nvfp4,
                    )?;
                    continue;
                }
                match dispatch32(node, self.weights) {
                    Dispatch32::Linear => return Err(error("slot verify linear dispatch")),
                    Dispatch32::Batched => Capture {
                        scope,
                        arena,
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
                        for (lane, lane_metadata) in metadata.iter().enumerate() {
                            let slot = lane / verify;
                            let position = lane % verify;
                            Capture {
                                scope,
                                arena,
                                weights: self.weights,
                                nvfp4: self.nvfp4,
                                attention: self.attention,
                                states: &mut lane_states[slot],
                                fp8_states: &mut lane_fp8[slot],
                                metadata: lane_metadata,
                                external: &self.lane_external[lane],
                                capacity: self.capacity,
                                fusion: &mut self.lane_fusion[lane],
                                mode: CaptureMode::Row(lane),
                            }
                            .record_row(node, lane)?;
                            // Snapshot after this lane's write: the restore target for an
                            // acceptance ending here. The last lane of a slot is never one —
                            // full acceptance keeps the live state.
                            if position + 1 < verify
                                && let Some(id) = node.states.first()
                                && let Some(snapshot) = checkpoints
                                    .get_mut(slot)
                                    .and_then(|per_slot| per_slot.get_mut(position))
                                    .and_then(|snapshot| snapshot.get_mut(id))
                            {
                                let states = lane_states
                                    .get(slot)
                                    .and_then(|states| states.get(id))
                                    .ok_or_else(|| error("slot verify checkpoint state"))?;
                                for (output, input) in snapshot.iter_mut().zip(states) {
                                    scope.record(api::memcpy(output, input))?;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(profile) = profile.as_mut() {
                profile.graph().finish(scope)?;
            }
            Ok(())
        })
        .map_err(device_error)
    }

    /// Capture the pooled speculation graph over per-slot state sets. Lane rows of the
    /// shared arena feed the row-grouped fused GEMV directly; pure activation nodes record
    /// once across every lane; state, metadata and checkpoint copies record per lane
    /// against `lane_states[lane / verify]`.
    /// # Errors
    /// Rejects mismatched lane slices, a slot count outside `2..=CB_DECODE_SLOTS`, a verify
    /// width outside `2..=MAX_VERIFICATION_WIDTH`, an exceeded checkpoint budget or CUDA
    /// capture failures.
    pub fn build_slot_verify(
        &mut self,
        lane_states: &mut [States],
        lane_fp8: &mut [Fp8Caches],
        verify: usize,
    ) -> Result<SlotVerifyGraph> {
        let slots = lane_states.len();
        let lanes = slots
            .checked_mul(verify)
            .ok_or_else(|| Error::invalid("slot verify width overflow"))?;
        if !(2..=crate::constants::CB_DECODE_SLOTS).contains(&slots)
            || !(2..=crate::constants::MAX_VERIFICATION_WIDTH).contains(&verify)
            || lane_fp8.len() != slots
            || self.lane_external.len() < lanes
            || self.lane_fusion.len() < lanes
        {
            return Err(Error::invalid("slot verify lanes"));
        }
        let mut checkpoints = self.slot_checkpoints(lane_states, verify)?;
        let mut arena = ActivationArena::new_batched(
            self.device,
            self.graph,
            lanes,
            usize::try_from(self.device.profile()?.arena_budget_bytes()).map_err(device_error)?,
        )?;
        let mut metadata = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            metadata.push(
                api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                    .sync_on(&self.device.stream)
                    .map_err(device_error)?,
            );
        }
        let mut profile = profile::SlotProfile::from_env(self.device);
        let graph = self.capture_slot_verify(
            lane_states,
            lane_fp8,
            (verify, &metadata),
            &mut arena,
            &mut checkpoints,
            &mut profile,
        )?;
        let mut restore = Vec::with_capacity(slots);
        for (slot, per_slot) in checkpoints.iter().enumerate().take(slots) {
            let mut graphs = Vec::with_capacity(per_slot.len());
            for snapshot in per_slot {
                graphs.push(
                    CudaGraph::scope(&self.device.stream, |scope| {
                        for (id, tensors) in snapshot {
                            for (output, input) in lane_states
                                .get_mut(slot)
                                .and_then(|states| states.get_mut(id))
                                .ok_or_else(|| error("slot verify restore state"))?
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
            }
            restore.push(graphs);
        }
        let hidden = take_result(&mut arena, self.graph.hidden)?;
        let logits = take_result(&mut arena, self.graph.logits)?;
        Ok(SlotVerifyGraph {
            graph,
            restore,
            readbacks: crate::device::Readbacks::default(),
            metadata,
            hidden,
            logits,
            slots,
            verify,
            cursors: vec![0; slots],
            pending: vec![0; slots],
            profile,
            _arena: arena,
        })
    }
}

impl SlotVerifyGraph {
    /// Replay one speculation step. Every slot runs all `verify` lanes: active lanes carry
    /// the submitted token plus its draft candidates, inactive slots mask their lanes with
    /// a `-1` state position. Returns one `(hidden, logits)` row per lane in lane order.
    /// # Errors
    /// Rejects unknown slots, duplicate lanes, out-of-vocabulary tokens, cursors without
    /// room for the whole chain, positions out of capacity and CUDA replay failures.
    pub fn run_verify(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
        capacity: usize,
        vocabulary: usize,
    ) -> Result<BatchOutput> {
        if self.pending.iter().any(|&count| count != 0) {
            return Err(Error::invalid("uncommitted slot verification"));
        }
        let per_lane = verify_lanes(lanes, &self.cursors, self.verify, capacity, vocabulary)?;
        let total = self.slots * self.verify;
        for (lane, metadata) in self.metadata.iter_mut().enumerate() {
            let values = match per_lane[lane] {
                Some((token, rope_pos, state_pos)) => [
                    i32::try_from(rope_pos).map_err(device_error)?,
                    i32::try_from(token).map_err(device_error)?,
                    i32::try_from(state_pos).map_err(device_error)?,
                    0,
                ],
                None => [0, 0, crate::constants::INACTIVE_LANE_STATE_POSITION, 0],
            };
            super::metadata::update(&self.graph, metadata, values)?;
        }
        let hidden_size = self.hidden.size() / total;
        let vocabulary = self.logits.size() / total;
        let mut sources = Vec::with_capacity(2 * total);
        for (lane, entry) in per_lane.iter().enumerate() {
            let active = entry.is_some();
            sources.push(active.then(|| ReadbackSource {
                tensor: Arc::clone(&self.hidden),
                skip: lane * hidden_size,
                len: hidden_size,
            }));
            sources.push(active.then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: lane * vocabulary,
                len: vocabulary,
            }));
        }
        let rows = self.readbacks.run(device, &self.graph, &sources)?;
        if let Some(profile) = &self.profile
            && let Some(&(.., rope_pos, _)) = lanes.first()
        {
            profile.report("slot_verify", lanes.len(), rope_pos);
        }
        for (slot, group) in per_lane.chunks_exact(self.verify).enumerate() {
            self.pending[slot] = group.iter().flatten().count();
        }
        let mut rows = rows.into_iter();
        (0..total)
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot verify hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot verify logits readback"))?,
                ))
            })
            .collect()
    }

    /// Commit `accepted` candidate lanes of `slot` (not counting the submitted token, which
    /// always commits): a partial chain restores the conv/delta snapshot taken after the
    /// last accepted lane; a full chain keeps the live state. The cursor advances past the
    /// submitted token plus the accepted candidates.
    /// # Errors
    /// Rejects unknown slots, out-of-range counts and CUDA restore failures.
    pub fn commit(&mut self, device: &CudaDevice, slot: usize, accepted: usize) -> Result<()> {
        if slot >= self.slots || accepted >= self.pending[slot] {
            return Err(Error::invalid("slot verify commit"));
        }
        if let Some(restore) = self
            .restore
            .get(slot)
            .and_then(|per_slot| per_slot.get(accepted))
            .filter(|_| accepted + 1 < self.verify)
        {
            restore
                .launch()
                .sync_on(&device.stream)
                .map_err(device_error)?;
        }
        self.cursors[slot] += 1 + accepted;
        self.pending[slot] = 0;
        Ok(())
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        &self.cursors
    }

    /// Forget a failed replay when the owning sequence releases its lease.
    pub fn discard(&mut self, slot: usize) {
        if let Some(pending) = self.pending.get_mut(slot) {
            *pending = 0;
        }
    }

    /// Seed a slot's cursor after binding a sequence's covered prefix into it.
    /// # Errors
    /// Rejects unknown slot indices.
    pub fn set_cursor(&mut self, slot: usize, position: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("slot verify cursor"))?;
        *cursor = position;
        self.pending[slot] = 0;
        Ok(())
    }
}

type VerifyMetadata = Option<(u32, usize, usize)>;

/// Validate every active slot as a contiguous prefix before updating any device metadata.
fn verify_lanes(
    lanes: &[SlotVerifyLane],
    cursors: &[usize],
    verify: usize,
    capacity: usize,
    vocabulary: usize,
) -> Result<Vec<VerifyMetadata>> {
    let total = cursors.len() * verify;
    if lanes.is_empty() || lanes.len() > total {
        return Err(Error::invalid("slot verify lane count"));
    }
    let mut per_lane = vec![None; total];
    for &(slot, position, token, rope_pos, state_pos) in lanes {
        if slot >= cursors.len() || position >= verify {
            return Err(Error::invalid("slot verify lane"));
        }
        let lane = slot * verify + position;
        if per_lane[lane].is_some()
            || usize::try_from(token).map_err(device_error)? >= vocabulary
            || state_pos >= capacity
            || state_pos != rope_pos
            || cursors[slot].checked_add(position) != Some(state_pos)
        {
            return Err(Error::invalid("slot verify token or position"));
        }
        per_lane[lane] = Some((token, rope_pos, state_pos));
    }
    for group in per_lane.chunks_exact(verify) {
        let prefix = group.iter().take_while(|entry| entry.is_some()).count();
        if group[prefix..].iter().any(Option::is_some) {
            return Err(Error::invalid("slot verify chain is not a prefix"));
        }
    }
    Ok(per_lane)
}

#[cfg(test)]
mod tests;
