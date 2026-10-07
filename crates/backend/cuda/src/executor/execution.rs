use super::{CudaBackend, CudaTicket, ModelOutput, Sequence, SlotLease, Speculation, prefix};
use crate::resident::slot_batch::SlotPool;
use infer_core::{Error, Result, StateId};
use infer_ir::{
    ExecutionInput, ExecutionProgram, ExecutionTask, OutputReadout, StepPlan, TaskOutput,
};
use std::collections::BTreeMap;
use std::sync::Arc;
/// Token granularity at which a prefill state becomes a reusable prompt prefix.
const PREFIX_GRANULARITY: usize = 256;

/// Snapshot the KV rows a sequence has covered, when they form a reusable prompt prefix.
///
/// Returns `None` unless the sequence is a logits-readout generation that just consumed prefill
/// tokens at a cacheable granularity. Capture head-major KV rows and complete recurrent
/// and draft state so restoration resumes the same inference trajectory.
fn cacheable_prefix(
    device: &crate::device::CudaDevice,
    loaded: &crate::loading::LoadedModel,
    state: &Sequence,
    task: &ExecutionTask,
) -> Option<prefix::CachedPrefix> {
    if state.poisoned
        || state.readout != OutputReadout::Logits
        || !matches!(
            task.tokens,
            ExecutionInput::Full(_) | ExecutionInput::Prefill { .. }
        )
    {
        return None;
    }
    let covered = state.history.len();
    if covered == 0 || covered > state.capacity || !covered.is_multiple_of(PREFIX_GRANULARITY) {
        return None;
    }
    let target = super::prefix_state::ProgramSnapshot::capture(
        device,
        &state.program,
        loaded.graph(),
        covered,
    )
    .ok()?;
    let draft = if let Some(spec) = &state.speculation {
        Some(super::prefix_state::DraftSnapshot {
            program: super::prefix_state::ProgramSnapshot::capture(
                device,
                &spec.program,
                loaded.draft_graph()?,
                covered.saturating_sub(crate::constants::MTP_KV_OFFSET),
            )
            .ok()?,
            hidden: spec.last_hidden.clone(),
        })
    } else {
        None
    };
    let bytes = target
        .bytes
        .saturating_add(draft.as_ref().map_or(0, |d| d.program.bytes));
    Some(prefix::CachedPrefix {
        tokens: state.history.clone(),
        target,
        draft,
        bytes,
    })
}

impl CudaBackend {
    pub(super) fn execute(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<CudaTicket> {
        self.check_tasks(program, step, tasks)?;
        let profile = super::profiling::StepProfile::start(self, tasks);
        let phases = super::profiling::PhaseScope::install(profile.is_some());
        self.busy = true;
        // Snapshots are collected while a sequence is mutably borrowed and inserted afterwards,
        // so the cache borrow never overlaps the state borrow.
        let mut snapshots: Vec<prefix::CachedPrefix> = Vec::new();
        // From here every failure belongs to the ticket, including partially executed batches.
        let result = self.run_grouped(tasks, &mut snapshots);
        if let Some(profile) = profile {
            let recorded = phases.map(super::profiling::PhaseScope::finish);
            profile.finish(
                self,
                tasks,
                result.as_ref().ok().map(Vec::as_slice),
                recorded.unwrap_or_default(),
            );
        }
        for snapshot in snapshots {
            self.prefix.insert(snapshot);
        }
        if result.is_err() {
            for task in tasks {
                if let Some(state) = self.states.get_mut(&task.state) {
                    state.poisoned = true;
                }
            }
        }
        Ok(CudaTicket {
            owner: Arc::clone(&self.owner),
            drain_required: result.is_err(),
            result: Some(result),
        })
    }

    /// Execute one step's tasks in order, replaying each consecutive run of eligible
    /// decode tasks through the shared slot graph and everything else serially.
    fn run_grouped(
        &mut self,
        tasks: &[ExecutionTask],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<Vec<TaskOutput>> {
        let mut outputs: Vec<Option<TaskOutput>> = tasks.iter().map(|_| None).collect();
        let eligible: Vec<bool> = tasks
            .iter()
            .map(|task| self.decode_eligible(task))
            .collect();
        self.ensure_slot_pool(&eligible);
        let mut index = 0;
        while index < tasks.len() {
            if !eligible[index] {
                self.run_serial(tasks, index, &mut outputs, snapshots)?;
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < tasks.len() && eligible[end] {
                end += 1;
            }
            self.run_segment(tasks, index..end, &mut outputs, snapshots)?;
            index = end;
        }
        outputs
            .into_iter()
            .map(|output| output.ok_or_else(|| Error::invariant("CUDA task output missing")))
            .collect()
    }

    /// A task rides the slot graph when it is a decode at the sequence cursor, the
    /// sequence reads logits only, and it either already leases a
    /// slot or its KV fits the slot capacity. Sampling is irrelevant: the batched path
    /// leaves token decisions to the engine, exactly like a non-speculative serial step.
    fn decode_eligible(&self, task: &ExecutionTask) -> bool {
        let Some(state) = self.states.get(&task.state) else {
            return false;
        };
        if !matches!(task.tokens, ExecutionInput::Decode { position, .. } if position == state.history.len())
            || state.readout != OutputReadout::Logits
        {
            return false;
        }
        state.slot.as_ref().map_or_else(
            || {
                state.capacity
                    <= self
                        .slots
                        .as_ref()
                        .map_or(crate::constants::CB_SLOT_TOKENS, SlotPool::capacity)
            },
            |lease| lease.batched && lease.position == state.history.len(),
        )
    }

    /// Capture the slot pool on the first step that could batch two decodes; any failure
    /// disables batching permanently rather than affecting the request that triggered it.
    /// With a draft loaded the pool captures the pooled speculation graph instead: every
    /// sequence speculates from its first decode step, so a decode-only pool would sit
    /// unused while squeezing admission memory.
    fn ensure_slot_pool(&mut self, eligible: &[bool]) {
        if self.slots.is_some()
            || self.slots_disabled
            || !eligible.windows(2).any(|pair| pair[0] && pair[1])
        {
            return;
        }
        // Size shared KV storage from admitted requests, not the maximum context.
        // Long contexts outside this pool retain their private execution graph.
        let capacity = self
            .states
            .values()
            .filter(|state| {
                state.readout == OutputReadout::Logits
                    && state.capacity <= crate::constants::CB_SLOT_TOKENS
            })
            .map(|state| state.capacity)
            .max()
            .unwrap_or(1)
            .next_power_of_two();
        // Reserve the configured capture bucket, not the transient number of
        // arrivals seen on this tick. Otherwise a two-request first tick fixes
        // the pool at two slots for the server's entire lifetime.
        let width = self
            .maximum_states
            .clamp(2, crate::constants::CB_DECODE_SLOTS);
        for width in (2..=width).rev() {
            loop {
                match self.loaded.slot_pool(width, capacity) {
                    Ok(pool) => {
                        for state in self.states.values_mut() {
                            if state.capacity <= capacity
                                && state.speculation.is_some()
                                && state.readout == OutputReadout::Logits
                            {
                                state.program.discard_verification();
                            }
                        }
                        if self.loaded.device().reclaim_barrier().is_err() {
                            self.slots_disabled = true;
                            return;
                        }
                        self.draft_slots = self.prepare_draft_pool(width, capacity);
                        self.slots = Some(pool);
                        return;
                    }
                    Err(error) => {
                        super::profiling::pool_failure("target", width, capacity, &error);
                        if !self.reclaim_cached_for_pool(&error) {
                            break;
                        }
                    }
                }
            }
        }
        self.slots_disabled = true;
    }

    fn prepare_draft_pool(&mut self, width: usize, capacity: usize) -> Option<SlotPool> {
        loop {
            match self.loaded.draft_slot_pool(width, capacity) {
                Ok(pool) => return pool,
                Err(error) => {
                    super::profiling::pool_failure("draft", width, capacity, &error);
                    if !self.reclaim_cached_for_pool(&error) {
                        return None;
                    }
                }
            }
        }
    }

    /// Idle programs and cached prefixes must not permanently disable active batching.
    /// Drain deferred frees before retrying the constructor's physical-memory admission.
    fn reclaim_cached_for_pool(&mut self, error: &Error) -> bool {
        error.code == infer_core::ErrorCode::Capacity
            && (self.pool.evict() || self.prefix.evict())
            && self.loaded.device().reclaim_barrier().is_ok()
    }

    /// One consecutive run of eligible decode tasks: batched through the slot pool when
    /// one exists, serial otherwise.
    fn run_segment(
        &mut self,
        tasks: &[ExecutionTask],
        segment: std::ops::Range<usize>,
        outputs: &mut [Option<TaskOutput>],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<()> {
        if self.slots.is_none() {
            for index in segment {
                self.run_serial(tasks, index, outputs, snapshots)?;
            }
            return Ok(());
        }
        let batched = self.execute_slot_segment(tasks, segment.clone(), outputs)?;
        for index in segment {
            if !batched.contains(&index) {
                self.run_serial(tasks, index, outputs, snapshots)?;
            }
        }
        Ok(())
    }

    /// Assign slots to one eligible segment, replay it and commit the results.
    /// Returns the batched task indices; unassigned tasks are left for the serial path.
    /// Bound sequences always batch (their state lives in the slot), while unbound ones
    /// only bind when at least two lanes share a plain decode replay. An existing
    /// verification pool can accept one lane without recreating private checkpoints.
    fn execute_slot_segment(
        &mut self,
        tasks: &[ExecutionTask],
        segment: std::ops::Range<usize>,
        outputs: &mut [Option<TaskOutput>],
    ) -> Result<Vec<usize>> {
        let pool = self
            .slots
            .as_mut()
            .ok_or_else(|| Error::invariant("CUDA slot pool"))?;
        let mut participants: Vec<(usize, usize)> = Vec::new();
        let mut unbound: Vec<usize> = Vec::new();
        for index in segment {
            let state = self
                .states
                .get(&tasks[index].state)
                .ok_or_else(|| Error::invariant("validated CUDA state"))?;
            match &state.slot {
                Some(lease) => participants.push((index, lease.slot)),
                None => unbound.push(index),
            }
        }
        let mut newly: Vec<(usize, usize)> = Vec::new();
        for &index in &unbound {
            if participants.len() + newly.len() >= pool.width() {
                break;
            }
            let Some(slot) = pool.take_free() else {
                break;
            };
            newly.push((index, slot));
        }
        let minimum = if pool.verify().is_some() { 1 } else { 2 };
        if participants.is_empty() && newly.len() < minimum {
            for (_, slot) in newly {
                pool.release(slot);
            }
            return Ok(Vec::new());
        }
        participants.extend_from_slice(&newly);
        participants.sort_unstable_by_key(|&(index, _)| index);
        let device = self.loaded.device();
        if let Err(error) = slot_replay(
            (pool, self.draft_slots.as_mut()),
            &mut self.states,
            device,
            tasks,
            &participants,
            &newly,
            outputs,
        ) {
            // A failed replay poisons the participating sequences, never the pool.
            for &(index, slot) in &participants {
                if let Some(state) = self.states.get_mut(&tasks[index].state) {
                    state.poisoned = true;
                    state.slot = None;
                }
                pool.release(slot);
            }
            return Err(error);
        }
        Ok(participants.iter().map(|&(index, _)| index).collect())
    }

    /// Serial execution for one task; a slot-leased sequence has stale private state and
    /// must never reach here.
    fn run_serial(
        &mut self,
        tasks: &[ExecutionTask],
        index: usize,
        outputs: &mut [Option<TaskOutput>],
        snapshots: &mut Vec<prefix::CachedPrefix>,
    ) -> Result<()> {
        let task = &tasks[index];
        let state = self
            .states
            .get_mut(&task.state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        if state.slot.is_some() {
            return Err(Error::invariant(
                "slot-leased CUDA sequence cannot run serially",
            ));
        }
        if state.speculation.is_some()
            && (matches!(task.tokens, ExecutionInput::Decode { .. })
                || state.program.prefill_width() < crate::constants::PREFILL_LANES)
        {
            self.loaded.ensure_verification(&mut state.program)?;
        }
        let output = state.run(task)?;
        if let Some(snapshot) = cacheable_prefix(self.loaded.device(), &self.loaded, state, task) {
            snapshots.push(snapshot);
        }
        outputs[index] = Some(TaskOutput {
            request: task.request,
            output,
        });
        Ok(())
    }
    pub(super) fn complete(&mut self, ticket: &mut CudaTicket) -> Result<Option<Vec<TaskOutput>>> {
        if !Arc::ptr_eq(&self.owner, &ticket.owner) || ticket.result.is_none() || !self.busy {
            return Err(Error::invalid("foreign or consumed CUDA completion"));
        }
        if ticket.drain_required {
            if let Err(error) = self.loaded.device().drain() {
                // Retain all state ownership when the driver cannot establish completion.
                self.fatal = Some(error);
                return Ok(None);
            }
            ticket.drain_required = false;
        }
        self.busy = false;
        ticket
            .result
            .take()
            .ok_or_else(|| Error::invariant("CUDA completion missing"))?
            .map(Some)
    }
}

/// Bind newly assigned sequences into their slots, replay one decode step for every
/// participant and commit each lane's token, logits and cursor in task order.
fn slot_replay(
    pools: (&mut SlotPool, Option<&mut SlotPool>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &crate::device::CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    newly: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
) -> Result<()> {
    let (pool, draft) = pools;
    for &(index, slot) in newly {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        let rows = state.history.len();
        pool.bind(
            slot,
            state.program.states(),
            state.program.fp8_states(),
            rows,
            device,
        )?;
        state.program.discard_verification();
        state.budget = state
            .budget
            .checked_sub(state.program.released_verification_bytes())
            .ok_or_else(|| Error::invariant("private verification admission credit"))?;
        state.slot = Some(SlotLease {
            slot,
            position: rows,
            batched: true,
        });
    }
    if let Some(width) = pool.verify() {
        return slot_speculate(
            (pool, draft),
            states,
            device,
            tasks,
            participants,
            outputs,
            width,
        );
    }
    let mut lanes = Vec::with_capacity(participants.len());
    for &(index, slot) in participants {
        let state = states
            .get(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        let &ExecutionInput::Decode { token, .. } = &tasks[index].tokens else {
            return Err(Error::invariant("slot decode task"));
        };
        let position = state.history.len();
        lanes.push((slot, token, position, position));
    }
    let rows = pool.run_lanes(device, &lanes)?;
    for ((_, logits), &(index, _)) in rows.into_iter().zip(participants) {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("validated CUDA state"))?;
        tasks[index].tokens.commit(&mut state.history)?;
        if let Some(lease) = &mut state.slot {
            lease.position += 1;
        }
        outputs[index] = Some(TaskOutput {
            request: tasks[index].request,
            output: ModelOutput {
                logits,
                hidden: Vec::new(),
                tokens: Vec::new(),
            },
        });
    }
    Ok(())
}

/// Batch compatible drafts, then verify all candidate chains in one target replay.
fn slot_speculate(
    pools: (&mut SlotPool, Option<&mut SlotPool>),
    states: &mut BTreeMap<StateId, Sequence>,
    device: &crate::device::CudaDevice,
    tasks: &[ExecutionTask],
    participants: &[(usize, usize)],
    outputs: &mut [Option<TaskOutput>],
    width: usize,
) -> Result<()> {
    let (pool, mut draft) = pools;
    let batched = match draft.as_deref_mut() {
        Some(draft) => super::profiling::phase("draft_propose", || {
            super::drafting::propose(draft, states, device, tasks, participants, width)
        })?,
        None => None,
    };
    let defer_draft = batched.is_some();
    let candidates = match batched {
        Some(candidates) => candidates,
        None => participants
            .iter()
            .map(|&(index, _)| {
                states
                    .get_mut(&tasks[index].state)
                    .ok_or_else(|| Error::invariant("slot speculation state"))?
                    .slot_candidates(&tasks[index], width)
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let mut lanes = Vec::new();
    for (&(index, slot), chain) in participants.iter().zip(&candidates) {
        let position = states[&tasks[index].state].history.len();
        for (offset, &token) in chain.iter().enumerate() {
            lanes.push((slot, offset, token, position + offset, position + offset));
        }
    }
    let mut rows = super::profiling::phase("verify", || pool.run_verify(device, &lanes))?;
    for (&(index, slot), chain) in participants.iter().zip(candidates) {
        let state = states
            .get_mut(&tasks[index].state)
            .ok_or_else(|| Error::invariant("slot speculation state"))?;
        let output = state.slot_decide(
            &tasks[index],
            &chain,
            &mut rows[slot * width..][..chain.len()],
            defer_draft,
        )?;
        pool.commit_verify(device, slot, output.tokens.len())?;
        if let Some(lease) = &mut state.slot {
            lease.position = state.history.len();
        }
        outputs[index] = Some(TaskOutput {
            request: tasks[index].request,
            output,
        });
    }
    if defer_draft {
        super::drafting::catch_up(
            draft.ok_or_else(|| Error::invariant("batched draft pool"))?,
            states,
            device,
            tasks,
            participants,
            outputs,
            &rows,
            width,
        )?;
    }
    Ok(())
}

/// Greedy decision at one sequence position; the engine output stage uses the same sampler.
struct Sampler<'a> {
    sampling: &'a infer_ir::Sampling,
    request: u64,
    prompt: &'a [u32],
    scratch: &'a mut infer_workloads::SamplingWorkspace,
}
impl Sampler<'_> {
    fn pick(&mut self, logits: &[f32], index: usize, generated: &[u32]) -> Result<u32> {
        infer_workloads::sample_with_history(
            logits,
            self.sampling,
            self.request,
            index,
            infer_workloads::SamplingHistory {
                prompt: self.prompt,
                generated,
            },
            self.scratch,
        )
    }
}
impl Sequence {
    /// Keep the draft cursor aligned even on non-greedy or capacity-limited steps.
    fn slot_candidates(&mut self, task: &ExecutionTask, width: usize) -> Result<Vec<u32>> {
        let ExecutionInput::Decode { position, token } = task.tokens else {
            return Err(Error::invariant("slot speculation decode"));
        };
        let mut chain = vec![token];
        let Some(spec) = self.speculation.as_mut() else {
            return Ok(chain);
        };
        let prompt_len = *spec.prompt_len.get_or_insert(position);
        let count = spec.depth.min(width - 1).min(self.capacity - position - 1);
        if let Some(sampling) = task.sampling.as_ref().filter(|s| s.temperature == 0.0)
            && count > 0
            && position > 0
        {
            let mut generated = self.history[prompt_len..].to_vec();
            generated.push(token);
            let mut sampler = Sampler {
                sampling,
                request: task.request.get(),
                prompt: &self.history[..prompt_len],
                scratch: &mut spec.scratch,
            };
            chain.extend(propose(
                &mut spec.program,
                &mut sampler,
                &mut generated,
                (token, position),
                &spec.last_hidden,
                count,
            )?);
        } else if position > 0 {
            spec.program.step(
                token,
                position,
                position - crate::constants::MTP_KV_OFFSET,
                Some(&spec.last_hidden),
                false,
            )?;
        }
        Ok(chain)
    }

    /// Return only accepted proposals. On rejection the engine samples the replacement
    /// from the last accepted row and submits it next tick, avoiding an extra target pass.
    fn slot_decide(
        &mut self,
        task: &ExecutionTask,
        chain: &[u32],
        rows: &mut [(Vec<f32>, Vec<f32>)],
        defer_draft: bool,
    ) -> Result<ModelOutput> {
        let position = self.history.len();
        task.tokens.commit(&mut self.history)?;
        let mut accepted = Vec::new();
        if let Some(spec) = self.speculation.as_mut() {
            let prompt_len = *spec.prompt_len.get_or_insert(position);
            if let Some(sampling) = task.sampling.as_ref() {
                let mut generated = self.history[prompt_len..].to_vec();
                let mut sampler = Sampler {
                    sampling,
                    request: task.request.get(),
                    prompt: &self.history[..prompt_len],
                    scratch: &mut spec.scratch,
                };
                for (offset, &candidate) in chain[1..].iter().enumerate() {
                    if sampler.pick(&rows[offset].1, position + offset + 1, &generated)?
                        != candidate
                    {
                        break;
                    }
                    accepted.push(candidate);
                    generated.push(candidate);
                    if sampling.is_eos(candidate) {
                        break;
                    }
                }
            }
            if !defer_draft {
                // Even fully accepted proposals used draft hidden rows during drafting.
                // Replace their KV with target-conditioned rows before the next proposal.
                let hidden: Vec<_> = rows[..accepted.len()].iter().map(|r| r.0.clone()).collect();
                spec.replay(&accepted, &hidden, position + 1)?;
            }
            spec.last_hidden.clone_from(&rows[accepted.len()].0);
        }
        self.history.extend_from_slice(&accepted);
        Ok(ModelOutput {
            logits: std::mem::take(&mut rows[accepted.len()].1),
            hidden: Vec::new(),
            tokens: accepted,
        })
    }

    fn run(&mut self, task: &ExecutionTask) -> Result<ModelOutput> {
        let tokens = task.tokens.delta(self.history.len())?;
        let readout = task.tokens.readout();
        if let Some(output) = self.speculate(task, readout)? {
            return Ok(output);
        }
        let speculate = self.speculation.is_some();
        let mut offset = 0;
        let mut logits = Vec::new();
        // Target hidden rows prime the draft; a chunked prefill spans several calls.
        let mut priming = Vec::new();
        while offset < tokens.len() {
            let width = self.program.prefill_width();
            let count = if width >= crate::constants::PREFILL_LANES
                || width > 1 && tokens.len() - offset >= width
            {
                width.min(tokens.len() - offset)
            } else {
                1
            };
            let position = self.history.len() + offset;
            let last = offset + count == tokens.len();
            let read_logits = last && readout != OutputReadout::None;
            let read_hidden = self.readout == OutputReadout::Full || speculate;
            let results = if count > 1 {
                super::profiling::phase("prefill_chunk", || {
                    self.program.prefill_batch_readout(
                        &tokens[offset..offset + count],
                        position,
                        read_logits,
                        read_hidden,
                    )
                })?
            } else {
                vec![super::profiling::phase("prefill_step", || {
                    self.program.step_readout(
                        tokens[offset],
                        position,
                        position,
                        None,
                        read_logits,
                        read_hidden,
                    )
                })?]
            };
            for (lane, (hidden, row_logits)) in results.into_iter().enumerate() {
                if speculate {
                    priming.push((position + lane, tokens[offset + lane], hidden.clone()));
                }
                if self.readout == OutputReadout::Full {
                    self.hidden.push(hidden);
                }
                if !row_logits.is_empty() {
                    logits = row_logits;
                }
            }
            offset += count;
        }
        if !matches!(task.tokens, ExecutionInput::Decode { .. }) {
            self.prime_draft(&priming)?;
        }
        task.tokens.commit(&mut self.history)?;
        Ok(ModelOutput {
            logits,
            hidden: if readout == OutputReadout::Full {
                self.hidden.clone()
            } else {
                vec![]
            },
            tokens: Vec::new(),
        })
    }

    /// Feed a prompt timeline through the draft: its KV mirrors the target from index one onward.
    fn prime_draft(&mut self, rows: &[(usize, u32, Vec<f32>)]) -> Result<()> {
        let Some(mut spec) = self.speculation.take() else {
            return Ok(());
        };
        let mut previous = std::mem::take(&mut spec.last_hidden);
        let mut result = Ok(());
        let mut start = 0;
        while start < rows.len() {
            let position = rows[start].0;
            let chunk = &rows[start..(start + crate::constants::PREFILL_LANES).min(rows.len())];
            // A fused draft batch needs contiguous positions; anything else replays token by
            // token through the ordinary single-step path.
            let contiguous = chunk
                .iter()
                .enumerate()
                .all(|(offset, (index, _, _))| *index == position + offset);
            if contiguous {
                let mut tokens = Vec::with_capacity(chunk.len());
                let mut externals = Vec::new();
                for (offset, (_, token, hidden)) in chunk.iter().enumerate() {
                    tokens.push(*token);
                    // MTP consumes the hidden of the position before the token it embeds; the
                    // prompt's first position has no predecessor and its state write is masked.
                    let source = if offset == 0 {
                        &previous
                    } else {
                        &chunk[offset - 1].2
                    };
                    externals.extend_from_slice(if source.len() == hidden.len() {
                        source
                    } else {
                        hidden
                    });
                }
                if let Err(error) = spec.program.prime_batch(&tokens, position, &externals) {
                    result = Err(error);
                    break;
                }
                previous.clone_from(&chunk[chunk.len() - 1].2);
                start += chunk.len();
                continue;
            }
            let (index, token, hidden) = &rows[start];
            if *index > 0
                && let Err(error) = spec.program.step(
                    *token,
                    *index,
                    *index - crate::constants::MTP_KV_OFFSET,
                    Some(&previous),
                    false,
                )
            {
                result = Err(error);
                break;
            }
            previous.clone_from(hidden);
            start += 1;
        }
        if result.is_ok() {
            spec.last_hidden = previous;
        }
        self.speculation = Some(spec);
        result
    }

    /// Greedy MTP speculation. `None` keeps the ordinary single-step path.
    fn speculate(
        &mut self,
        task: &ExecutionTask,
        readout: OutputReadout,
    ) -> Result<Option<ModelOutput>> {
        let Some(sampling) = task.sampling.as_ref() else {
            return Ok(None);
        };
        let ExecutionInput::Decode { position, .. } = task.tokens else {
            return Ok(None);
        };
        let width = self.program.batch_width();
        if self.speculation.is_none()
            || readout != OutputReadout::Logits
            // Rejection under a non-zero temperature needs residual sampling the engine sampler
            // cannot reproduce from plain target logits, so speculation stays greedy-only.
            || sampling.temperature != 0.0
            || width < 2
            || position == 0
            || position != self.history.len()
            || position.saturating_add(width) > self.capacity
        {
            return Ok(None);
        }
        // Use the same accept/rollback contract as pooled verification. On rejection
        // return the target logits at the accepted prefix; the output stage samples
        // the replacement and submits it on the next step. An extra scalar target
        // forward here repeats the rejected work and defeats weight sharing.
        let chain = self.slot_candidates(task, width)?;
        let mut rows = self.program.step_batch(&chain, position)?;
        let output = self.slot_decide(task, &chain, &mut rows, false)?;
        self.program.commit_batch(output.tokens.len() + 1)?;
        Ok(Some(output))
    }
}

impl Speculation {
    /// Rebuild the draft timeline after speculation: rewind, then replay the accepted prefix.
    fn replay(&mut self, decided: &[u32], hidden_of: &[Vec<f32>], start: usize) -> Result<()> {
        self.program.rewind_attention(start - 1)?;
        for (offset, token) in decided.iter().enumerate() {
            let index = start + offset;
            self.program.step(
                *token,
                index,
                index - crate::constants::MTP_KV_OFFSET,
                Some(&hidden_of[offset]),
                false,
            )?;
        }
        Ok(())
    }
}
/// Draft proposals for the positions after the submitted token. The first step consumes that
/// token itself, so it doubles as the draft's catch-up and proposes `position + 1`.
fn propose(
    draft: &mut super::super::resident::DeviceProgram,
    sampler: &mut Sampler<'_>,
    proposed: &mut Vec<u32>,
    submitted: (u32, usize),
    previous: &[f32],
    count: usize,
) -> Result<Vec<u32>> {
    let mut proposals = Vec::new();
    let (mut token, start) = submitted;
    let mut hidden = previous.to_vec();
    for index in (start..).take(count) {
        let (next_hidden, logits) = draft.step(
            token,
            index,
            index - crate::constants::MTP_KV_OFFSET,
            Some(&hidden),
            true,
        )?;
        let candidate = sampler.pick(&logits, index + 1, proposed)?;
        proposed.push(candidate);
        proposals.push(candidate);
        token = candidate;
        hidden = next_hidden;
    }
    Ok(proposals)
}
