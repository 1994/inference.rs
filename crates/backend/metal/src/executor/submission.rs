use super::{CachedPrefix, MetalBackend, MetalDevice};
use infer_core::{Error, Result, StateId};
use infer_ir::{ExecutionProgram, ExecutionTask, StepPlan};
use infer_state::blocks::BlockLease;

pub(super) struct EncodingUndo {
    pages: Vec<(StateId, Vec<BlockLease>)>,
    pinned_tails: Vec<BlockLease>,
}
pub(super) struct EncodedBatch {
    pub starts: Vec<usize>,
    pub dispatches: usize,
    pub pending_cache: Vec<CachedPrefix>,
}
impl MetalBackend {
    pub(super) fn begin_encoding(&mut self, tasks: &[ExecutionTask]) -> Result<EncodingUndo> {
        let mut pages = Vec::with_capacity(tasks.len());
        for task in tasks {
            let s = &self.sequences[&task.state];
            pages.push((task.state, s.blocks.clone()));
        }
        let pinned_tails = self.kv.pin_shared_tails(tasks.iter().map(|task| {
            let s = &self.sequences[&task.state];
            (s.blocks.as_slice(), s.tokens.len())
        }))?;
        // Protect old shared tails from cache eviction until commit or rollback.
        // Unshared tails stay unpinned so this guard does not introduce extra COW.
        Ok(EncodingUndo {
            pages,
            pinned_tails,
        })
    }
    pub(super) fn rollback_encoding(&mut self, undo: EncodingUndo) -> Result<()> {
        for (id, old) in undo.pages {
            let sequence = self
                .sequences
                .get_mut(&id)
                .ok_or_else(|| Error::invariant("encoding rollback sequence missing"))?;
            self.kv.rollback_append(&mut sequence.blocks, old)?;
            self.update_table(id)?;
        }
        self.kv.release(&undo.pinned_tails)
    }
    pub(super) fn finish_encoding(&mut self, undo: EncodingUndo) {
        // Source pages of queued COW copies remain pinned until GPU completion.
        self.inflight_pins = undo.pinned_tails;
    }
    pub(super) fn release_encoding_pins(&mut self) -> Result<()> {
        self.kv.release(&self.inflight_pins)?;
        self.inflight_pins.clear();
        Ok(())
    }
    pub(super) fn encode_batch(
        &mut self,
        command: &metal::CommandBufferRef,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<EncodedBatch> {
        let mut starts = Vec::with_capacity(tasks.len());
        let mut dispatches = 0;
        let mut pending_cache = vec![];
        let mut pending_bytes = 0u64;
        for task in tasks {
            self.grow_pages(task.state, task.tokens.len(), command)?;
            let s = &self.sequences[&task.state];
            self.encode_pending_prefix(command, s)?;
            let start = s.tokens.len();
            starts.push(start);
            MetalDevice::write_indices_range_idle(
                &s.token_buffer,
                start,
                task.tokens.delta(start)?,
            )?;
            let mut position = start;
            while position < task.tokens.len() {
                let chunk = self.next_chunk(position, task.tokens.len());
                let end = chunk.end;
                dispatches += self.encode_chunk(command, program, step, task, chunk)?;
                let s = &self.sequences[&task.state];
                if end.is_multiple_of(self.config.block_size)
                    && pending_bytes < self.config.prefix_cache_bytes
                    && pending_cache.len() < crate::constants::MAX_PREFIX_ENTRIES
                    && let Some(cached) = self.cache_boundary(
                        command,
                        s,
                        &task.tokens.prefix(&s.tokens, end)?,
                        self.config.prefix_cache_bytes - pending_bytes,
                    )?
                {
                    pending_bytes += cached.bytes;
                    pending_cache.push(cached);
                }
                position = end;
            }
        }
        Ok(EncodedBatch {
            starts,
            dispatches,
            pending_cache,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalConfig, MetalKernels};
    use infer_core::*;
    use infer_ir::*;
    use infer_kernel_api::KernelRegistry;
    use infer_models::ModelPackage;
    use infer_spi::BackendProvider;
    use std::{
        path::Path,
        time::{Duration, Instant},
    };

    fn step(program: &ExecutionProgram, states: &[StateId], count: usize) -> Result<StepPlan> {
        Ok(StepPlan {
            id: StepId::ONE,
            decision: DecisionId::ONE,
            program: program.id,
            role: ExecutionRole::Prefill,
            work: states
                .iter()
                .map(|state| {
                    Ok(PlannedWork {
                        request: RequestId::new(state.get())?,
                        state: *state,
                        token_count: count,
                        role: ExecutionRole::Prefill,
                    })
                })
                .collect::<Result<_>>()?,
            cost: CostEstimate::default(),
            graph: None,
            quantum_overrun: false,
        })
    }
    fn compile_fixture(backend: &MetalBackend) -> Result<ExecutionProgram> {
        let mut kernels = KernelRegistry::default();
        kernels.register(&MetalKernels)?;
        infer_compiler::compile(
            ProgramId::ONE,
            infer_compiler::lower(
                backend.model(),
                backend.execution_graph(backend.model())?,
                PrecisionPlan::f32(),
            )?,
            &kernels,
            &backend.capabilities(),
            1 << 20,
        )
    }

    #[test]
    fn encoding_failure_rolls_back_a_batch_and_shared_cow_tails() -> Result<()> {
        if !MetalBackend::available() {
            return Ok(());
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
        let mut package = ModelPackage::open(root, ModelId::ONE)?;
        let mut backend = MetalBackend::from_package(
            &mut package,
            MetalConfig {
                prefix_cache_bytes: 0,
                block_size: 2,
                probe_bytes: 1 << 20,
                ..Default::default()
            },
        )?;
        let program = compile_fixture(&backend)?;
        backend.reserve_state(StateId::ONE, 8)?;
        let mut ticket = backend.submit(
            &program,
            &step(&program, &[StateId::ONE], 3)?,
            vec![ExecutionTask {
                request: RequestId::ONE,
                state: StateId::ONE,
                tokens: vec![1, 2, 3].into(),

                sampling: None,
            }],
        )?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while backend.poll(&mut ticket)?.is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_micros(100));
        }
        let second = StateId::new(2)?;
        backend.fork_sequence(StateId::ONE, second, 8)?;
        backend.gpu.synchronize();
        let before = backend.capture_execution_state()?;
        let free = backend.kv.free_blocks();
        let original = backend
            .sequences
            .get_mut(&second)
            .ok_or_else(|| Error::invariant("fixture state"))?
            .probes
            .replace(backend.gpu.zeros(1)?);
        let tasks = vec![
            ExecutionTask {
                request: RequestId::ONE,
                state: StateId::ONE,
                tokens: vec![1, 2, 3, 5].into(),

                sampling: None,
            },
            ExecutionTask {
                request: RequestId::new(2)?,
                state: second,
                tokens: vec![1, 2, 3, 8].into(),

                sampling: None,
            },
        ];
        let error = backend
            .submit(
                &program,
                &step(&program, &[StateId::ONE, second], 1)?,
                tasks,
            )
            .err()
            .ok_or_else(|| Error::invariant("encoding failure expected"))?;
        assert_eq!(error.code, ErrorCode::InvalidInput);
        assert_eq!(backend.capture_execution_state()?, before);
        assert_eq!(backend.kv.free_blocks(), free);
        backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 3)])?;
        backend
            .sequences
            .get_mut(&second)
            .ok_or_else(|| Error::invariant("fixture state"))?
            .probes = original;
        let mut ticket = backend.submit(
            &program,
            &step(&program, &[second], 1)?,
            vec![ExecutionTask {
                request: RequestId::new(2)?,
                state: second,
                tokens: vec![1, 2, 3, 8].into(),

                sampling: None,
            }],
        )?;
        assert_eq!(backend.inflight_pins.len(), 1);
        assert_eq!(backend.kv.references(backend.inflight_pins[0])?, 2);
        backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 4)])?;
        while backend.poll(&mut ticket)?.is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_micros(100));
        }
        assert_eq!(backend.inflight_pins, []);
        backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 4)])?;
        Ok(())
    }
}
