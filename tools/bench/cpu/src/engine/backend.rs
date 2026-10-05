//! A persistent device-contract double isolates Engine CPU allocations from native driver allocations.
use infer_core::{Error, Result, StateId, map::BoundedMap};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, ModelOutput, OutputReadout,
    StepPlan, TaskOutput,
};
use infer_spi::BackendProvider;
struct State {
    output: Option<ModelOutput>,
}
pub struct Backend {
    states: BoundedMap<StateId, State>,
    outputs: Vec<ModelOutput>,
    batches: Vec<Vec<TaskOutput>>,
    resources: infer_spi::ResourcePool,
    vocabulary: usize,
}
pub struct Ticket(Option<Vec<TaskOutput>>);
impl Backend {
    pub fn new(requests: usize, batch: usize, vocabulary: usize) -> Result<Self> {
        Ok(Self {
            states: BoundedMap::new(requests)?,
            outputs: (0..requests)
                .map(|_| ModelOutput {
                    logits: vec![1.0; vocabulary],
                    hidden: Vec::new(),
                })
                .collect(),
            batches: (0..4).map(|_| Vec::with_capacity(batch)).collect(),
            resources: infer_spi::ResourcePool::new(requests + 4)?,
            vocabulary,
        })
    }
}
impl BackendProvider for Backend {
    type Ticket = Ticket;
    fn identity(&self) -> &'static str {
        "persistent-device-contract-double"
    }
    fn capabilities(&self) -> DeviceCapabilities {
        infer_ir::testing::reference_capabilities()
    }
    fn validate_program(&self, _: &ModelIr, _: &ExecutionProgram) -> Result<()> {
        Ok(())
    }
    fn supports_recompute_preemption(&self) -> bool {
        true
    }
    fn completion_timing(&self, _: &Ticket) -> Option<infer_ir::ExecutionTiming> {
        Some(infer_ir::ExecutionTiming {
            elapsed_us: 1,
            source: infer_ir::TimingSource::CpuWall,
        })
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let _ = reply.send(infer_spi::execute_resource(self, command));
        Ok(ticket)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        _: usize,
        _readout: OutputReadout,
    ) -> Result<()> {
        let output = self
            .outputs
            .pop()
            .ok_or_else(|| Error::invariant("readback credit missing"))?;
        self.states.insert(
            state,
            State {
                output: Some(output),
            },
        )?;
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        let state = self
            .states
            .remove(&state)
            .ok_or_else(|| Error::invariant("release state missing"))?;
        if let Some(output) = state.output {
            self.outputs.push(output);
        }
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Ticket> {
        self.submit_borrowed(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        _: &ExecutionProgram,
        _step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Ticket> {
        let mut batch = self
            .batches
            .pop()
            .ok_or_else(|| Error::invariant("batch credit missing"))?;
        for task in tasks {
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("submit state missing"))?;
            let mut output = state
                .output
                .take()
                .ok_or_else(|| Error::invariant("readback still leased"))?;
            let readout = task.tokens.readout();
            output.logits.resize(
                if readout == OutputReadout::None {
                    0
                } else {
                    self.vocabulary
                },
                1.0,
            );
            batch.push(TaskOutput {
                request: task.request,
                output,
            });
        }
        Ok(Ticket(Some(batch)))
    }
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.states
            .get_mut(&state)
            .ok_or_else(|| Error::invariant("recycle owner missing"))?
            .output = Some(output);
        Ok(())
    }
    fn recycle_batch(&mut self, mut outputs: Vec<TaskOutput>) -> Result<()> {
        outputs.clear();
        self.batches.push(outputs);
        Ok(())
    }
    fn poll(&mut self, ticket: &mut Ticket) -> Result<Option<Vec<TaskOutput>>> {
        Ok(ticket.0.take())
    }
}
