//! Bounded host traces and layer probes.
use super::{HostConfig, LayerProbe, OpTrace};
use infer_core::{KernelId, OpId, TensorId};
use infer_ir::{ExecutionTask, StepPlan};
use std::{collections::BTreeMap, collections::VecDeque, time::Instant};

pub(super) struct HostObservations<'a> {
    traces: &'a mut VecDeque<OpTrace>,
    trace_dropped: &'a mut u64,
    probes: &'a mut VecDeque<LayerProbe>,
    probe_dropped: &'a mut u64,
    layer_outputs: &'a BTreeMap<OpId, usize>,
    slots: &'a BTreeMap<TensorId, usize>,
    config: &'a HostConfig,
    hidden_size: usize,
}
impl<'a> HostObservations<'a> {
    pub(super) const fn new(
        traces: &'a mut VecDeque<OpTrace>,
        trace_dropped: &'a mut u64,
        probes: &'a mut VecDeque<LayerProbe>,
        probe_dropped: &'a mut u64,
        metadata: (
            &'a BTreeMap<OpId, usize>,
            &'a BTreeMap<TensorId, usize>,
            &'a HostConfig,
        ),
        hidden_size: usize,
    ) -> Self {
        Self {
            traces,
            trace_dropped,
            probes,
            probe_dropped,
            layer_outputs: metadata.0,
            slots: metadata.1,
            config: metadata.2,
            hidden_size,
        }
    }

    pub(super) fn record(
        &mut self,
        node: &infer_ir::TensorNode,
        kernel: KernelId,
        task: &ExecutionTask,
        dispatch: (&StepPlan, usize),
        timing: (Instant, u64),
        buffers: &[Vec<f32>],
    ) {
        let (step, position) = dispatch;
        let (start, timestamp_ns) = timing;
        if self.traces.len() == self.config.trace_capacity {
            self.traces.pop_front();
            *self.trace_dropped = self.trace_dropped.saturating_add(1);
        }
        self.traces.push_back(OpTrace {
            token_count: 1,
            request: task.request,
            step: step.id,
            state: task.state,
            op: node.id,
            kernel,
            position,
            elapsed_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            timestamp_ns,
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            states: node.states.clone(),
        });
        let sample_bytes = self.hidden_size as u64 * crate::constants::F32_BYTES_U64
            + size_of::<LayerProbe>() as u64;
        let probe_capacity =
            usize::try_from(self.config.probe_bytes / sample_bytes).unwrap_or(usize::MAX);
        if probe_capacity > 0
            && let Some(layer) = self.layer_outputs.get(&node.id)
        {
            if self.probes.len() == probe_capacity {
                self.probes.pop_front();
                *self.probe_dropped = self.probe_dropped.saturating_add(1);
            }
            self.probes.push_back(LayerProbe {
                request: task.request,
                step: step.id,
                state: task.state,
                op: node.id,
                layer: *layer,
                position,
                hidden: buffers[self.slots[&node.outputs[0]]].clone(),
            });
        }
    }
}
