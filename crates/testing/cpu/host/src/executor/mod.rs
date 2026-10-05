//! Executable tensor dataflow and incremental physical hybrid state on the host.
mod checkpoint;
mod execution;
mod inspection;
mod kernels;
mod provider;
mod setup;
mod state;
use infer_core::{Error, OpId, Result, StateId, TensorId};
use infer_ir::{
    DataflowGraph, ExecutionTask, ExecutionTiming, ModelIr, ModelOutput, TaskOutput, TensorSpec,
};
use infer_models::HostTensor;
use infer_state::{cache::PrefixCache, physical::PhysicalTensor};
use observation::HostObservations;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, collections::VecDeque, time::Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostConfig {
    pub memory_bytes: u64,
    pub page_tokens: usize,
    pub trace_capacity: usize,
    pub prefix_cache_bytes: u64,
    pub probe_bytes: u64,
}
impl Default for HostConfig {
    fn default() -> Self {
        Self {
            memory_bytes: 512 * 1024 * 1024,
            page_tokens: 16,
            trace_capacity: 8192,
            prefix_cache_bytes: 16 * 1024 * 1024,
            probe_bytes: 0,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sequence {
    capacity: usize,
    reserved_bytes: u64,
    tokens: Vec<u32>,
    hidden: Vec<Vec<f32>>,
    logits: Vec<f32>,
    tensors: BTreeMap<TensorId, PhysicalTensor>,
}
pub use infer_ir::{LayerProbe, OpTrace};
pub type HostInspection = infer_ir::ExecutionStats;
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    identity: String,
    sequences: BTreeMap<StateId, Sequence>,
    tokens_executed: u64,
    prefix_hits: u64,
}
pub struct HostBackend {
    model: ModelIr,
    graph: DataflowGraph,
    weights: BTreeMap<TensorId, HostTensor>,
    slots: BTreeMap<TensorId, usize>,
    specs: BTreeMap<TensorId, TensorSpec>,
    identity: String,
    config: HostConfig,
    sequences: BTreeMap<StateId, Sequence>,
    tokens_executed: u64,
    traces: VecDeque<OpTrace>,
    trace_dropped: u64,
    prefixes: PrefixCache<Sequence>,
    prefix_hits: u64,
    trace_origin: Instant,
    layer_outputs: BTreeMap<OpId, usize>,
    probes: VecDeque<LayerProbe>,
    probe_dropped: u64,
}
impl Sequence {
    fn readout(&self, readout: infer_ir::OutputReadout) -> ModelOutput {
        ModelOutput {
            logits: if readout == infer_ir::OutputReadout::None {
                vec![]
            } else {
                self.logits.clone()
            },
            hidden: if readout == infer_ir::OutputReadout::Full {
                self.hidden.clone()
            } else {
                vec![]
            },
        }
    }
}
fn publish_prefix(
    prefixes: &mut PrefixCache<Sequence>,
    sequence: &Sequence,
    page_tokens: usize,
    cache_bytes: u64,
) -> Result<()> {
    if sequence.tokens.len().is_multiple_of(page_tokens) && cache_bytes > 0 {
        let bytes = sequence_bytes(sequence);
        if !prefixes.contains(&sequence.tokens) {
            prefixes.insert(sequence.tokens.clone(), sequence.clone(), bytes)?;
        }
    }
    Ok(())
}
pub struct HostTicket {
    result: Option<Vec<TaskOutput>>,
    timing: ExecutionTiming,
}
fn sequence_bytes(sequence: &Sequence) -> u64 {
    sequence
        .tensors
        .values()
        .map(PhysicalTensor::allocated_bytes)
        .sum::<usize>() as u64
        + sequence
            .hidden
            .iter()
            .map(|r| r.len() as u64 * 4)
            .sum::<u64>()
        + sequence.tokens.len() as u64 * 4
        + sequence.logits.len() as u64 * 4
}

fn validate_checkpoint_tensors(sequence: &Sequence, expected: &Sequence) -> Result<()> {
    for (id, tensor) in &sequence.tensors {
        match (tensor, expected.tensors.get(id)) {
            (
                PhysicalTensor::Kv { keys, values },
                Some(PhysicalTensor::Kv {
                    keys: ek,
                    values: ev,
                }),
            ) => {
                keys.validate()?;
                values.validate()?;
                if keys.rows != sequence.tokens.len()
                    || values.rows != keys.rows
                    || keys.width != ek.width
                    || values.width != ev.width
                    || keys.capacity != ek.capacity
                    || values.capacity != ev.capacity
                    || keys.page_rows != ek.page_rows
                    || values.page_rows != ev.page_rows
                {
                    return Err(Error::invalid("corrupt KV checkpoint"));
                }
            }
            (
                PhysicalTensor::Conv {
                    channels,
                    kernel,
                    history,
                },
                Some(PhysicalTensor::Conv {
                    channels: ec,
                    kernel: ek,
                    history: eh,
                }),
            ) if channels == ec
                && kernel == ek
                && history.len() == eh.len()
                && history.iter().all(|v| v.is_finite()) => {}
            (
                PhysicalTensor::Delta {
                    heads,
                    key_dim,
                    value_dim,
                    recurrent,
                },
                Some(PhysicalTensor::Delta {
                    heads: eh,
                    key_dim: ek,
                    value_dim: ev,
                    recurrent: er,
                }),
            ) if heads == eh
                && key_dim == ek
                && value_dim == ev
                && recurrent.len() == er.len()
                && recurrent.iter().all(|v| v.is_finite()) => {}
            _ => return Err(Error::invalid("corrupt hybrid tensor checkpoint")),
        }
    }
    Ok(())
}

impl Sequence {
    fn validate_input(&self, task: &ExecutionTask, vocab_size: usize) -> Result<()> {
        task.tokens
            .validate(&self.tokens, self.capacity, vocab_size)?;
        Ok(())
    }
}
mod observation;
mod registry;
pub use registry::HostKernels;
