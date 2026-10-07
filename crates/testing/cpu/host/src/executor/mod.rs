//! Executable tensor dataflow and incremental physical hybrid state on the host.
mod checkpoint;
mod execution;
mod inspection;
mod kernels;
pub use kernels::execute as reference_operation;
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

/// One mebibyte in bytes, for readable budget expressions.
const MIB: u64 = 1024 * 1024;
/// Default host memory budget in bytes (512 MiB).
const DEFAULT_MEMORY_BYTES: u64 = 512 * MIB;
/// Default number of tokens stored in one paged row block.
const DEFAULT_PAGE_TOKENS: usize = 16;
/// Default number of operation traces retained before the oldest is dropped.
const DEFAULT_TRACE_CAPACITY: usize = 8192;
/// Default prefix-cache budget in bytes (16 MiB).
const DEFAULT_PREFIX_CACHE_BYTES: u64 = 16 * MIB;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostConfig {
    pub memory_bytes: u64,
    pub block_size: usize,
    pub trace_capacity: usize,
    pub prefix_cache_bytes: u64,
    pub probe_bytes: u64,
}
impl Default for HostConfig {
    fn default() -> Self {
        Self {
            memory_bytes: DEFAULT_MEMORY_BYTES,
            block_size: DEFAULT_PAGE_TOKENS,
            trace_capacity: DEFAULT_TRACE_CAPACITY,
            prefix_cache_bytes: DEFAULT_PREFIX_CACHE_BYTES,
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
            tokens: Vec::new(),
        }
    }
}
fn publish_prefix(
    prefixes: &mut PrefixCache<Sequence>,
    sequence: &Sequence,
    block_size: usize,
    cache_bytes: u64,
) -> Result<()> {
    if sequence.tokens.len().is_multiple_of(block_size) && cache_bytes > 0 {
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
            .map(|r| r.len() as u64 * crate::constants::F32_BYTES_U64)
            .sum::<u64>()
        + sequence.tokens.len() as u64 * crate::constants::TOKEN_BYTES_U64
        + sequence.logits.len() as u64 * crate::constants::F32_BYTES_U64
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
