//! Capture-local common subexpression elimination for adjacent `SwiGLU` inputs.
use super::super::{ProgramWeights, program::ActivationQuantization};
use crate::mlp::ProjectionWeight;
use infer_core::TensorId;
use infer_ir::{DataflowGraph, TensorOp};

const SWIGLU_NODE_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Fp4(u32),
    Fp8Token,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    first: usize,
    second: usize,
    input: TensorId,
    columns: usize,
    kind: QuantKind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantPairRole {
    First(Pair),
    Second(Pair),
}
impl QuantPairRole {
    const fn pair(self) -> Pair {
        match self {
            Self::First(pair) | Self::Second(pair) => pair,
        }
    }
    const fn index(self) -> usize {
        match self {
            Self::First(pair) => pair.first,
            Self::Second(pair) => pair.second,
        }
    }
}
pub struct QuantPairPlan {
    roles: Vec<Option<QuantPairRole>>,
}
impl QuantPairPlan {
    pub fn new(graph: &DataflowGraph, weights: &ProgramWeights) -> Self {
        Self::from_contracts(
            graph,
            |id| weights.constants.contains_key(&id),
            |id| match (
                weights.projections.get(&id),
                weights.activation_quantization(id),
            ) {
                (Some(ProjectionWeight::Fp4(..)), Some(ActivationQuantization::Fp4(scale)))
                    if scale.is_finite() && scale > 0.0 =>
                {
                    Some(QuantKind::Fp4(scale.to_bits()))
                }
                (Some(ProjectionWeight::Fp8(..)), Some(ActivationQuantization::Fp8Token)) => {
                    Some(QuantKind::Fp8Token)
                }
                _ => None,
            },
        )
    }
    fn from_contracts(
        graph: &DataflowGraph,
        constant: impl Fn(TensorId) -> bool,
        contract: impl Fn(TensorId) -> Option<QuantKind>,
    ) -> Self {
        let mut roles = vec![None; graph.nodes.len()];
        for (index, nodes) in graph.nodes.windows(SWIGLU_NODE_COUNT).enumerate() {
            let [gate, up, active, product] = nodes else {
                continue;
            };
            if gate.op != TensorOp::Linear
                || up.op != TensorOp::Linear
                || active.op != TensorOp::Silu
                || product.op != TensorOp::Multiply
                || gate.inputs.len() != 2
                || up.inputs.len() != 2
                || gate.outputs.len() != 1
                || up.outputs.len() != 1
                || active.outputs.len() != 1
                || product.outputs.len() != 1
                || !gate.states.is_empty()
                || !up.states.is_empty()
                || gate.inputs[0] != up.inputs[0]
                || constant(gate.inputs[0])
                || active.inputs != gate.outputs
                || product.inputs != [active.outputs[0], up.outputs[0]]
                || gate.outputs[0] == gate.inputs[0]
                || up.outputs[0] == up.inputs[0]
                || gate.outputs[0] == up.outputs[0]
            {
                continue;
            }
            let Some(kind) = contract(gate.inputs[1]) else {
                continue;
            };
            if contract(up.inputs[1]) != Some(kind) {
                continue;
            }
            let elements = |id| {
                graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == id)
                    .and_then(|tensor| tensor.elements().ok())
            };
            let Some(columns) = elements(gate.inputs[0]) else {
                continue;
            };
            let Some(outputs) = elements(gate.outputs[0]) else {
                continue;
            };
            if columns == 0
                || outputs == 0
                || elements(up.outputs[0]) != Some(outputs)
                || elements(active.outputs[0]) != Some(outputs)
                || elements(product.outputs[0]) != Some(outputs)
            {
                continue;
            }
            let pair = Pair {
                first: index,
                second: index + 1,
                input: gate.inputs[0],
                columns,
                kind,
            };
            roles[index] = Some(QuantPairRole::First(pair));
            roles[index + 1] = Some(QuantPairRole::Second(pair));
        }
        Self { roles }
    }
    pub fn role(&self, index: usize) -> Option<QuantPairRole> {
        self.roles.get(index).copied().flatten()
    }
}

/// Every field describes the actual operand or the existing quantization buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantKey {
    pub input_pointer: u64,
    pub rows: usize,
    pub columns: usize,
    pub shape: [i32; 2],
    pub strides: [i32; 2],
    pub kind: QuantKind,
    pub codes_pointer: u64,
    pub scales_pointer: u64,
}
#[derive(Clone, Copy)]
struct Primed {
    epoch: u64,
    pair: Pair,
    key: QuantKey,
}
#[derive(Clone, Copy)]
pub struct QuantTicket {
    pair: Pair,
    key: QuantKey,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantReuseStats {
    pub recorded: usize,
    pub reused: usize,
}
pub struct CaptureQuantReuse {
    epoch: u64,
    active: bool,
    enabled: bool,
    context: Option<QuantPairRole>,
    primed: Option<Primed>,
    #[cfg(test)]
    stats: QuantReuseStats,
    #[cfg(test)]
    total: QuantReuseStats,
}
impl Default for CaptureQuantReuse {
    fn default() -> Self {
        Self {
            epoch: 0,
            active: false,
            enabled: enabled(),
            context: None,
            primed: None,
            #[cfg(test)]
            stats: QuantReuseStats::default(),
            #[cfg(test)]
            total: QuantReuseStats::default(),
        }
    }
}
impl CaptureQuantReuse {
    pub const fn begin(&mut self) {
        self.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.active = true;
        #[cfg(test)]
        {
            self.stats = QuantReuseStats {
                recorded: 0,
                reused: 0,
            };
        }
    }
    pub const fn end(&mut self) {
        self.clear();
        self.active = false;
    }
    pub const fn clear(&mut self) {
        self.context = None;
        self.primed = None;
    }
    pub fn enter(&mut self, index: usize, role: Option<QuantPairRole>) {
        // A permit that was not consumed cannot certify the following node.
        if self.context.take().is_some() {
            self.primed = None;
        }
        let Some(role) = role.filter(|role| self.enabled && self.active && role.index() == index)
        else {
            self.clear();
            return;
        };
        match role {
            QuantPairRole::First(_) => self.primed = None,
            QuantPairRole::Second(pair) => {
                if !self
                    .primed
                    .is_some_and(|primed| primed.epoch == self.epoch && primed.pair == pair)
                {
                    self.primed = None;
                }
            }
        }
        self.context = Some(role);
    }
    /// Consumes one permit. Missing or incompatible permits retain the original quantizer.
    pub fn prepare(&mut self, key: Option<QuantKey>) -> (bool, Option<QuantTicket>) {
        let role = self.context.take();
        let Some((role, key)) = role.zip(key) else {
            self.clear();
            #[cfg(test)]
            self.recorded();
            return (false, None);
        };
        let pair = role.pair();
        if key.columns != pair.columns || key.kind != pair.kind {
            self.clear();
            #[cfg(test)]
            self.recorded();
            return (false, None);
        }
        match role {
            QuantPairRole::First(_) => {
                self.primed = None;
                #[cfg(test)]
                self.recorded();
                (false, Some(QuantTicket { pair, key }))
            }
            QuantPairRole::Second(_) => {
                let reused = self.primed.take().is_some_and(|primed| {
                    primed.epoch == self.epoch && primed.pair == pair && primed.key == key
                });
                if reused {
                    #[cfg(test)]
                    {
                        self.stats.reused += 1;
                        self.total.reused += 1;
                    }
                } else {
                    #[cfg(test)]
                    self.recorded();
                }
                (reused, None)
            }
        }
    }
    pub const fn finish(&mut self, ticket: Option<QuantTicket>, success: bool) {
        if !success {
            self.clear();
        } else if let Some(ticket) = ticket {
            self.primed = Some(Primed {
                epoch: self.epoch,
                pair: ticket.pair,
                key: ticket.key,
            });
        }
    }
    #[cfg(test)]
    const fn recorded(&mut self) {
        #[cfg(test)]
        {
            self.stats.recorded += 1;
            self.total.recorded += 1;
        }
    }
    #[cfg(test)]
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.clear();
        self.enabled = enabled;
    }
    #[cfg(test)]
    pub const fn stats(&self) -> QuantReuseStats {
        self.stats
    }
    #[cfg(test)]
    pub const fn total(&self) -> QuantReuseStats {
        self.total
    }
}

#[cfg(not(test))]
const fn enabled() -> bool {
    true
}
#[cfg(test)]
thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
#[cfg(test)]
fn enabled() -> bool {
    ENABLED.with(std::cell::Cell::get)
}
/// Applies only to workspaces constructed on this thread; unwinding restores the preference.
#[cfg(test)]
pub fn with_quant_reuse_disabled<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let restore = Restore(ENABLED.with(|enabled| enabled.replace(false)));
    let result = run();
    drop(restore);
    result
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_quant_reuse_plan.rs"]
mod tests;
