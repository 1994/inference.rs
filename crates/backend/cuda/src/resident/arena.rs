use crate::device::{CudaDevice, device_error};
use cuda_core::sys::CUdeviceptr;
use cutile::{cuda_async::device_buffer::DeviceAllocation, prelude::*};
use infer_core::{Error, Result, TensorId};
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Slot layout of one graph: `last_node` and element count per reused slot.
type ArenaLayout = (Vec<(usize, usize)>, BTreeMap<TensorId, usize>);

/// Reuses equal-sized device buffers after their last consumer.
///
/// Graph results stay live until readback; cuTile mutable ownership remains exclusive.
/// Batched arenas flatten lanes into one axis: every slot is `[lanes * size]`.
pub struct ActivationArena {
    indices: BTreeMap<TensorId, usize>,
    pub(crate) buffers: Vec<Option<Tensor<f32>>>,
    lanes: usize,
    elements: usize,
}

impl ActivationArena {
    /// # Errors
    /// Rejects invalid graphs, overflowing budgets and failed device allocation.
    pub fn new(device: &CudaDevice, graph: &DataflowGraph, byte_budget: usize) -> Result<Self> {
        Self::plan(device, graph, 1, byte_budget)
    }

    /// One slot per plan entry holding `lanes` concatenated rows: activations of
    /// a whole prompt batch, laid out so row `lane` starts at `lane * size`.
    /// # Errors
    /// Rejects zero lanes, invalid graphs, overflowing budgets and failed device allocation.
    pub fn new_batched(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::invalid("batched activation lanes"));
        }
        Self::plan(device, graph, lanes, byte_budget)
    }

    /// Slot layout after last-consumer reuse, with the tensor-to-slot mapping.
    fn layout(graph: &DataflowGraph) -> Result<ArenaLayout> {
        let mut planned = graph.clone();
        planned.plan_lifetimes()?;
        let mut slots: Vec<(usize, usize)> = Vec::new();
        let mut indices = BTreeMap::new();
        for lifetime in &planned.lifetimes {
            let slot = slots
                .iter()
                .position(|(last, size)| *last < lifetime.first_node && *size == lifetime.elements)
                .unwrap_or_else(|| {
                    slots.push((0, lifetime.elements));
                    slots.len() - 1
                });
            slots[slot].0 = lifetime.last_node;
            indices.insert(lifetime.tensor, slot);
        }
        Ok((slots, indices))
    }

    /// Device bytes this graph's activations occupy at `lanes` rows, without allocating.
    /// # Errors
    /// Rejects invalid graphs or overflowing sizes.
    pub fn required_bytes(graph: &DataflowGraph, lanes: usize) -> Result<usize> {
        let (slots, _) = Self::layout(graph)?;
        slots
            .iter()
            .try_fold(0usize, |sum, (_, size)| sum.checked_add(*size))
            .and_then(|sum| sum.checked_mul(lanes))
            .and_then(|elements| elements.checked_mul(crate::constants::F32_BYTES))
            .ok_or_else(|| Error::invalid("activation size overflow"))
    }

    fn plan(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
    ) -> Result<Self> {
        let (slots, indices) = Self::layout(graph)?;
        let elements = slots
            .iter()
            .try_fold(0usize, |sum, (_, size)| sum.checked_add(*size))
            .and_then(|sum| sum.checked_mul(lanes))
            .ok_or_else(|| Error::invalid("activation size overflow"))?;
        if elements
            .checked_mul(crate::constants::F32_BYTES)
            .is_none_or(|bytes| bytes > byte_budget)
        {
            return Err(Error::invalid("device activation budget exceeded"));
        }
        let mut buffers = Vec::with_capacity(slots.len());
        for (_, size) in slots {
            buffers.push(Some(
                api::zeros::<f32>(&[lanes * size])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ));
        }
        Ok(Self {
            indices,
            buffers,
            lanes,
            elements,
        })
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.elements * crate::constants::F32_BYTES
    }

    #[must_use]
    pub(crate) const fn lanes(&self) -> usize {
        self.lanes
    }

    /// Resolve once during graph preparation, not during token execution.
    /// # Errors
    /// Rejects tensors which are not graph activations.
    pub fn slot(&self, id: TensorId) -> Result<usize> {
        self.indices
            .get(&id)
            .copied()
            .ok_or_else(|| Error::invalid("unknown device activation"))
    }

    pub(crate) fn get(&self, id: TensorId) -> Result<&Tensor<f32>> {
        self.buffers[self.slot(id)?]
            .as_ref()
            .ok_or_else(|| Error::invariant("activation is currently an output"))
    }

    /// Flat view of one lane inside a batched slot.
    /// # Errors
    /// Rejects non-batched arenas and out-of-range lanes.
    pub(crate) fn row(&self, id: TensorId, lane: usize) -> Result<TensorView<'_, f32>> {
        let tensor = self.get(id)?;
        if self.lanes == 1 || !tensor.size().is_multiple_of(self.lanes) {
            return Err(Error::invalid("activation slot is not batched"));
        }
        let size = tensor.size() / self.lanes;
        if lane >= self.lanes {
            return Err(Error::invalid("activation lane out of bounds"));
        }
        let range = lane * size..(lane + 1) * size;
        tensor
            .slice(std::slice::from_ref(&range))
            .map_err(device_error)
    }
}

/// `DeviceAllocation` view of one row inside a batched slot. Carries no owner:
/// the slot itself is kept alive by the arena stored in `BatchGraph`, which
/// outlives every captured graph that references these rows.
struct RowAllocation {
    pointer: CUdeviceptr,
    bytes: usize,
    device: usize,
}

#[expect(
    unsafe_code,
    reason = "Audited row aliasing: the batched arena outlives its captured graphs; \
              row extents are checked against the slot at construction; kernels on one \
              stream order every row access"
)]
// SAFETY: getters are stable for the token's lifetime; `pointer`/`bytes` describe a
// row range verified against the slot's own extent at construction in `row_tensor`.
unsafe impl DeviceAllocation for RowAllocation {
    fn device_ptr(&self) -> CUdeviceptr {
        self.pointer
    }

    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn device_id(&self) -> usize {
        self.device
    }
}

/// Owned tensor covering row `lane` of a taken batched slot; dropping it frees
/// nothing. The returned Arc restores the slot once every row tensor is dropped.
/// # Errors
/// Rejects malformed slots, out-of-range lanes and offset overflows.
#[expect(
    unsafe_code,
    reason = "Audited from_foreign row view: the caller re-anchors the slot into the \
              arena before any launch and the arena outlives its graphs; construction \
              bounds the row inside the slot; same-stream ordering serializes accesses"
)]
pub fn row_tensor(
    slot: Tensor<f32>,
    lanes: usize,
    lane: usize,
) -> Result<(Tensor<f32>, Arc<Tensor<f32>>)> {
    if lanes == 0 || !slot.size().is_multiple_of(lanes) {
        return Err(Error::invalid("activation slot is not batched"));
    }
    let size = slot.size() / lanes;
    if lane >= lanes {
        return Err(Error::invalid("activation lane out of bounds"));
    }
    let offset = lane
        .checked_mul(size)
        .and_then(|row| row.checked_mul(crate::constants::F32_BYTES))
        .ok_or_else(|| Error::invalid("activation row offset overflow"))?;
    let shared = Arc::new(slot);
    let allocation = RowAllocation {
        pointer: shared.device_pointer().cu_deviceptr()
            + u64::try_from(offset).map_err(device_error)?,
        bytes: shared.num_bytes() - offset,
        device: shared.device_id(),
    };
    // SAFETY: the slot outlives every graph replay through the arena anchor; the row
    // lies fully inside it by construction; same-stream replay serializes row access.
    let row = unsafe {
        Tensor::from_foreign(
            Arc::new(allocation),
            vec![i32::try_from(size).map_err(device_error)?],
            vec![1],
        )
    };
    Ok((row, shared))
}
