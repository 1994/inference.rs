//! Contiguous pool state families with the same per-slot views used by homes and fallback graphs.
use super::{batch::States, program::ProgramWeights};
use crate::{
    constants::F32_BYTES,
    device::{CudaDevice, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::DataflowGraph;
use std::ops::{Deref, DerefMut};

/// Views drop before their owning allocations; the pool drops its captures before this object.
pub(super) struct SlotStates {
    lanes: Vec<States>,
    pub(super) roots: States,
}

impl SlotStates {
    pub(super) fn allocate(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize),
        budget: u64,
    ) -> Result<Self> {
        let (width, capacity) = geometry;
        let plan = super::program::state_shapes(graph, capacity, weights)?;
        let mut bytes = 0_u64;
        for (_, shapes) in &plan {
            for shape in shapes {
                let elements = shape
                    .iter()
                    .try_fold(width, |n, &dim| n.checked_mul(dim))
                    .ok_or_else(|| Error::invalid("pooled state overflow"))?;
                bytes = bytes
                    .checked_add(
                        (elements as u64)
                            .checked_mul(F32_BYTES as u64)
                            .ok_or_else(|| Error::invalid("pooled state byte overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid("pooled state budget overflow"))?;
            }
        }
        if bytes > budget {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "pooled state exceeds available budget",
            ));
        }
        let mut result = Self {
            lanes: (0..width).map(|_| States::new()).collect(),
            roots: States::new(),
        };
        for (id, shapes) in plan {
            let mut owners = Vec::new();
            for shape in shapes {
                let size = shape.iter().product::<usize>();
                let mut root = api::zeros::<f32>(&[width * size])
                    .sync_on(&device.stream)
                    .map_err(device_error)?;
                for (slot, lane) in result.lanes.iter_mut().enumerate() {
                    let (row, owner) = super::arena::row_tensor(root, width, slot)?;
                    root = Arc::try_unwrap(owner)
                        .map_err(|_| Error::invariant("unique pooled state owner"))?;
                    lane.entry(id)
                        .or_default()
                        .push(row.reshape(&shape).map_err(device_error)?);
                }
                owners.push(root);
            }
            result.roots.insert(id, owners);
        }
        Ok(result)
    }

    pub(super) fn parts(&mut self) -> (&mut [States], &States) {
        (&mut self.lanes, &self.roots)
    }
}

impl Deref for SlotStates {
    type Target = [States];
    fn deref(&self) -> &Self::Target {
        &self.lanes
    }
}
impl DerefMut for SlotStates {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lanes
    }
}
