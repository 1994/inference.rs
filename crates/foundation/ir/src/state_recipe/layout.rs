use super::{StateExtent, StateLayout, StateMemory, StateRecipe, StateRegionLayout};
use crate::OutputReadout;
use infer_core::{Error, Result};

/// Number of state memory classes whose byte offsets are accounted independently.
const STATE_MEMORY_CLASSES: usize = 3;

impl StateRecipe {
    /// # Errors
    /// Rejects capacities outside the compiled contract or overflowing sizes.
    pub fn layout(&self, capacity: usize, readout: OutputReadout) -> Result<StateLayout> {
        let mut layout = StateLayout {
            capacity,
            readout,
            private_bytes: 0,
            host_bytes: 0,
            kv_block_bytes: 0,
            max_pages: capacity
                .checked_div(self.block_size)
                .map(|_| capacity.div_ceil(self.block_size))
                .ok_or_else(|| Error::invalid("zero page size"))?,
        };
        self.visit(capacity, readout, |region| {
            let sum = match region.region.memory {
                StateMemory::DevicePrivate => &mut layout.private_bytes,
                StateMemory::HostMirror => &mut layout.host_bytes,
                StateMemory::KvBlock => &mut layout.kv_block_bytes,
            };
            *sum = sum
                .checked_add(region.bytes)
                .ok_or_else(|| Error::invalid("state size overflow"))?;
            Ok(())
        })?;
        Ok(layout)
    }
    /// Stream complete region descriptors without creating a per-request recipe or buffer list.
    /// # Errors
    /// Rejects invalid dimensions/overflow, or propagates the visitor's allocation error.
    pub fn visit(
        &self,
        capacity: usize,
        readout: OutputReadout,
        mut visitor: impl FnMut(StateRegionLayout) -> Result<()>,
    ) -> Result<()> {
        if capacity == 0 || capacity > self.max_tokens || self.block_size == 0 {
            return Err(Error::invalid("physical state capacity outside recipe"));
        }
        let mut offsets = [0u64; STATE_MEMORY_CLASSES];
        for region in &self.regions {
            let elements = match region.extent {
                StateExtent::Fixed(n) => Some(n),
                StateExtent::Tokens(width) => capacity.checked_mul(width),
                StateExtent::FullTokens(width) => {
                    if readout == OutputReadout::Full {
                        capacity.checked_mul(width)
                    } else {
                        Some(0)
                    }
                }
                StateExtent::Pages => Some(capacity.div_ceil(self.block_size)),
                StateExtent::Hidden(width) => if readout == OutputReadout::Full {
                    capacity
                } else {
                    1
                }
                .checked_mul(width),
            }
            .filter(|_| region.element_bytes > 0)
            .ok_or_else(|| Error::invalid("physical region shape overflow"))?;
            let allocated = if region.memory == StateMemory::DevicePrivate {
                elements.max(1)
            } else {
                elements
            };
            let bytes = allocated
                .checked_mul(region.element_bytes)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| Error::invalid("physical region bytes overflow"))?;
            let index = match region.memory {
                StateMemory::DevicePrivate => 0,
                StateMemory::HostMirror => 1,
                StateMemory::KvBlock => 2,
            };
            visitor(StateRegionLayout {
                region: *region,
                elements,
                offset: offsets[index],
                bytes,
            })?;
            offsets[index] = offsets[index]
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("physical region offset overflow"))?;
        }
        Ok(())
    }
}
