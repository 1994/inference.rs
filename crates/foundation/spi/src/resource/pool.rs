//! Persistent acknowledgement cells retain credit until both ticket and responder retire.
use super::{ResourceReply, ResourceResponder, ResourceTicket};
use infer_core::{Error, ErrorCode, Result};
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicU8,
    sync::atomic::AtomicUsize, sync::atomic::Ordering,
};

/// Upper bound on pre-allocated acknowledgement cells; larger startup requests are rejected.
const MAX_POOL_CAPACITY: usize = 1_048_576;

pub(super) struct Cell {
    leased: AtomicBool,
    readers: AtomicUsize,
    pub cancelled: AtomicBool,
    state: AtomicU8,
    value: Mutex<Option<Result<ResourceReply>>>,
}
pub(super) struct Reader {
    pub cell: Arc<Cell>,
}
impl Drop for Reader {
    fn drop(&mut self) {
        if self.cell.readers.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if let Ok(mut value) = self.cell.value.lock() {
            *value = None;
        }
        self.cell.state.store(0, Ordering::Relaxed);
        self.cell.cancelled.store(false, Ordering::Relaxed);
        self.cell.leased.store(false, Ordering::Release);
    }
}
impl Reader {
    pub fn poll(&self) -> Result<Option<ResourceReply>> {
        match self.cell.state.load(Ordering::Acquire) {
            0 => Ok(None),
            1 => self
                .cell
                .value
                .lock()
                .map_err(|_| Error::invariant("resource reply cell poisoned"))?
                .take()
                .ok_or_else(|| Error::invariant("resource reply consumed twice"))?
                .map(Some),
            _ => Err(Error::new(
                ErrorCode::Backend,
                "resource owner disconnected",
            )),
        }
    }
    pub fn cancel(&self) {
        self.cell.cancelled.store(true, Ordering::Release);
        if let Ok(mut value) = self.cell.value.lock() {
            *value = None;
        }
    }
}
pub(super) struct Reply {
    pub reader: Reader,
    published: bool,
}
impl Reply {
    pub fn send(
        mut self,
        result: Result<ResourceReply>,
    ) -> std::result::Result<(), Result<ResourceReply>> {
        let Ok(mut value) = self.reader.cell.value.lock() else {
            return Err(result);
        };
        self.published = true;
        if self.reader.cell.cancelled.load(Ordering::Acquire) {
            return Err(result);
        }
        *value = Some(result);
        drop(value);
        self.reader.cell.state.store(1, Ordering::Release);
        Ok(())
    }
}
impl Drop for Reply {
    fn drop(&mut self) {
        if !self.published {
            self.reader.cell.state.store(2, Ordering::Release);
        }
    }
}
/// Acknowledgements allocate once at startup. Publication and polling only move typed results.
pub struct ResourcePool {
    cells: Vec<Arc<Cell>>,
}
impl ResourcePool {
    /// # Errors
    /// Rejects zero/oversized capacities and allocation failure.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > MAX_POOL_CAPACITY {
            return Err(Error::invalid("invalid resource pool capacity"));
        }
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        cells.resize_with(capacity, || {
            Arc::new(Cell {
                leased: AtomicBool::new(false),
                readers: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                state: AtomicU8::new(0),
                value: Mutex::new(None),
            })
        });
        for cell in &cells {
            // Darwin's std mutex allocates lazily: force its OS storage on the cold startup path.
            drop(
                cell.value
                    .lock()
                    .map_err(|_| Error::invariant("new resource cell poisoned"))?,
            );
        }
        Ok(Self { cells })
    }
    /// # Errors
    /// Returns capacity until the previous ticket and backend responder have both retired.
    pub fn channel(&self) -> Result<(ResourceTicket, ResourceResponder)> {
        for cell in &self.cells {
            if cell
                .leased
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            cell.readers.store(2, Ordering::Relaxed);
            let receiver = Reader { cell: cell.clone() };
            let reply = Reply {
                reader: Reader { cell: cell.clone() },
                published: false,
            };
            return Ok((
                ResourceTicket::pooled(receiver),
                ResourceResponder::pooled(reply),
            ));
        }
        Err(Error::new(
            ErrorCode::Capacity,
            "resource acknowledgement pool exhausted",
        ))
    }
}
