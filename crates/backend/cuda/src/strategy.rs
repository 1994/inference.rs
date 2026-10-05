//! Kernel strategy independent of model names, storage and device ownership.
use infer_core::{Error, Result};
use serde::Serialize;

/// Validated tile dimensions for a row-major decode projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LinearTiling {
    rows: usize,
    columns: usize,
}

impl LinearTiling {
    /// # Errors
    /// Rejects unsupported tile sizes before specializing device code.
    pub fn new(rows: usize, columns: usize) -> Result<Self> {
        if ![1, 2, 4, 8, 16].contains(&rows)
            || !(64..=32768).contains(&columns)
            || !columns.is_power_of_two()
        {
            return Err(Error::invalid("unsupported CUDA linear tile"));
        }
        Ok(Self { rows, columns })
    }

    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }
}

/// Strategy boundary: model adapters supply dimensions, policies supply launch choices.
pub trait LinearStrategy {
    fn candidates(&self, columns: usize) -> Vec<LinearTiling>;
}

impl LinearStrategy for LinearTiling {
    fn candidates(&self, _columns: usize) -> Vec<LinearTiling> {
        vec![*self]
    }
}

/// Bounded search; the original 4 × 128 kernel remains the baseline.
pub struct DecodeSearch;

impl LinearStrategy for DecodeSearch {
    fn candidates(&self, columns: usize) -> Vec<LinearTiling> {
        let wide = columns.clamp(64, 32768).next_power_of_two();
        [
            (4, 128),
            (1, 128),
            (1, 256),
            (1, 512),
            (2, 128),
            (2, 256),
            (2, 512),
            (4, 256),
            (4, 512),
            (4, 1024),
            (8, 128),
            (8, 256),
            (8, 512),
            (16, 128),
            (16, 256),
            (1, wide),
            (2, wide),
            (4, wide),
        ]
        .into_iter()
        .map(|(rows, columns)| LinearTiling { rows, columns })
        .collect()
    }
}
