use super::{BASELINE, CANDIDATES, SUPPORTED_DIMENSION};
use serde::{Deserialize, Serialize};
const MEDIAN_WEIGHT: f32 = 0.5;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Shape {
    pub rows: usize,
    pub dimension: usize,
    pub epsilon_bits: u32,
}
impl Shape {
    pub fn new(rows: usize, dimension: usize, epsilon: f32) -> Option<Self> {
        (rows > 0
            && rows.checked_mul(dimension).is_some()
            && i32::try_from(rows).is_ok()
            && dimension == SUPPORTED_DIMENSION
            && epsilon.is_finite()
            && epsilon > 0.0)
            .then_some(Self {
                rows,
                dimension,
                epsilon_bits: epsilon.to_bits(),
            })
    }
    fn valid(self) -> bool {
        Self::new(self.rows, self.dimension, f32::from_bits(self.epsilon_bits)) == Some(self)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_version: u32,
    pub driver_build: Option<String>,
    pub kernel_fingerprint: String,
    pub shape: Shape,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Times {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
}
impl Times {
    fn valid(&self) -> bool {
        self.baseline_ms.len() >= super::super::MIN_REPS
            && self.baseline_ms.len() == self.candidate_ms.len()
            && self
                .baseline_ms
                .iter()
                .chain(&self.candidate_ms)
                .all(|v| v.is_finite() && *v > 0.0)
    }
    pub fn speedup(&self) -> f32 {
        median(&self.baseline_ms) / median(&self.candidate_ms)
    }
}
fn median(xs: &[f32]) -> f32 {
    let mut xs = xs.to_vec();
    xs.sort_by(f32::total_cmp);
    let n = xs.len();
    if n == 0 {
        return f32::NAN;
    }
    if n.is_multiple_of(2) {
        (xs[n / 2 - 1] + xs[n / 2]) * MEDIAN_WEIGHT
    } else {
        xs[n / 2]
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub occupancy: i32,
    pub candidate_first: bool,
    pub verified_elements: usize,
    pub cold: Times,
    pub warm: Times,
}
impl Trial {
    fn valid(&self) -> bool {
        (self.occupancy == BASELINE || CANDIDATES.contains(&self.occupancy))
            && self.verified_elements > 0
            && self.cold.valid()
            && self.warm.valid()
    }
    pub fn confirmed(&self) -> bool {
        self.valid() && self.score() >= super::super::MIN_CONFIRM_SPEEDUP
    }
    pub fn score(&self) -> f32 {
        self.cold.speedup().min(self.warm.speedup())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: i32,
    pub search: Vec<Trial>,
    pub confirmation: Trial,
}
impl Entry {
    #[must_use]
    pub fn valid(&self) -> bool {
        self.key.shape.valid()
            && !self.key.kernel_fingerprint.is_empty()
            && !self.key.hardware.is_empty()
            && self.confirmation.valid()
            && self.confirmation.candidate_first
            && self.confirmation.verified_elements == self.key.shape.rows * self.key.shape.dimension
            && self.selected == self.confirmation.occupancy
            && self.search.len() == CANDIDATES.len()
            && CANDIDATES.iter().all(|o| {
                self.search.iter().any(|t| {
                    t.occupancy == *o
                        && !t.candidate_first
                        && t.valid()
                        && t.verified_elements == self.confirmation.verified_elements
                })
            })
            && (self.selected == BASELINE
                || self.confirmation.confirmed()
                    && self
                        .search
                        .iter()
                        .any(|t| t.occupancy == self.selected && t.confirmed()))
    }
}
