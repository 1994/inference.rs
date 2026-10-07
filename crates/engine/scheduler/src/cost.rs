//! Bounded, deterministic calibration keyed by execution phase and batch shape.
use infer_core::{Error, Result, map::BoundedMap};
use infer_ir::{
    CostEstimate, CostModelConfig, CostModelInspection, CostObservation, CostQuery, ExecutionRole,
    TimingSource,
};
use infer_spi::{BatchCostSummary, CostModelProvider};
use serde::{Deserialize, Serialize};

/// Largest accepted `CostModelConfig::max_profiles` calibration table size.
const MAX_PROFILE_CAPACITY: usize = 4096;
/// Largest accepted `CostModelConfig::safety_margin_percent`.
const MAX_SAFETY_MARGIN_PERCENT: u32 = 200;
/// Manhattan radius, in log2 token/context/batch buckets, searched for a nearby profile.
const NEIGHBOR_RADIUS: i32 = 3;
/// Largest accepted serialized cost-calibration checkpoint, in bytes.
const MAX_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;

mod summary;
pub use summary::update_summary;

pub struct FallbackCosts;
impl CostModelProvider for FallbackCosts {
    fn identity(&self) -> &'static str {
        "fallback-cost-v1"
    }
    fn estimate(&self, work: &[CostQuery]) -> Result<CostEstimate> {
        aggregate(work)
    }
    fn supports_batch_summary(&self) -> bool {
        true
    }
    fn estimate_summary(&self, summary: &BatchCostSummary) -> Result<Option<CostEstimate>> {
        Ok(Some(summary.fallback))
    }
}
///
/// # Errors
/// Returns an invalid-input error for invalid queries, mixed programs or backends, or overflowing aggregate costs. An empty batch has zero cost.
pub fn aggregate(work: &[CostQuery]) -> Result<CostEstimate> {
    let mut cost = CostEstimate::default();
    for q in work {
        if q.tokens == 0
            || q.context_tokens == 0
            || q.fallback_per_token_us == 0
            || q.role == ExecutionRole::Mixed
        {
            return Err(Error::invalid("invalid cost query"));
        }
        if work
            .first()
            .is_some_and(|first| first.program != q.program || first.backend != q.backend)
        {
            return Err(Error::invalid("incompatible cost query program/backend"));
        }
        cost.gpu_us = cost
            .gpu_us
            .checked_add(
                q.fallback_per_token_us
                    .checked_mul(q.tokens as u64)
                    .ok_or_else(|| Error::invalid("execution cost overflow"))?,
            )
            .ok_or_else(|| Error::invalid("execution cost overflow"))?;
        cost.workspace_bytes = cost.workspace_bytes.max(q.workspace_bytes);
        cost.num_gpu_blocks = cost
            .num_gpu_blocks
            .checked_add(q.num_gpu_blocks)
            .ok_or_else(|| Error::invalid("state cost overflow"))?;
        if let Some(growth) = q.page_growth {
            let n = growth
                .required_pages(q.context_tokens)
                .ok_or_else(|| Error::invalid("invalid physical page growth"))?;
            cost.num_gpu_blocks = cost
                .num_gpu_blocks
                .checked_add(n)
                .ok_or_else(|| Error::invalid("page growth overflow"))?;
            cost.state_bytes = cost
                .state_bytes
                .checked_add(
                    growth
                        .bytes_per_page
                        .checked_mul(n as u64)
                        .ok_or_else(|| Error::invalid("page byte growth overflow"))?,
                )
                .ok_or_else(|| Error::invalid("page byte growth overflow"))?;
        }
        if let Some(growth) = q.logical_growth {
            let n = growth
                .required_pages(q.context_tokens)
                .ok_or_else(|| Error::invalid("invalid logical page growth"))?;
            cost.logical_pages = cost
                .logical_pages
                .checked_add(n)
                .ok_or_else(|| Error::invalid("logical growth overflow"))?;
        }
        for (total, value) in [
            (&mut cost.state_bytes, q.state_bytes),
            (&mut cost.transfer_us, q.transfer_us),
            (&mut cost.encoder_us, q.encoder_us),
        ] {
            *total = total
                .checked_add(value)
                .ok_or_else(|| Error::invalid("resource cost overflow"))?;
        }
    }
    Ok(cost)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct Shape {
    role: ExecutionRole,
    tokens: u32,
    context: u32,
    batch: u32,
}
fn bucket(n: usize) -> u32 {
    usize::BITS - n.max(1).saturating_sub(1).leading_zeros()
}
fn shape(work: &[CostQuery]) -> Result<Shape> {
    if work.is_empty() {
        return Err(Error::invalid("empty cost observation"));
    }
    let role = work[0].role;
    let tokens = work
        .iter()
        .try_fold(0usize, |n, q| n.checked_add(q.tokens))
        .ok_or_else(|| Error::invalid("cost token count overflow"))?;
    Ok(Shape {
        role: if work.iter().all(|q| q.role == role) {
            role
        } else {
            ExecutionRole::Mixed
        },
        tokens: bucket(tokens),
        context: bucket(
            work.iter()
                .map(|q| q.context_tokens)
                .max()
                .ok_or_else(|| Error::invalid("empty cost observation"))?,
        ),
        batch: bucket(work.len()),
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Profile {
    shape: Shape,
    mean_us: u64,
    reference_us: u64,
    samples: u64,
    updated: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Calibration {
    version: u32,
    binding: String,
    config: CostModelConfig,
    profiles: Vec<Profile>,
    ratio_ppm: u64,
    observations: u64,
    evictions: u64,
    last_source: Option<TimingSource>,
}
pub struct CalibratedCosts {
    state: Calibration,
    profiles: BoundedMap<Shape, Profile>,
}
impl CalibratedCosts {
    ///
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(binding: String, config: CostModelConfig) -> Result<Self> {
        if binding.is_empty()
            || config.max_profiles == 0
            || config.max_profiles > MAX_PROFILE_CAPACITY
            || config.ewma_alpha_percent == 0
            || config.ewma_alpha_percent > crate::constants::PERCENT
            || config.safety_margin_percent > MAX_SAFETY_MARGIN_PERCENT
        {
            return Err(Error::invalid("invalid cost calibration configuration"));
        }
        let profiles = BoundedMap::new(config.max_profiles)?;
        Ok(Self {
            state: Calibration {
                version: 1,
                binding,
                config,
                profiles: vec![],
                ratio_ppm: crate::constants::PARTS_PER_MILLION,
                observations: 0,
                evictions: 0,
                last_source: None,
            },
            profiles,
        })
    }
    fn calibrated(&self, mut cost: CostEstimate, key: Shape) -> Result<CostEstimate> {
        let candidate = self.profiles.get(&key).or_else(|| self.nearest(key));
        let predicted = candidate.map_or_else(
            || {
                (u128::from(cost.gpu_us) * u128::from(self.state.ratio_ppm))
                    .div_ceil(u128::from(crate::constants::PARTS_PER_MILLION))
            },
            |p| {
                (u128::from(cost.gpu_us) * u128::from(p.mean_us))
                    .div_ceil(u128::from(p.reference_us))
            },
        );
        cost.gpu_us = u64::try_from(
            predicted
                .checked_mul(
                    u128::from(crate::constants::PERCENT)
                        + u128::from(self.state.config.safety_margin_percent),
                )
                .ok_or_else(|| Error::invalid("calibrated execution cost overflow"))?
                .div_ceil(u128::from(crate::constants::PERCENT)),
        )
        .map_err(|_| Error::invalid("calibrated execution cost overflow"))?
        .max(1);
        Ok(cost)
    }
    fn nearest(&self, key: Shape) -> Option<&Profile> {
        let mut best: Option<(u32, &Profile)> = None;
        for dx in -NEIGHBOR_RADIUS..=NEIGHBOR_RADIUS {
            for dy in -NEIGHBOR_RADIUS..=NEIGHBOR_RADIUS {
                for dz in -NEIGHBOR_RADIUS..=NEIGHBOR_RADIUS {
                    let distance = dx.unsigned_abs() + dy.unsigned_abs() + dz.unsigned_abs();
                    if distance > NEIGHBOR_RADIUS.unsigned_abs() {
                        continue;
                    }
                    let Some((tokens, context, batch)) = key
                        .tokens
                        .checked_add_signed(dx)
                        .zip(key.context.checked_add_signed(dy))
                        .zip(key.batch.checked_add_signed(dz))
                        .map(|((tokens, context), batch)| (tokens, context, batch))
                    else {
                        continue;
                    };
                    if let Some(profile) = self.profiles.get(&Shape {
                        role: key.role,
                        tokens,
                        context,
                        batch,
                    }) {
                        let rank = (distance, std::cmp::Reverse(profile.samples), profile.shape);
                        if best.is_none_or(|(old_distance, old)| {
                            rank < (old_distance, std::cmp::Reverse(old.samples), old.shape)
                        }) {
                            best = Some((distance, profile));
                        }
                    }
                }
            }
        }
        best.map(|(_, profile)| profile)
    }
    fn blend(&self, old: u64, new: u64) -> u64 {
        let alpha = u128::from(self.state.config.ewma_alpha_percent);
        u64::try_from(
            (u128::from(old) * (u128::from(crate::constants::PERCENT) - alpha)
                + u128::from(new) * alpha)
                .div_ceil(u128::from(crate::constants::PERCENT)),
        )
        .unwrap_or(u64::MAX)
    }
}
impl CostModelProvider for CalibratedCosts {
    fn identity(&self) -> &'static str {
        "shape-ewma-cost-v1"
    }
    fn estimate(&self, work: &[CostQuery]) -> Result<CostEstimate> {
        let cost = aggregate(work)?;
        if work.is_empty() || !self.state.config.adaptive || self.state.observations == 0 {
            return Ok(cost);
        }
        let key = shape(work)?;
        self.calibrated(cost, key)
    }
    fn supports_batch_summary(&self) -> bool {
        true
    }
    fn estimate_summary(&self, summary: &BatchCostSummary) -> Result<Option<CostEstimate>> {
        if summary.batch == 0 || !self.state.config.adaptive || self.state.observations == 0 {
            return Ok(Some(summary.fallback));
        }
        let key = summary::shape(summary);
        self.calibrated(summary.fallback, key).map(Some)
    }
    fn observe(&mut self, observation: &CostObservation) -> Result<()> {
        let reference = aggregate(&observation.work)?.gpu_us;
        let key = shape(&observation.work)?;
        if observation.timing.elapsed_us == 0 || reference == 0 {
            return Err(Error::invalid("zero cost observation"));
        }
        let backend = observation.work[0].backend;
        if !observation.timing.matches_backend(backend) {
            return Err(Error::invalid("cost timing source does not match backend"));
        }
        if !self.state.config.adaptive {
            return Ok(());
        }
        let elapsed = observation.timing.elapsed_us;
        let observations = self
            .state
            .observations
            .checked_add(1)
            .ok_or_else(|| Error::invalid("cost observation counter overflow"))?;
        let ratio = u64::try_from(
            (u128::from(elapsed) * u128::from(crate::constants::PARTS_PER_MILLION))
                .div_ceil(u128::from(reference)),
        )
        .unwrap_or(u64::MAX);
        self.state.ratio_ppm = if self.state.observations == 0 {
            ratio
        } else {
            self.blend(self.state.ratio_ppm, ratio)
        };
        self.state.observations = observations;
        self.state.last_source = Some(observation.timing.source);
        let profile = if let Some(old) = self.profiles.get(&key) {
            Profile {
                shape: key,
                mean_us: self.blend(old.mean_us, elapsed),
                reference_us: self.blend(old.reference_us, reference),
                samples: old.samples.saturating_add(1),
                updated: self.state.observations,
            }
        } else {
            Profile {
                shape: key,
                mean_us: elapsed,
                reference_us: reference,
                samples: 1,
                updated: self.state.observations,
            }
        };
        if !self.profiles.contains_key(&key)
            && self.profiles.len() == self.state.config.max_profiles
        {
            let oldest = self
                .profiles
                .values()
                .min_by_key(|p| (p.updated, &p.shape))
                .ok_or_else(|| Error::invariant("calibration eviction has no profile"))?
                .shape;
            self.profiles.remove(&oldest);
            self.state.evictions = self.state.evictions.saturating_add(1);
        }
        self.profiles.insert(key, profile)?;
        Ok(())
    }
    fn capture_state(&self) -> Result<Option<Vec<u8>>> {
        let mut saved = self.state.clone();
        saved.profiles = self.profiles.values().cloned().collect();
        saved.profiles.sort_unstable_by_key(|profile| profile.shape);
        Ok(Some(
            serde_json::to_vec(&saved).map_err(|e| Error::invalid(e.to_string()))?,
        ))
    }
    fn restore_state(&mut self, data: Option<&[u8]>) -> Result<()> {
        let data = data.ok_or_else(|| Error::invalid("cost calibration checkpoint required"))?;
        if data.len() > MAX_CHECKPOINT_BYTES {
            return Err(Error::invalid("cost checkpoint exceeds budget"));
        }
        let saved: Calibration =
            serde_json::from_slice(data).map_err(|e| Error::invalid(e.to_string()))?;
        if saved.version != 1
            || saved.binding != self.state.binding
            || saved.config != self.state.config
            || saved.profiles.len() > saved.config.max_profiles
            || saved.ratio_ppm == 0
            || saved.evictions > saved.observations
            || (saved.observations == 0) != saved.profiles.is_empty()
            || (saved.observations == 0) != saved.last_source.is_none()
        {
            return Err(Error::invalid(
                "cost checkpoint binding/configuration mismatch",
            ));
        }
        let mut profiles = BoundedMap::new(saved.config.max_profiles)?;
        for p in &saved.profiles {
            if p.mean_us == 0
                || p.reference_us == 0
                || p.samples == 0
                || p.samples > saved.observations
                || p.updated == 0
                || p.updated > saved.observations
                || p.shape.tokens > usize::BITS
                || p.shape.context > usize::BITS
                || p.shape.batch > usize::BITS
                || profiles.insert(p.shape, p.clone())?.is_some()
            {
                return Err(Error::invalid("invalid cost checkpoint profile"));
            }
        }
        self.state = saved;
        self.state.profiles.clear();
        self.profiles = profiles;
        Ok(())
    }
    fn inspect(&self) -> CostModelInspection {
        CostModelInspection {
            provider: self.identity().into(),
            adaptive: self.state.config.adaptive,
            profiles: self.profiles.len(),
            observations: self.state.observations,
            evictions: self.state.evictions,
            last_source: self.state.last_source,
        }
    }
}
