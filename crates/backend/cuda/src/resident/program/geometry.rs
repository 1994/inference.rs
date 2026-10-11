//! One description of the graph rows retained by capture, workspace preparation and admission.
use super::ProgramWeights;

/// Each prompt width owns an activation arena; projection buffers share distinct widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramGeometry {
    pub verification: usize,
    /// Wide and narrow prompt graphs, respectively; zero means absent.
    pub prompts: [usize; 2],
}
impl ProgramGeometry {
    pub(crate) const fn new(weights: &ProgramWeights, requested: bool) -> Self {
        let verification = requested || weights.prefill_width < crate::constants::PREFILL_LANES;
        let wide_prompt =
            verification || weights.narrow_prefill_width < crate::constants::PREFILL_LANES;
        Self::from_flags(
            weights.batch_width,
            weights.prefill_width,
            weights.narrow_prefill_width,
            verification,
            wide_prompt,
        )
    }
    pub(crate) const fn from_flags(
        batch: usize,
        wide: usize,
        narrow: usize,
        verification: bool,
        wide_prompt: bool,
    ) -> Self {
        Self {
            verification: if verification && batch > 1 { batch } else { 0 },
            prompts: [
                if wide_prompt && wide >= crate::constants::PREFILL_LANES {
                    wide
                } else {
                    0
                },
                if narrow >= crate::constants::PREFILL_LANES && (narrow < wide || !wide_prompt) {
                    narrow
                } else {
                    0
                },
            ],
        }
    }
    pub(crate) const fn prompt_lanes(self) -> usize {
        self.prompts[0] + self.prompts[1]
    }
    pub(crate) const fn arena_lanes(self) -> usize {
        1 + self.verification + self.prompt_lanes()
    }
    pub(crate) fn widths(self) -> Vec<usize> {
        let mut widths = vec![1];
        for width in [self.verification, self.prompts[1], self.prompts[0]] {
            if width > 1 && !widths.contains(&width) {
                widths.push(width);
            }
        }
        widths
    }
}
