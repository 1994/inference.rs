//! Multimodal rotary geometry, independent of the execution backend.
use crate::{PreparedPrompt, VisualPlacement};
use infer_core::{Error, Result};
use infer_ir::PositionSpec;

/// Number of temporal and spatial rotary axes.
pub const ROTARY_AXES: usize = 3;

/// Axis (time, height, width) used by each rotary frequency.
/// # Errors
/// Rejects sections that do not cover the rotary half-head.
pub fn axes(spec: &PositionSpec, half: usize) -> Result<Vec<i32>> {
    if spec.multimodal_sections.is_empty() {
        return Ok(vec![0; half]);
    }
    let [time, height, width] = spec.multimodal_sections.as_slice() else {
        return Err(Error::invalid("MRoPE requires three sections"));
    };
    if time
        .checked_add(*height)
        .and_then(|n| n.checked_add(*width))
        != Some(half)
    {
        return Err(Error::invalid(
            "MRoPE sections must cover the rotary half-head",
        ));
    }
    let mut result = vec![0; half];
    if spec.interleaved {
        for (axis, count) in [(1usize, *height), (2usize, *width)] {
            for index in 0..count {
                let lane = index
                    .checked_mul(ROTARY_AXES)
                    .and_then(|n| n.checked_add(axis))
                    .filter(|n| *n < half)
                    .ok_or_else(|| Error::invalid("interleaved MRoPE section overflow"))?;
                result[lane] = if axis == 1 { 1 } else { 2 };
            }
        }
    } else {
        result[*time..time + height].fill(1);
        result[time + height..].fill(2);
    }
    Ok(result)
}

/// Three-axis positions for merged image patches, including intervening text.
/// # Errors
/// Rejects malformed, overlapping or out-of-range media spans and grids.
pub fn image_positions(prompt: &PreparedPrompt, merge: usize) -> Result<Vec<[usize; ROTARY_AXES]>> {
    if merge == 0 || prompt.spans.len() != prompt.images.len() {
        return Err(Error::invalid("MRoPE media geometry"));
    }
    let mut result = Vec::with_capacity(prompt.tokens.len());
    let mut cursor = 0usize;
    for (span, image) in prompt.spans.iter().zip(&prompt.images) {
        let (t, h, w) = image.grid;
        if t == 0
            || h == 0
            || w == 0
            || !h.is_multiple_of(merge)
            || !w.is_multiple_of(merge)
            || span.start < result.len()
            || span.end > prompt.tokens.len()
            || span.end <= span.start
        {
            return Err(Error::invalid("MRoPE grid/span mismatch"));
        }
        let (h, w) = (h / merge, w / merge);
        if t.checked_mul(h).and_then(|n| n.checked_mul(w)) != Some(span.len()) {
            return Err(Error::invalid("MRoPE span must contain every merged patch"));
        }
        while result.len() < span.start {
            result.push([cursor; ROTARY_AXES]);
            cursor = cursor
                .checked_add(1)
                .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        }
        let limit = cursor
            .checked_add(t.max(h).max(w))
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        for time in 0..t {
            for row in 0..h {
                for col in 0..w {
                    result.push([cursor + time, cursor + row, cursor + col]);
                }
            }
        }
        cursor = limit;
    }
    while result.len() < prompt.tokens.len() {
        result.push([cursor; ROTARY_AXES]);
        cursor = cursor
            .checked_add(1)
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
    }
    Ok(result)
}

/// Reconstruct text positions from placed media and return the next decode coordinate.
/// # Errors
/// Rejects duplicate/out-of-range placements and overflowing coordinates.
pub fn placed_positions(
    tokens: usize,
    media: &[VisualPlacement],
) -> Result<(Vec<[usize; ROTARY_AXES]>, usize)> {
    let mut placed = std::collections::BTreeMap::new();
    for item in media {
        if item.position >= tokens || placed.insert(item.position, item.rotary_position).is_some() {
            return Err(Error::invalid(
                "MRoPE placement outside prompt or duplicated",
            ));
        }
    }
    let mut next = 0usize;
    let mut result = Vec::with_capacity(tokens);
    for index in 0..tokens {
        let coordinates = placed
            .get(&index)
            .copied()
            .flatten()
            .unwrap_or([next; ROTARY_AXES]);
        let end = coordinates
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::invalid("MRoPE position overflow"))?;
        next = next.max(end);
        result.push(coordinates);
    }
    Ok((result, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PromptImage;

    #[test]
    fn frequency_axes_match_interleaved_and_contiguous_sections() {
        let mut spec = PositionSpec {
            rope_theta: 5_000_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![24, 20, 20],
            interleaved: true,
        };
        let actual = axes(&spec, 64).unwrap();
        for (index, axis) in actual.iter().enumerate() {
            assert_eq!(*axis, if index < 60 { [0, 1, 2][index % 3] } else { 0 });
        }
        spec.interleaved = false;
        assert_eq!(
            axes(&spec, 64).unwrap(),
            [vec![0; 24], vec![1; 20], vec![2; 20]].concat()
        );
        assert!(axes(&spec, 32).is_err());
        spec.multimodal_sections.clear();
        assert_eq!(axes(&spec, 32).unwrap(), vec![0; 32]);
    }

    #[test]
    fn image_axes_and_decode_position_are_independent_of_kv_length() {
        let prompt = PreparedPrompt {
            tokens: vec![0; 95],
            spans: std::iter::once(4..81).collect(),
            images: vec![PromptImage {
                pixels: vec![],
                grid: (1, 14, 22),
            }],
        };
        let expected = image_positions(&prompt, 2).unwrap();
        assert_eq!(expected[3], [3, 3, 3]);
        assert_eq!(expected[4], [4, 4, 4]);
        assert_eq!(expected[80], [4, 10, 14]);
        assert_eq!(expected[81], [15, 15, 15]);
        assert_eq!(expected[94], [28, 28, 28]);
        let media: Vec<_> = (4..81)
            .map(|position| VisualPlacement {
                position,
                embedding: vec![],
                rotary_position: Some(expected[position]),
            })
            .collect();
        let (actual, next) = placed_positions(95, &media).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(next, 29);
        assert_eq!(placed_positions(95, &[]).unwrap().1, 95);
    }

    #[test]
    fn media_geometry_is_validated_before_positions_are_emitted() {
        let mut prompt = PreparedPrompt {
            tokens: vec![0; 10],
            spans: vec![2..6, 7..9],
            images: vec![
                PromptImage {
                    pixels: vec![],
                    grid: (1, 4, 4),
                },
                PromptImage {
                    pixels: vec![],
                    grid: (1, 2, 4),
                },
            ],
        };
        let positions = image_positions(&prompt, 2).unwrap();
        assert_eq!(positions[6], [4, 4, 4]);
        assert_eq!(positions[7], [5, 5, 5]);
        assert_eq!(positions[9], [7, 7, 7]);
        prompt.spans[1] = 4..6;
        assert!(image_positions(&prompt, 2).is_err());
        assert!(image_positions(&prompt, 0).is_err());
    }
}
