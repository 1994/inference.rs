//! Multimodal prompt assembly: media placeholders, expansion and encoder gating.
//!
//! The chat template emits one placeholder token per image. The processor then expands each
//! placeholder into one token per *merged* visual patch, so the token stream and the vision
//! embeddings line up one-to-one. Every mismatch is an error: a prompt that cannot be aligned
//! with its media must never fall back to the text-only path.
use crate::image::PreprocessedImage;
use infer_core::{Error, Result};
use infer_ir::Modality;
use infer_spi::{ImportedModel, ModalityEncoder};
use std::ops::Range;

/// One preprocessed image attached to a prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptImage {
    /// Patch vectors from [`crate::ImageProcessor::preprocess`].
    pub pixels: Vec<f32>,
    /// `(temporal, height, width)` patch grid.
    pub grid: (usize, usize, usize),
}

impl PromptImage {
    /// Wraps one preprocessed image.
    #[must_use]
    pub fn new(processed: PreprocessedImage) -> Self {
        Self {
            pixels: processed.pixels,
            grid: processed.grid,
        }
    }
}

/// Number of merged visual tokens an image contributes.
/// # Errors
/// Rejects grids that do not divide by the merge unit or overflow.
pub fn visual_tokens(grid: (usize, usize, usize), encoder: &ModalityEncoder) -> Result<usize> {
    let unit = crate::vision::merge_unit(encoder)?;
    let patches = grid
        .0
        .checked_mul(grid.1)
        .and_then(|value| value.checked_mul(grid.2))
        .ok_or_else(|| Error::invalid("patch grid overflow"))?;
    if !patches.is_multiple_of(unit) {
        return Err(Error::invalid(
            "patch grid does not divide by the merge unit",
        ));
    }
    Ok(patches / unit)
}

/// Encoder of one modality, or an explicit failure when the model has none.
///
/// # Errors
/// Returns an unsupported error naming the modality when the family declares no encoder.
pub fn require_encoder<'a>(
    imported: &'a ImportedModel,
    modality: &Modality,
) -> Result<&'a ModalityEncoder> {
    imported
        .modalities
        .iter()
        .find(|plan| &plan.modality == modality)
        .and_then(|plan| plan.encoder.as_ref())
        .ok_or_else(|| {
            Error::unsupported(format!(
                "model declares no {modality:?} encoder, yet the request carries {modality:?} media"
            ))
        })
}

/// Placeholder token ids of one modality.
/// # Errors
/// Returns an unsupported error naming the modality when the family declares no placeholders.
pub fn placeholders<'a>(imported: &'a ImportedModel, modality: &Modality) -> Result<&'a [u32]> {
    imported
        .modalities
        .iter()
        .find(|plan| &plan.modality == modality)
        .map(|plan| plan.placeholder_tokens.as_slice())
        .filter(|tokens| !tokens.is_empty())
        .ok_or_else(|| {
            Error::unsupported(format!("model declares no {modality:?} placeholder tokens"))
        })
}

/// Result of expanding media placeholders in a token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedPrompt {
    /// Token stream with every placeholder replaced by its media's token run.
    pub tokens: Vec<u32>,
    /// Token range each image's embeddings occupy, in media order.
    pub spans: Vec<Range<usize>>,
}

/// Expand every media placeholder into one token per merged patch.
///
/// `counts[i]` is the token count of the `i`-th media item. The stream must hold exactly
/// `counts.len()` placeholders: a different number means the prompt and its media disagree.
///
/// # Errors
/// Rejects a placeholder count that disagrees with the media count, an empty media list or
/// overflowing token counts.
pub fn expand_placeholders(
    tokens: &[u32],
    counts: &[usize],
    placeholder: u32,
) -> Result<ExpandedPrompt> {
    if counts.is_empty() {
        return Err(Error::invalid("no media to expand"));
    }
    let found = tokens.iter().filter(|token| **token == placeholder).count();
    if found != counts.len() {
        return Err(Error::invalid(format!(
            "prompt holds {found} media placeholders for {} media items",
            counts.len()
        )));
    }
    let capacity = tokens.len()
        + counts
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| Error::invalid("expanded prompt overflow"))?;
    let mut expanded = Vec::with_capacity(capacity);
    let mut spans = Vec::with_capacity(counts.len());
    let mut media = counts.iter();
    for token in tokens {
        if *token != placeholder {
            expanded.push(*token);
            continue;
        }
        let count = *media
            .next()
            .ok_or_else(|| Error::invariant("media placeholders were counted inconsistently"))?;
        if count == 0 {
            return Err(Error::invalid("media contributes no visual tokens"));
        }
        let start = expanded.len();
        expanded.extend(std::iter::repeat_n(placeholder, count));
        spans.push(start..expanded.len());
    }
    Ok(ExpandedPrompt {
        tokens: expanded,
        spans,
    })
}

/// Visual token counts of every image, validated against the declared encoder.
/// # Errors
/// Rejects a model without an image encoder or a grid that disagrees with the merge unit.
pub fn image_token_counts(imported: &ImportedModel, images: &[PromptImage]) -> Result<Vec<usize>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let encoder = require_encoder(imported, &Modality::Image)?;
    images
        .iter()
        .map(|image| visual_tokens(image.grid, encoder))
        .collect()
}

/// One decoded image attached to a prompt.
///
/// Decoding compressed formats is deliberately out of scope: the caller hands over RGB8 bytes and
/// their dimensions, so this crate stays free of image codecs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawImage {
    pub height: usize,
    pub width: usize,
    /// `height × width × 3` bytes, red first.
    pub pixels: Vec<u8>,
}

/// A prompt with its media preprocessed and its placeholders expanded.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedPrompt {
    /// Token stream the model consumes.
    pub tokens: Vec<u32>,
    /// Preprocessed images, in placeholder order.
    pub images: Vec<PromptImage>,
    /// Token range each image's embeddings replace, in the same order.
    pub spans: Vec<Range<usize>>,
}

/// Tokenize `text`, preprocess `images` and expand every placeholder.
///
/// Text-only prompts take the ordinary path. A prompt carrying media fails loudly when the model
/// declares no encoder, no placeholder, or a placeholder count that disagrees with the media —
/// there is no silent fall back to interpreting the placeholders as text.
///
/// # Errors
/// Rejects a missing encoder or placeholder, a placeholder/media mismatch, malformed image buffers
/// or a preprocessing failure.
pub fn prepare(
    assets: &crate::input::text::TextAssets,
    processor: &crate::image::ImageProcessor,
    imported: &ImportedModel,
    text: &str,
    images: &[RawImage],
    modality: &Modality,
) -> Result<PreparedPrompt> {
    let tokens = assets.encode(text, false)?;
    if images.is_empty() {
        return Ok(PreparedPrompt {
            tokens,
            images: Vec::new(),
            spans: Vec::new(),
        });
    }
    let encoder = require_encoder(imported, modality)?;
    let placeholder = *placeholders(imported, modality)?
        .first()
        .ok_or_else(|| Error::invalid("media modality declares no placeholder token"))?;
    let mut prepared = Vec::with_capacity(images.len());
    let mut counts = Vec::with_capacity(images.len());
    for image in images {
        let processed = processor.preprocess(&crate::image::RgbImage {
            height: image.height,
            width: image.width,
            pixels: image.pixels.clone(),
        })?;
        counts.push(visual_tokens(processed.grid, encoder)?);
        prepared.push(PromptImage::new(processed));
    }
    let expanded = expand_placeholders(&tokens, &counts, placeholder)?;
    Ok(PreparedPrompt {
        tokens: expanded.tokens,
        images: prepared,
        spans: expanded.spans,
    })
}

/// One visual embedding destined for a prompt position.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualPlacement {
    /// Token position the embedding replaces.
    pub position: usize,
    /// Hidden state of that visual token.
    pub embedding: Vec<f32>,
    /// Absolute three-axis rotary coordinate; absent for ordinary one-dimensional `RoPE`.
    pub rotary_position: Option<[usize; crate::mrope::ROTARY_AXES]>,
}

/// Align encoded media with the expanded prompt.
///
/// `embeddings[i]` is the flattened `tokens × hidden` output of the `i`-th media item's encoder.
/// Every span must hold exactly the placeholder tokens of that item and every embedding must match
/// its span length: a mismatch would silently shift the media relative to the text.
///
/// # Errors
/// Rejects a media/embedding count mismatch, a wrong embedding width, a span that does not hold the
/// placeholder token, or a model without declared placeholders.
pub fn place_visual_embeddings(
    prepared: &PreparedPrompt,
    imported: &ImportedModel,
    modality: &Modality,
    embeddings: &[Vec<f32>],
    hidden: usize,
) -> Result<Vec<VisualPlacement>> {
    if prepared.images.len() != embeddings.len() || prepared.spans.len() != embeddings.len() {
        return Err(Error::invalid(format!(
            "prompt holds {} media spans for {} encodings",
            prepared.spans.len(),
            embeddings.len()
        )));
    }
    let placeholder = *placeholders(imported, modality)?
        .first()
        .ok_or_else(|| Error::invalid("media modality declares no placeholder token"))?;
    let positions = if imported.model.position.multimodal_sections.is_empty() {
        None
    } else {
        Some(crate::mrope::image_positions(
            prepared,
            require_encoder(imported, modality)?.spatial_merge_size,
        )?)
    };
    let mut placements = Vec::new();
    for (span, encoded) in prepared.spans.iter().zip(embeddings) {
        let tokens = span.end.saturating_sub(span.start);
        if encoded.len() != tokens * hidden {
            return Err(Error::invalid(format!(
                "encoding holds {} values for {tokens} tokens of width {hidden}",
                encoded.len()
            )));
        }
        for (offset, token) in prepared.tokens[span.clone()].iter().enumerate() {
            if *token != placeholder {
                return Err(Error::invalid(
                    "media span does not hold the modality placeholder token",
                ));
            }
            placements.push(VisualPlacement {
                position: span.start + offset,
                rotary_position: positions.as_ref().map(|p| p[span.start + offset]),
                embedding: encoded[offset * hidden..(offset + 1) * hidden].to_vec(),
            });
        }
    }
    Ok(placements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_registry;
    use infer_core::ModelId;

    const IMAGE: u32 = 248_056;
    const VISION_START: u32 = 248_053;
    const VISION_END: u32 = 248_054;

    fn shipped() -> Result<ImportedModel> {
        let root = std::env::var("INFER_VISION_PACKAGE").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../examples/qwen3.8-27b").into()
        });
        let config = std::fs::read(std::path::Path::new(&root).join("config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?;
        let provider = default_registry().resolve(&config)?;
        provider.import(ModelId::ONE, &config)
    }

    fn encoder(imported: &ImportedModel) -> &ModalityEncoder {
        require_encoder(imported, &Modality::Image).expect("image encoder")
    }

    #[test]
    fn placeholder_runs_cover_every_merged_patch() -> Result<()> {
        let imported = shipped()?;
        let encoder = encoder(&imported);
        let images = [
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 4),
            },
            PromptImage {
                pixels: Vec::new(),
                grid: (1, 4, 6),
            },
        ];
        let counts = image_token_counts(&imported, &images)?;
        // 4x4 patches merge into 2x2 groups, 4x6 into 2x3.
        assert_eq!(counts, vec![4, 6]);
        let tokens = [
            VISION_START,
            IMAGE,
            VISION_END,
            7,
            VISION_START,
            IMAGE,
            VISION_END,
        ];
        let expanded = expand_placeholders(&tokens, &counts, IMAGE)?;
        assert_eq!(expanded.tokens.len(), tokens.len() + 8);
        assert_eq!(expanded.spans, vec![1..5, 8..14]);
        for span in &expanded.spans {
            assert!(expanded.tokens[span.clone()].iter().all(|id| *id == IMAGE));
        }
        assert_eq!(&expanded.tokens[1..2], &[IMAGE]);
        assert_eq!(encoder.out_hidden_size, 5120);
        Ok(())
    }

    /// The end-to-end prompt path: render, tokenize, preprocess and expand.
    ///
    /// Skipped unless the real package (with its tokenizer) is configured.
    #[test]
    fn prepared_prompt_lines_up_tokens_and_media() -> Result<()> {
        let Ok(package) = std::env::var("INFER_VISION_PACKAGE") else {
            return Ok(());
        };
        let imported = shipped()?;
        let assets = crate::input::text::TextAssets::open(&package, 4096)?;
        let processor = crate::image::ImageProcessor::from_json(
            &std::fs::read(std::path::Path::new(&package).join("preprocessor_config.json"))
                .map_err(|error| Error::invalid(error.to_string()))?,
        )?;
        let text = "<|vision_start|><|image_pad|><|vision_end|>describe";
        let image = RawImage {
            height: 4,
            width: 4,
            pixels: vec![128; 4 * 4 * 3],
        };
        let prepared = prepare(
            &assets,
            &processor,
            &imported,
            text,
            std::slice::from_ref(&image),
            &Modality::Image,
        )?;
        assert_eq!(prepared.images.len(), 1);
        assert_eq!(prepared.spans.len(), 1);
        let span = prepared.spans[0].clone();
        let (start, end) = (span.start, span.end);
        assert!(prepared.tokens[start..end].iter().all(|id| *id == IMAGE));
        // The placeholder itself is gone; only the expanded run remains.
        assert_eq!(
            prepared.tokens.iter().filter(|id| **id == IMAGE).count(),
            span.len()
        );
        let text_only = prepare(
            &assets,
            &processor,
            &imported,
            "describe",
            &[],
            &Modality::Image,
        )?;
        assert!(text_only.spans.is_empty() && text_only.images.is_empty());
        // A prompt whose text carries no placeholder must not silently drop the image.
        assert!(
            prepare(
                &assets,
                &processor,
                &imported,
                "describe",
                &[image],
                &Modality::Image
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn placements_follow_the_expanded_spans() -> Result<()> {
        let imported = shipped()?;
        let counts = [4usize, 6];
        let tokens = [
            VISION_START,
            IMAGE,
            VISION_END,
            7,
            VISION_START,
            IMAGE,
            VISION_END,
        ];
        let expanded = expand_placeholders(&tokens, &counts, IMAGE)?;
        let prepared = PreparedPrompt {
            tokens: expanded.tokens,
            images: vec![
                PromptImage {
                    pixels: Vec::new(),
                    grid: (1, 4, 4),
                },
                PromptImage {
                    pixels: Vec::new(),
                    grid: (1, 4, 6),
                },
            ],
            spans: expanded.spans,
        };
        let hidden = 3;
        let encodings = vec![vec![1.0f32; 4 * hidden], vec![2.0f32; 6 * hidden]];
        let placements =
            place_visual_embeddings(&prepared, &imported, &Modality::Image, &encodings, hidden)?;
        assert_eq!(placements.len(), 10);
        assert_eq!(placements[0].position, 1);
        assert_eq!(placements[0].embedding, vec![1.0; hidden]);
        assert_eq!(placements[4].position, 8);
        assert_eq!(placements[4].embedding, vec![2.0; hidden]);
        // A short encoding, a wrong width or a missing span must all fail loudly.
        assert!(
            place_visual_embeddings(
                &prepared,
                &imported,
                &Modality::Image,
                &[vec![1.0f32; 4 * hidden]],
                hidden
            )
            .is_err()
        );
        assert!(
            place_visual_embeddings(
                &prepared,
                &imported,
                &Modality::Image,
                &[vec![1.0f32; 4 * hidden], vec![2.0f32; 6 * hidden + 1]],
                hidden
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn placeholder_mismatch_is_an_explicit_error() {
        let counts = [4usize];
        let none = [VISION_START, VISION_END];
        assert!(expand_placeholders(&none, &counts, IMAGE).is_err());
        let two = [IMAGE, IMAGE];
        assert!(expand_placeholders(&two, &counts, IMAGE).is_err());
        assert!(expand_placeholders(&[IMAGE], &[], IMAGE).is_err());
        assert!(expand_placeholders(&[IMAGE], &[0], IMAGE).is_err());
    }

    #[test]
    fn missing_encoder_is_unsupported_not_silent() -> Result<()> {
        let mut imported = shipped()?;
        imported.modalities.clear();
        let Err(error) = require_encoder(&imported, &Modality::Image) else {
            panic!("a model without an image encoder must fail");
        };
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
        assert!(error.message.contains("Image"));
        let images = [PromptImage {
            pixels: Vec::new(),
            grid: (1, 4, 4),
        }];
        assert!(image_token_counts(&imported, &images).is_err());
        assert!(placeholders(&imported, &Modality::Image).is_err());
        assert!(placeholders(&imported, &Modality::Audio).is_err());
        Ok(())
    }

    #[test]
    fn square_grid_matches_the_visual_token_count() -> Result<()> {
        let imported = shipped()?;
        let encoder = encoder(&imported);
        assert_eq!(visual_tokens((2, 4, 4), encoder)?, 8);
        assert_eq!(visual_tokens((1, 4, 6), encoder)?, 6);
        // Nine patches cannot split into merge units of four.
        assert!(visual_tokens((1, 3, 3), encoder).is_err());
        Ok(())
    }
}
