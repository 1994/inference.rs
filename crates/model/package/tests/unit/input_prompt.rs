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
