//! Multimodal prompt execution on the resident path.
//!
//! Media tokens carry externally computed embeddings instead of a vocabulary lookup. The resident
//! single-token step already accepts such an override, so a prompt is prefilled token by token with
//! the media positions overridden. The alignment between media placeholders and encodings is
//! checked before any device work: a prompt whose media is not covered would otherwise execute as
//! ordinary text and silently drop the image.
use crate::resident::DeviceProgram;
use infer_core::{Error, Result};
use infer_models::VisualPlacement;
use infer_spi::ImportedModel;
use std::collections::{BTreeMap, BTreeSet};

/// Media placeholder tokens the model declares, across every modality.
#[must_use]
pub fn placeholder_tokens(imported: &ImportedModel) -> BTreeSet<u32> {
    imported
        .modalities
        .iter()
        .flat_map(|plan| plan.placeholder_tokens.iter().copied())
        .collect()
}

/// Positions of the media placeholders in `tokens`.
#[must_use]
pub fn media_positions(imported: &ImportedModel, tokens: &[u32]) -> Vec<usize> {
    let placeholders = placeholder_tokens(imported);
    tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| placeholders.contains(token))
        .map(|(position, _)| position)
        .collect()
}

/// Require that the encodings cover exactly the prompt's media placeholders.
///
/// # Errors
/// Rejects a prompt with media placeholders and no encodings, encodings for a prompt without
/// placeholders, duplicate positions, positions outside the prompt or zero-width encodings.
pub fn require_placements(
    imported: &ImportedModel,
    tokens: &[u32],
    placements: &[VisualPlacement],
) -> Result<()> {
    let declared = media_positions(imported, tokens);
    let mut given: Vec<usize> = placements.iter().map(|placed| placed.position).collect();
    given.sort_unstable();
    let unique = {
        let mut deduped = given.clone();
        deduped.dedup();
        deduped
    };
    if unique.len() != given.len() {
        return Err(Error::invalid("duplicate media placements"));
    }
    if declared.is_empty() && given.is_empty() {
        return Ok(());
    }
    if declared.is_empty() {
        return Err(Error::unsupported(
            "media encodings were supplied for a prompt without media placeholders",
        ));
    }
    if given.is_empty() {
        return Err(Error::unsupported(format!(
            "prompt holds {} media placeholder tokens without encodings",
            declared.len()
        )));
    }
    if declared != given {
        return Err(Error::invalid(format!(
            "prompt holds {} media placeholder tokens at different positions than the {} encodings",
            declared.len(),
            given.len()
        )));
    }
    if placements
        .iter()
        .any(|placed| placed.embedding.is_empty() || placed.position >= tokens.len())
    {
        return Err(Error::invalid("media placement is empty or out of range"));
    }
    Ok(())
}

/// Prefill a prompt, overriding media positions with their encodings, and return the logits of the
/// last prompt token.
///
/// # Errors
/// Rejects placements that do not cover the prompt's media, wrong embedding widths, out-of-range
/// positions and CUDA failures.
pub fn prefill(
    program: &mut DeviceProgram,
    imported: &ImportedModel,
    tokens: &[u32],
    placements: &[VisualPlacement],
    hidden: usize,
    position: usize,
) -> Result<Vec<f32>> {
    require_placements(imported, tokens, placements)?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty multimodal prompt"));
    }
    for placed in placements {
        if placed.embedding.len() != hidden {
            return Err(Error::invalid(format!(
                "media encoding holds {} values for a hidden size of {hidden}",
                placed.embedding.len()
            )));
        }
    }
    let mut by_position: BTreeMap<usize, &[f32]> = BTreeMap::new();
    for placed in placements {
        by_position.insert(placed.position, placed.embedding.as_slice());
    }
    let (rotary, _) = infer_models::mrope::placed_positions(tokens.len(), placements)?;
    program.reset()?;
    let mut logits = Vec::new();
    for (offset, token) in tokens.iter().enumerate() {
        let last = offset + 1 == tokens.len();
        let external = by_position.get(&offset).copied();
        let (_, row) = program.step_positioned(
            token.to_owned(),
            rotary[offset].map(|axis| position + axis),
            position + offset,
            external,
            last,
        )?;
        if last {
            logits = row;
        }
    }
    Ok(logits)
}

/// Greedy decode after a multimodal prefill.
///
/// # Errors
/// Rejects an empty vocabulary, an invalid budget or CUDA failures.
pub fn generate(
    program: &mut DeviceProgram,
    imported: &ImportedModel,
    tokens: &[u32],
    placements: &[VisualPlacement],
    hidden: usize,
    budget: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    if budget == 0 {
        return Err(Error::invalid("generation budget must be nonzero"));
    }
    let mut logits = prefill(program, imported, tokens, placements, hidden, 0)?;
    let (_, next_rotary) = infer_models::mrope::placed_positions(tokens.len(), placements)?;
    let mut generated = Vec::with_capacity(budget);
    for _ in 0..budget {
        let token = argmax(&logits)?;
        if stop.contains(&token) {
            break;
        }
        generated.push(token);
        let position = tokens.len() + generated.len() - 1;
        let (_, row) = program.step(
            token,
            next_rotary + generated.len() - 1,
            position,
            None,
            true,
        )?;
        logits = row;
    }
    Ok(generated)
}

/// Index of the largest logit, the greedy next token.
/// # Errors
/// Rejects an empty or non-finite logit row.
pub fn argmax(logits: &[f32]) -> Result<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::invalid("logits must be finite"));
        }
        if best.is_none_or(|(_, current)| *value > current) {
            best = Some((index, *value));
        }
    }
    best.map(|(index, _)| u32::try_from(index).unwrap_or(u32::MAX))
        .ok_or_else(|| Error::invalid("logits must not be empty"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_core::ModelId;

    const IMAGE: u32 = 248_056;

    fn shipped() -> Result<ImportedModel> {
        let root = std::env::var("INFER_VISION_PACKAGE").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../examples/qwen3.8-27b").into()
        });
        let config = std::fs::read(std::path::Path::new(&root).join("config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?;
        let provider = infer_models::default_registry().resolve(&config)?;
        provider.import(ModelId::ONE, &config)
    }

    #[test]
    fn placeholders_without_encodings_are_rejected() -> Result<()> {
        let imported = shipped()?;
        let tokens = [7u32, IMAGE, IMAGE, 9];
        assert_eq!(media_positions(&imported, &tokens), vec![1, 2]);
        let Err(error) = require_placements(&imported, &tokens, &[]) else {
            panic!("media placeholders without encodings must fail");
        };
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
        assert!(error.message.contains("without encodings"));
        // A text-only prompt stays valid.
        require_placements(&imported, &[1, 2, 3], &[])?;
        // Encodings for a prompt without placeholders are equally wrong.
        let stray = [VisualPlacement {
            rotary_position: None,
            position: 0,
            embedding: vec![0.0; 4],
        }];
        let Err(error) = require_placements(&imported, &[1, 2, 3], &stray) else {
            panic!("encodings without placeholders must fail");
        };
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
        Ok(())
    }

    #[test]
    fn misaligned_placements_are_rejected() -> Result<()> {
        let imported = shipped()?;
        let tokens = [7u32, IMAGE, IMAGE, 9];
        let wrong_position = [
            VisualPlacement {
                rotary_position: None,
                position: 0,
                embedding: vec![0.0; 4],
            },
            VisualPlacement {
                rotary_position: None,
                position: 1,
                embedding: vec![0.0; 4],
            },
        ];
        assert!(require_placements(&imported, &tokens, &wrong_position).is_err());
        let duplicate = [
            VisualPlacement {
                rotary_position: None,
                position: 1,
                embedding: vec![0.0; 4],
            },
            VisualPlacement {
                rotary_position: None,
                position: 1,
                embedding: vec![0.0; 4],
            },
        ];
        assert!(require_placements(&imported, &tokens, &duplicate).is_err());
        let empty = [
            VisualPlacement {
                rotary_position: None,
                position: 1,
                embedding: Vec::new(),
            },
            VisualPlacement {
                rotary_position: None,
                position: 2,
                embedding: vec![0.0; 4],
            },
        ];
        assert!(require_placements(&imported, &tokens, &empty).is_err());
        let aligned = [
            VisualPlacement {
                rotary_position: None,
                position: 1,
                embedding: vec![0.0; 4],
            },
            VisualPlacement {
                rotary_position: None,
                position: 2,
                embedding: vec![0.0; 4],
            },
        ];
        require_placements(&imported, &tokens, &aligned)?;
        Ok(())
    }

    #[test]
    fn greedy_argmax_is_strict_and_finite() -> Result<()> {
        assert_eq!(argmax(&[0.5, 2.0, 1.0])?, 1);
        assert_eq!(argmax(&[-3.0, -2.0])?, 1);
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[f32::NAN, 1.0]).is_err());
        Ok(())
    }
}
