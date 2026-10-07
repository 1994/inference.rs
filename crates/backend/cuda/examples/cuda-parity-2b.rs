//! End-to-end parity against the official Qwen3-VL implementation on a small package.
//!
//! Consumes the dump produced by `tools/vision/qwen3vl_2b_parity_reference.py` and reproduces the
//! same step with this repository's pipeline: the provider imports the package, the real processor
//! and tokenizer assemble the prompt, the vision tower encodes the image, the encodings are placed
//! at the media tokens and the resident program prefills and decodes.
//!
//! Both sides run with deepstack disabled, so this measures the shared path rather than the
//! released checkpoint's feature set.
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use infer_backend_cuda::{loading::LoadedModel, vision};
    use infer_core::{Error, ModelId};
    use infer_ir::Modality;
    use infer_models::{TextAssets, place_visual_embeddings, prepare};

    let (root, reference_path) = paths()?;
    cutile::jit_cache::enable_default()?;
    let loaded = LoadedModel::open(device_of(&root)?, &root, ModelId::ONE, load_options())?;
    let imported = loaded.imported().clone();
    let vision = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("package declares no vision encoder"))?;
    let hidden = loaded.model().hidden_size;

    let reference = load_reference(&reference_path)?;
    let assets = TextAssets::open(&root, 4096)?;
    let processor = image_processor(&root)?;
    let prepared = prepare(
        &assets,
        &processor,
        &imported,
        &reference.text,
        std::slice::from_ref(&reference.image),
        &Modality::Image,
    )?;

    // 1. Token stream: expansion must reproduce the reference ids exactly.
    let tokens_match = prepared.tokens == reference.ids;

    // 2. Encoded media and 3. final prompt logits.
    let official_pixels = std::env::var_os("INFER_PARITY_OFFICIAL_PIXELS").is_some();
    let official = infer_models::PromptImage {
        pixels: reference.pixels,
        grid: reference.grid,
    };
    let image = if official_pixels {
        &official
    } else {
        &prepared.images[0]
    };
    let encoded = vision.encode(loaded.device(), image)?;
    let (vision_abs, vision_rel) = compare_media(&encoded, &reference.vision);
    let placements =
        place_visual_embeddings(&prepared, &imported, &Modality::Image, &[encoded], hidden)?;
    let mut program = loaded.sequence(4096)?;
    let logits = vision::prefill(
        &mut program,
        &imported,
        &prepared.tokens,
        &placements,
        hidden,
        0,
    )?;
    let (max_abs, theirs) = compare_logits(&logits, &reference.logits);
    let mine = vision::argmax(&logits)?;

    // 4. Greedy continuation.
    let generated = vision::generate(
        &mut program,
        &imported,
        &prepared.tokens,
        &placements,
        hidden,
        reference.generated.len(),
        &[],
    )?;
    let matching = generated
        .iter()
        .zip(reference.generated.iter())
        .take_while(|(left, right)| left == right)
        .count();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": tokens_match && mine == theirs && matching == reference.generated.len()
                && vision_rel.is_finite() && vision_rel <= VISION_TOLERANCE,
            "package": root,
            "official_pixels": official_pixels,
            "reference": reference_path,
            "prompt_tokens": prepared.tokens.len(),
            "reference_prompt_tokens": reference.ids.len(),
            "tokens_match": tokens_match,
            "media_tokens": placements.len(),
            "first_token": mine,
            "reference_first_token": theirs,
            "vision_max_abs": vision_abs,
            "vision_max_rel": vision_rel,
            "logits_max_abs": max_abs,
            "generated": generated,
            "reference_generated": reference.generated,
            "matching_prefix": matching,
            "scope": "deepstack disabled on both sides: preprocess -> tower -> merger -> merge -> decode",
        }))?
    );
    if !tokens_match {
        return Err(Error::invariant("prompt token streams disagree").into());
    }
    if mine != theirs || matching != reference.generated.len() {
        return Err(Error::invariant("greedy continuations disagree").into());
    }
    if !vision_rel.is_finite() || vision_rel > VISION_TOLERANCE {
        return Err(Error::invariant("vision embeddings exceed the parity tolerance").into());
    }
    Ok(())
}

/// Match the per-stage vision golden gate; never accept a matching first token alone.
#[cfg(target_os = "linux")]
const VISION_TOLERANCE: f32 = 1e-2;

/// Open the device that runs the parity check.
#[cfg(target_os = "linux")]
fn device_of(
    _root: &str,
) -> Result<infer_backend_cuda::device::CudaDevice, Box<dyn std::error::Error>> {
    Ok(infer_backend_cuda::device::CudaDevice::new(0)?)
}

/// Loading options for a parity run: autotuning would add measurement noise to the comparison.
#[cfg(target_os = "linux")]
fn load_options() -> infer_backend_cuda::loading::LoadOptions {
    infer_backend_cuda::loading::LoadOptions {
        autotune: false,
        ..infer_backend_cuda::loading::LoadOptions::default()
    }
}

/// Decode the reference `(temporal, height, width)` patch grid.
#[cfg(target_os = "linux")]
fn decode_grid(bytes: &[u8]) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    use infer_core::Error;
    let values: Vec<i64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    let axis = |index: usize| -> Result<usize, Box<dyn std::error::Error>> {
        usize::try_from(values[index]).map_err(|_| Error::invalid("reference patch grid").into())
    };
    Ok((axis(0)?, axis(1)?, axis(2)?))
}

/// Absolute and reference-relative difference of the encoded media.
#[cfg(target_os = "linux")]
fn compare_media(mine: &[f32], theirs: &[f32]) -> (f32, f32) {
    if mine.len() != theirs.len() || mine.is_empty() {
        return (f32::INFINITY, f32::INFINITY);
    }
    let mut max_abs = 0f32;
    for (left, right) in mine.iter().zip(theirs.iter()) {
        if !left.is_finite() || !right.is_finite() {
            return (f32::INFINITY, f32::INFINITY);
        }
        max_abs = max_abs.max((left - right).abs());
    }
    let scale = theirs
        .iter()
        .fold(0f32, |scale, value| scale.max(value.abs()));
    (
        max_abs,
        if scale > 0.0 {
            max_abs / scale
        } else {
            max_abs
        },
    )
}

/// Largest absolute logits difference and the reference's greedy token.
#[cfg(target_os = "linux")]
fn compare_logits(mine: &[f32], theirs: &[f32]) -> (f32, u32) {
    let mut max_abs = 0f32;
    let mut best = 0usize;
    for (index, value) in theirs.iter().enumerate() {
        if *value > theirs[best] {
            best = index;
        }
    }
    for (left, right) in mine.iter().zip(theirs.iter()) {
        max_abs = max_abs.max((left - right).abs());
    }
    (max_abs, u32::try_from(best).unwrap_or(u32::MAX))
}

/// Everything the reference run dumped for one image and prompt.
#[cfg(target_os = "linux")]
struct Reference {
    image: infer_models::RawImage,
    text: String,
    ids: Vec<u32>,
    logits: Vec<f32>,
    generated: Vec<u32>,
    vision: Vec<f32>,
    pixels: Vec<f32>,
    grid: (usize, usize, usize),
}

/// Read the reference dump written by the Python reference runner.
#[cfg(target_os = "linux")]
fn load_reference(path: &str) -> Result<Reference, Box<dyn std::error::Error>> {
    use infer_core::Error;
    use infer_models::SafetensorsFile;

    let decode = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| i64::from_le_bytes(*chunk))
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
            .collect()
    };
    let mut file = SafetensorsFile::open(path)?;
    let image_bytes = file.read_bytes("image", 1 << 28)?;
    let shape = file.read_bytes("image_shape", 1 << 12)?;
    let text_bytes = file.read_bytes("text", 1 << 20)?;
    let ids = decode(&file.read_bytes("input_ids", 1 << 20)?);
    let logits = file.read_f32("logits", 1 << 30)?.data;
    let generated = decode(&file.read_bytes("generated", 1 << 16)?);
    let vision = file.read_f32("vision_embeddings", 1 << 30)?.data;
    let pixels = file.read_f32("pixel_values", 1 << 30)?.data;
    let grid = decode_grid(&file.read_bytes("grid_thw", 1 << 12)?)?;
    let dimensions: Vec<i64> = shape
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    let text = String::from_utf8(text_bytes)
        .map_err(|error| Error::invalid(format!("reference prompt text: {error}")))?;
    Ok(Reference {
        image: infer_models::RawImage {
            height: usize::try_from(dimensions[0])
                .map_err(|_| Error::invalid("reference height"))?,
            width: usize::try_from(dimensions[1]).map_err(|_| Error::invalid("reference width"))?,
            pixels: image_bytes,
        },
        text,
        ids,
        logits,
        generated,
        vision,
        pixels,
        grid,
    })
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CUDA parity checks require Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn image_processor(root: &str) -> Result<infer_models::ImageProcessor, Box<dyn std::error::Error>> {
    Ok(infer_models::ImageProcessor::from_json(&std::fs::read(
        std::path::Path::new(root).join("preprocessor_config.json"),
    )?)?)
}

#[cfg(target_os = "linux")]
fn paths() -> Result<(String, String), Box<dyn std::error::Error>> {
    use infer_core::Error;
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .ok_or_else(|| Error::invalid("model package path required"))?;
    let reference_path = args
        .next()
        .ok_or_else(|| Error::invalid("reference safetensors path required"))?;
    Ok((root, reference_path))
}
