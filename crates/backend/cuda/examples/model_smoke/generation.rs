use infer_core::{Error, Result};
use std::{io::Read, path::Path};

pub fn eos_tokens(root: &Path, configured: Option<u32>) -> Result<Vec<u32>> {
    let mut tokens: Vec<u32> = configured.into_iter().collect();
    let path = infer_models::package_path(root, "generation_config.json")?;
    if path.exists() {
        let file = std::fs::File::open(path).map_err(|e| Error::invalid(e.to_string()))?;
        let mut bytes = Vec::new();
        file.take(1_048_577)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::invalid(e.to_string()))?;
        if bytes.len() > 1_048_576 {
            return Err(Error::invalid("generation config budget exceeded"));
        }
        let config: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| Error::invalid(e.to_string()))?;
        if let Some(value) = config.get("eos_token_id") {
            let values = value
                .as_array()
                .map_or_else(|| vec![value], |array| array.iter().collect());
            for value in values {
                let token = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| Error::invalid("invalid generation EOS token"))?;
                tokens.push(token);
            }
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    Ok(tokens)
}
