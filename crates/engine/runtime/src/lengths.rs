//! Resolved single-sequence length limits shared by every request entry point.
//!
//! `--max-model-len` is the total context of one sequence, prompt and output together. It is
//! resolved once, after model metadata is known and before any backend resource is planned, so
//! the HTTP adapters, the workload plan, admission and capacity planning read the same numbers.
//! The input cap and the total context stay separate: an input cap is not a `--max-model-len`.
use crate::RuntimeConfig;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Where an effective length limit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitSource {
    /// The limit follows the model's supported context ceiling.
    Model,
    /// The limit follows an explicit service setting.
    Service,
    /// The limit follows a built-in bounded default.
    Default,
}

/// Effective source of each limit in [`ResolvedLengthLimits`], for configuration readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LengthLimitSources {
    pub total: LimitSource,
    pub input_cap: LimitSource,
    pub output_cap: LimitSource,
}

/// Single-sequence length limits after the model ceiling, service overrides and defaults combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedLengthLimits {
    /// Model-supported context ceiling (`M`).
    pub model_limit: usize,
    /// Effective service total context, prompt and output together (`L`).
    pub total: usize,
    /// Effective accepted prompt-token cap.
    pub input_cap: usize,
    /// Effective generated-token cap.
    pub output_cap: usize,
    pub sources: LengthLimitSources,
}

impl ResolvedLengthLimits {
    /// Combine the model ceiling with the service settings exactly once.
    ///
    /// # Errors
    /// Rejects a zero or unattainable explicit limit instead of silently shrinking it.
    pub fn resolve(model_limit: usize, config: &RuntimeConfig) -> Result<Self> {
        if model_limit == 0 {
            return Err(Error::invalid("model context limit must be positive"));
        }
        let (total, total_source) = match config.max_model_len {
            None => (model_limit, LimitSource::Model),
            Some(0) => return Err(Error::invalid("max-model-len must be positive")),
            Some(requested) if requested > model_limit => {
                return Err(Error::invalid(format!(
                    "max-model-len {requested} exceeds the model context limit {model_limit}"
                )));
            }
            Some(requested) => (requested, LimitSource::Service),
        };
        // The input cap never exceeds the total context, so a prompt can always leave room for
        // at least one output token when the total is positive.
        let (input_cap, input_source) = if config.max_input_tokens <= total {
            (config.max_input_tokens, LimitSource::Service)
        } else {
            (total, LimitSource::Model)
        };
        if input_cap == 0 {
            return Err(Error::invalid("input token cap must be positive"));
        }
        let (output_cap, output_source) = match config.max_output_tokens {
            None => (total, LimitSource::Default),
            Some(0) => return Err(Error::invalid("max-output-tokens must be positive")),
            Some(requested) if requested > total => {
                return Err(Error::invalid(format!(
                    "max-output-tokens {requested} exceeds the total context {total}"
                )));
            }
            Some(requested) => (requested, LimitSource::Service),
        };
        Ok(Self {
            model_limit,
            total,
            input_cap,
            output_cap,
            sources: LengthLimitSources {
                total: total_source,
                input_cap: input_source,
                output_cap: output_source,
            },
        })
    }
    /// Check one request's encoded prompt against its output budget.
    ///
    /// # Errors
    /// Returns an invalid-input error when the request cannot fit, without truncating it.
    pub fn check(&self, prompt_tokens: usize, output_tokens: usize) -> Result<()> {
        if prompt_tokens == 0 {
            return Err(Error::invalid("encoded prompt must not be empty"));
        }
        if prompt_tokens > self.input_cap {
            return Err(Error::invalid(format!(
                "prompt of {prompt_tokens} tokens exceeds the input cap {}",
                self.input_cap
            )));
        }
        if output_tokens > self.output_cap {
            return Err(Error::invalid(format!(
                "requested output of {output_tokens} tokens exceeds the output cap {}",
                self.output_cap
            )));
        }
        let requested = prompt_tokens
            .checked_add(output_tokens)
            .ok_or_else(|| Error::invalid("prompt + output token count overflow"))?;
        if requested > self.total {
            return Err(Error::invalid(format!(
                "prompt {prompt_tokens} + output {output_tokens} exceeds the total context {}",
                self.total
            )));
        }
        Ok(())
    }
    /// Generated-token budget left for a prompt, after the output cap and the total context.
    ///
    /// # Errors
    /// Returns an invalid-input error when the prompt leaves no room for output.
    pub fn remaining_output(&self, prompt_tokens: usize) -> Result<usize> {
        let remaining = self.total.saturating_sub(prompt_tokens);
        if remaining == 0 {
            return Err(Error::invalid(format!(
                "prompt of {prompt_tokens} tokens leaves no room in the {} token total context",
                self.total
            )));
        }
        Ok(remaining.min(self.output_cap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(
        max_model_len: Option<usize>,
        max_output_tokens: Option<usize>,
        max_input_tokens: usize,
    ) -> RuntimeConfig {
        RuntimeConfig {
            max_model_len,
            max_output_tokens,
            max_input_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn an_unset_deployment_limit_follows_the_model_ceiling() {
        let limits = ResolvedLengthLimits::resolve(4096, &config(None, None, 65536)).unwrap();
        assert_eq!(limits.model_limit, 4096);
        assert_eq!(limits.total, 4096);
        assert_eq!(limits.input_cap, 4096);
        assert_eq!(limits.output_cap, 4096);
        assert_eq!(limits.sources.total, LimitSource::Model);
        assert_eq!(limits.sources.input_cap, LimitSource::Model);
        assert_eq!(limits.sources.output_cap, LimitSource::Default);
    }

    #[test]
    fn an_explicit_total_context_shrinks_the_input_and_output_caps() {
        let limits =
            ResolvedLengthLimits::resolve(262_144, &config(Some(8192), Some(1024), 65536)).unwrap();
        assert_eq!(limits.total, 8192);
        assert_eq!(limits.input_cap, 8192);
        assert_eq!(limits.output_cap, 1024);
        assert_eq!(limits.sources.total, LimitSource::Service);
        assert_eq!(limits.sources.output_cap, LimitSource::Service);
    }

    #[test]
    fn an_input_cap_below_the_total_context_is_reported_as_the_service_cap() {
        let limits = ResolvedLengthLimits::resolve(262_144, &config(None, None, 4096)).unwrap();
        assert_eq!(limits.total, 262_144);
        assert_eq!(limits.input_cap, 4096);
        assert_eq!(limits.sources.input_cap, LimitSource::Service);
    }

    #[test]
    fn unattainable_or_zero_explicit_limits_are_rejected_not_shrunk() {
        for config in [
            config(Some(4097), None, 65536),
            config(Some(0), None, 65536),
            config(None, Some(0), 65536),
            RuntimeConfig {
                max_model_len: Some(64),
                max_output_tokens: Some(128),
                ..Default::default()
            },
        ] {
            assert!(
                ResolvedLengthLimits::resolve(4096, &config).is_err(),
                "{config:?}"
            );
        }
        assert!(ResolvedLengthLimits::resolve(0, &config(None, None, 65536)).is_err());
    }

    #[test]
    fn a_request_fits_only_when_prompt_plus_output_stays_inside_the_total() {
        let limits = ResolvedLengthLimits::resolve(64, &config(Some(8), Some(4), 8)).unwrap();
        assert!(limits.check(4, 4).is_ok());
        assert!(limits.check(5, 4).is_err());
        assert!(limits.check(4, 5).is_err());
        assert!(limits.check(0, 1).is_err());
        // Checked arithmetic stays defensive even though both caps already bound the sum.
        assert!(limits.check(usize::MAX, 1).is_err());
    }

    #[test]
    fn the_remaining_output_budget_respects_the_cap_and_the_total() {
        let limits = ResolvedLengthLimits::resolve(64, &config(Some(16), Some(4), 16)).unwrap();
        assert_eq!(limits.remaining_output(1).unwrap(), 4);
        assert_eq!(limits.remaining_output(14).unwrap(), 2);
        assert!(limits.remaining_output(16).is_err());
    }
}
