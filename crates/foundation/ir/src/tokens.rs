//! Shared CPU token ownership and incremental phase inputs. Cloning never copies token payload.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, ops::Deref, ops::Range, sync::Arc};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenBuffer(
    Arc<Vec<u32>>,
    #[serde(skip)] Option<infer_core::credits::CreditLease>,
    #[serde(skip)] Option<infer_core::credits::CreditLease>,
);
impl From<Vec<u32>> for TokenBuffer {
    fn from(tokens: Vec<u32>) -> Self {
        Self(Arc::new(tokens), None, None)
    }
}
impl FromIterator<u32> for TokenBuffer {
    fn from_iter<T: IntoIterator<Item = u32>>(tokens: T) -> Self {
        tokens.into_iter().collect::<Vec<_>>().into()
    }
}
impl Deref for TokenBuffer {
    type Target = Vec<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl TokenBuffer {
    pub fn attach_byte_credit(&mut self, credit: infer_core::credits::CreditLease) {
        self.2 = Some(credit);
    }
    pub fn attach_credit(&mut self, credit: infer_core::credits::CreditLease) {
        self.1 = Some(credit);
    }
    /// Allocate bounded mutable generation storage on a preparation worker.
    /// # Errors
    /// Returns capacity failure instead of silently exceeding the host budget.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
        Ok(tokens.into())
    }
    /// # Errors
    /// Rejects invalid local/global ranges or position overflow.
    pub fn span_at(&self, range: Range<usize>, offset: usize) -> Result<TokenSpan> {
        let mut span = self.span(range)?;
        span.range = span
            .range
            .start
            .checked_add(offset)
            .zip(span.range.end.checked_add(offset))
            .map(|(start, end)| start..end)
            .ok_or_else(|| Error::invalid("token position overflow"))?;
        span.offset = offset;
        Ok(span)
    }
    pub fn push(&mut self, token: u32) {
        Arc::make_mut(&mut self.0).push(token);
    }
    #[must_use]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// # Errors
    /// Rejects an empty or out-of-bounds span without copying its payload.
    pub fn span(&self, range: Range<usize>) -> Result<TokenSpan> {
        if range.is_empty() || self.get(range.clone()).is_none() {
            return Err(Error::invalid("invalid token span"));
        }
        Ok(TokenSpan {
            storage: self.clone(),
            range,
            offset: 0,
        })
    }
}
#[derive(Debug, Clone)]
pub struct TokenSpan {
    storage: TokenBuffer,
    range: Range<usize>,
    offset: usize,
}
impl Deref for TokenSpan {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.storage[self.range.start - self.offset..self.range.end - self.offset]
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputReadout {
    None,
    Logits,
    Full,
}
/// Full-context input is a compatibility path for numerical diagnostics, not runtime dispatch.
#[derive(Debug, Clone)]
pub enum ExecutionInput {
    Full(TokenBuffer),
    Prefill {
        span: TokenSpan,
        readout: OutputReadout,
    },
    Decode {
        position: usize,
        token: u32,
    },
}
impl From<Vec<u32>> for ExecutionInput {
    fn from(tokens: Vec<u32>) -> Self {
        Self::Full(tokens.into())
    }
}
impl ExecutionInput {
    /// # Errors
    /// Materializes a full prefix only on the cold decode cache-publication path.
    pub fn prefix<'a>(&'a self, history: &'a [u32], end: usize) -> Result<Cow<'a, [u32]>> {
        match self {
            Self::Full(tokens) => tokens.get(..end).map(Cow::Borrowed),
            Self::Prefill { span, .. } if span.offset == 0 => {
                span.storage.get(..end).map(Cow::Borrowed)
            }
            Self::Prefill { .. } | Self::Decode { .. } if end <= self.len() => {
                let mut prefix = Vec::with_capacity(end);
                let history_end = end.min(history.len());
                prefix.extend_from_slice(&history[..history_end]);
                if end > history.len() {
                    prefix.extend_from_slice(&self.delta(history.len())?[..end - history.len()]);
                }
                return Ok(Cow::Owned(prefix));
            }
            Self::Prefill { .. } | Self::Decode { .. } => None,
        }
        .ok_or_else(|| Error::invalid("invalid cache publication prefix"))
    }
    /// Starting position of the owned delta. Full-context diagnostics use the driver history cursor.
    #[must_use]
    pub const fn computed_frontier(&self) -> usize {
        match self {
            Self::Full(_) => 0,
            Self::Prefill { span, .. } => span.range.start,
            Self::Decode { position, .. } => *position,
        }
    }
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Full(t) => t.len(),
            Self::Prefill { span, .. } => span.range.end,
            Self::Decode { position, .. } => position.saturating_add(1),
        }
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[must_use]
    pub const fn readout(&self) -> OutputReadout {
        match self {
            Self::Full(_) => OutputReadout::Full,
            Self::Prefill { readout, .. } => *readout,
            Self::Decode { .. } => OutputReadout::Logits,
        }
    }
    /// # Errors
    /// Rejects a stale cursor or an invalid range. Decode borrows its inline token.
    pub fn delta(&self, committed: usize) -> Result<&[u32]> {
        match self {
            Self::Full(tokens) => tokens
                .get(committed..)
                .ok_or_else(|| Error::invalid("token cursor exceeds input")),
            Self::Prefill { span, .. } if span.range.start == committed => Ok(span),
            Self::Decode { position, token } if *position == committed => {
                Ok(std::slice::from_ref(token))
            }
            _ => Err(Error::invalid("incremental token cursor mismatch")),
        }
    }
    /// # Errors
    /// Validates the compatibility prefix and the incremental cursor without scanning old history.
    pub fn validate<'a>(
        &'a self,
        history: &[u32],
        capacity: usize,
        vocab: usize,
    ) -> Result<&'a [u32]> {
        if self.len() > capacity || matches!(self, Self::Full(t) if !t.starts_with(history)) {
            return Err(Error::invalid("token input/state mismatch"));
        }
        let delta = self.delta(history.len())?;
        if delta.is_empty() || delta.iter().any(|t| *t as usize >= vocab) {
            return Err(Error::invalid("empty or out-of-vocabulary token delta"));
        }
        Ok(delta)
    }
    /// # Errors
    /// Advances CPU history only after the caller has accepted the submission.
    pub fn commit(&self, history: &mut Vec<u32>) -> Result<()> {
        history.extend_from_slice(self.delta(history.len())?);
        Ok(())
    }
}

impl PartialEq for TokenBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for TokenBuffer {}
