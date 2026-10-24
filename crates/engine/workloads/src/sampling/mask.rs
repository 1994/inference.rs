//! Borrowed vocabulary bitset shared by grammar adapters and the host sampler.
use infer_core::{Error, Result};

/// Number of token bits carried by one grammar mask word.
const MASK_WORD_BITS: usize = 32;

/// Legal next tokens for one generation prefix. Bit `token % 32` of word `token / 32`
/// indicates whether that token may be selected. EOS has no special exemption.
///
/// The backing words belong to the caller. Construct a new view after advancing a matcher;
/// sharing this immutable view does not share or mutate its grammar state. Padding bits beyond
/// the vocabulary are ignored, allowing adapters to initialize all words to `u32::MAX`.
#[derive(Debug, Clone, Copy)]
pub struct TokenMask<'a> {
    words: &'a [u32],
    vocabulary: usize,
}

impl<'a> TokenMask<'a> {
    /// Validate the bitset shape without allocating or copying its words.
    /// # Errors
    /// Rejects an empty vocabulary, token IDs outside the u32 ABI, and a word count that does not
    /// exactly match the vocabulary. A correctly sized all-zero mask is valid here; sampling it
    /// fails because the current prefix has no legal continuation.
    pub fn new(words: &'a [u32], vocabulary: usize) -> Result<Self> {
        if vocabulary == 0 || u32::try_from(vocabulary - 1).is_err() {
            return Err(Error::invalid("invalid token mask vocabulary"));
        }
        if words.len() != vocabulary.div_ceil(MASK_WORD_BITS) {
            return Err(Error::invalid(
                "token mask word count does not match vocabulary",
            ));
        }
        Ok(Self { words, vocabulary })
    }

    #[must_use]
    pub const fn vocabulary(self) -> usize {
        self.vocabulary
    }

    /// Out-of-vocabulary IDs and padding bits never authorize a token.
    #[must_use]
    pub const fn allows(self, token: usize) -> bool {
        token < self.vocabulary
            && self.words[token / MASK_WORD_BITS] & (1_u32 << (token % MASK_WORD_BITS)) != 0
    }
}
