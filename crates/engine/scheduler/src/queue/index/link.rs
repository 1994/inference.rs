//! Compact optional node links preserve the checkpoint's zero-based nullable indices.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{borrow::Borrow, num::NonZeroUsize};

pub(super) fn serialize<S: Serializer>(
    link: impl Borrow<Option<NonZeroUsize>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    link.borrow()
        .map(|slot| slot.get() - 1)
        .serialize(serializer)
}
pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NonZeroUsize>, D::Error> {
    Option::<usize>::deserialize(deserializer)?
        .map(|slot| {
            slot.checked_add(1)
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| serde::de::Error::custom("ordered node link overflow"))
        })
        .transpose()
}
