//! Named device-ABI constants shared by the submission descriptor and batch handles.

/// Bit offset of the generation field inside a packed `infer_core::BatchHandle`.
///
/// A handle stores its 32-bit slot index in the low half and its 32-bit generation in the high half.
pub const HANDLE_GENERATION_SHIFT: u32 = 32;
/// Device ABI execution-role code for `infer_ir::ExecutionRole::Mixed`.
///
/// Role codes follow enum declaration order, so the mixed phase is the fourth code after
/// `Prefill` (0), `Decode` (1) and `Forward` (2).
pub const ROLE_MIXED: u32 = 3;
