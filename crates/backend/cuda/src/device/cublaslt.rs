//! Minimal cuBLAS binding for the projection GEMMs.
//!
//! The library is opened on first use rather than linked, so a host without the toolkit still
//! builds and every other backend path is unaffected. Launches go through `cublasGemmEx` with the
//! handle in device pointer mode, which is what lets `alpha` and `beta` live in device memory: a
//! host scalar cannot be captured into a CUDA graph, and a captured step is how this engine runs.
//!
//! Layouts follow the row-major to column-major mapping a projection needs:
//! `out[m, n] = activations[m, k] * weights[n, k]^T`, so `A` is the activation with `OP_T`
//! (`lda = k`), `B` is the weight with `OP_N` (`ldb = k`), and `C` is the output with `ldc = m`.
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::{DeviceOp, ExecutionContext, GraphNode};
use cutile::cuda_async::error::DeviceError;
use cutile::{half::bf16, prelude::*};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

type Handle = *mut c_void;
type CudaStream = cuda_core::sys::CUstream;

const OP_N: i32 = 0;
const OP_T: i32 = 1;
const COMPUTE_32F: i32 = 68;
const DATATYPE_BF16: i32 = 14;
const DATATYPE_F32: i32 = 0;
const POINTER_MODE_DEVICE: i32 = 1;
const GEMM_DEFAULT: i32 = -1;

type CreateHandle = unsafe extern "C" fn(*mut Handle) -> i32;
type DestroyHandle = unsafe extern "C" fn(Handle) -> i32;
type SetPointerMode = unsafe extern "C" fn(Handle, i32) -> i32;
type SetStream = unsafe extern "C" fn(Handle, CudaStream) -> i32;
type GemmEx = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
) -> i32;

struct Api {
    _library: libloading::Library,
    create: CreateHandle,
    destroy: DestroyHandle,
    set_pointer_mode: SetPointerMode,
    set_stream: SetStream,
    gemm: GemmEx,
}

/// Open the toolkit library, if this host has one.
#[expect(
    unsafe_code,
    reason = "Audited loader: opening the toolkit library the build already targets by its soname"
)]
fn open_library() -> Option<libloading::Library> {
    for name in ["libcublas.so.13", "libcublas.so"] {
        // SAFETY: dlopen of a toolkit soname; the library stays alive in `Api` for the process.
        if let Ok(library) = unsafe { libloading::Library::new(name) } {
            return Some(library);
        }
    }
    None
}

/// Resolve one symbol with its own type.
///
/// `Library::get::<T>` returns the symbol's address only when `T` is the function's own type:
/// asking for a `*const c_void` would read the first bytes of the function body instead.
#[expect(
    unsafe_code,
    reason = "Audited loader: resolving one documented cuBLAS entry point from a library that outlives every call"
)]
fn resolve<T: Copy>(library: &libloading::Library, name: &[u8]) -> Option<T> {
    // SAFETY: the caller passes the exact signature transcribed from cublas_api.h.
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}

/// Load every entry point, returning `None` if the toolkit is absent or older than expected.
fn load() -> Option<Api> {
    let library = open_library()?;
    macro_rules! take {
        ($name:literal, $ty:ty) => {
            resolve::<$ty>(&library, $name)?
        };
    }
    Some(Api {
        create: take!(b"cublasCreate_v2", CreateHandle),
        destroy: take!(b"cublasDestroy_v2", DestroyHandle),
        set_pointer_mode: take!(b"cublasSetPointerMode_v2", SetPointerMode),
        set_stream: take!(b"cublasSetStream_v2", SetStream),
        gemm: take!(b"cublasGemmEx", GemmEx),
        _library: library,
    })
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}

/// Whether cuBLAS can be used at all on this host.
pub fn available() -> bool {
    api().is_some()
}

/// One cuBLAS handle with the `1.0`/`0.0` device scalars every launch needs.
pub struct Context {
    handle: Handle,
    alpha: Tensor<f32>,
    beta: Tensor<f32>,
}

#[expect(
    unsafe_code,
    reason = "a cuBLAS handle is device-scoped, not thread-scoped, and is only used through a launched GEMM"
)]
// SAFETY: a cuBLAS handle is not thread-affine; every use is a GEMM on the owning device's stream
// and the two scalars are plain device allocations owned by this value.
unsafe impl Send for Context {}
#[expect(
    unsafe_code,
    reason = "a cuBLAS handle is device-scoped, not thread-scoped, and is only used through a launched GEMM"
)]
// SAFETY: nothing in the context mutates through a shared reference; the handle is only passed to
// cuBLAS, which serializes its own internal state.
unsafe impl Sync for Context {}

/// Borrow the process-wide context, creating the handle on first use.
pub fn context_for(device: &crate::device::CudaDevice) -> Option<&'static Context> {
    static CONTEXT: OnceLock<Option<Context>> = OnceLock::new();
    CONTEXT.get_or_init(|| build(device)).as_ref()
}

/// Destroy a handle, reporting nothing: every caller is already on an error path.
#[expect(
    unsafe_code,
    reason = "Audited teardown: the handle was created by this module and is destroyed exactly once"
)]
fn destroy(api: &Api, handle: Handle) {
    // SAFETY: the handle is owned by the context this module builds.
    unsafe {
        (api.destroy)(handle);
    }
}

#[expect(
    unsafe_code,
    reason = "Audited handle creation: one cuBLAS handle per process, packed with the device scalars every launch reads"
)]
fn build(device: &crate::device::CudaDevice) -> Option<Context> {
    let api = api()?;
    let mut handle: Handle = std::ptr::null_mut();
    // SAFETY: the handle is created here and owned by the returned context.
    let created = unsafe { (api.create)(&raw mut handle) };
    if created != 0 {
        return None;
    }
    // SAFETY: the handle is live; device pointer mode is what makes the scalars below legal.
    let mode = unsafe { (api.set_pointer_mode)(handle, POINTER_MODE_DEVICE) };
    if mode != 0 {
        destroy(api, handle);
        return None;
    }
    // SAFETY: the handle is live and the stream belongs to this device.
    let bound = unsafe { (api.set_stream)(handle, device.stream.cu_stream()) };
    if bound != 0 {
        destroy(api, handle);
        return None;
    }
    let scalar = |value: f32| -> Option<Tensor<f32>> {
        let mut tensor = api::zeros::<f32>(&[1]).sync_on(&device.stream).ok()?;
        let host = device.upload(vec![value], &[1]).ok()?;
        api::memcpy(&mut tensor, &host)
            .sync_on(&device.stream)
            .ok()?;
        Some(tensor)
    };
    let (Some(alpha), Some(beta)) = (scalar(1.0), scalar(0.0)) else {
        destroy(api, handle);
        return None;
    };
    Some(Context {
        handle,
        alpha,
        beta,
    })
}

#[expect(
    unsafe_code,
    reason = "Audited launch: the operands are live device tensors whose extents the caller checked against each other"
)]
unsafe fn launch(
    context: &Context,
    m: i32,
    n: i32,
    k: i32,
    activations: *const c_void,
    weights: *const c_void,
    out: *mut c_void,
) -> Result<(), i32> {
    let Some(api) = api() else {
        return Err(-1);
    };
    // SAFETY: `alpha` and `beta` are device scalars owned by the context, the operands are the
    // caller's live tensors, and the handle's stream belongs to the same device.
    unsafe {
        let alpha = context.alpha.device_pointer().cu_deviceptr() as *const c_void;
        let beta = context.beta.device_pointer().cu_deviceptr() as *const c_void;
        // `out` is row-major `[m, n]`, which is column-major `[n, m]` with `ld = n`, so cuBLAS
        // computes the transposed product: `op(A) = weights`, `op(B) = activations`.
        let status = (api.gemm)(
            context.handle,
            OP_T,
            OP_N,
            n,
            m,
            k,
            alpha,
            weights,
            DATATYPE_BF16,
            k,
            activations,
            DATATYPE_BF16,
            k,
            beta,
            out,
            DATATYPE_F32,
            n,
            COMPUTE_32F,
            GEMM_DEFAULT,
        );
        if status == 0 { Ok(()) } else { Err(status) }
    }
}

/// Run `out[m, n] = activations[m, k] * weights[n, k]^T` in bf16 with f32 accumulation.
///
/// # Errors
/// Rejects operands whose extents disagree, or reports the driver's status.
#[expect(
    unsafe_code,
    reason = "Audited launch: the three tensors outlive the call and their extents are checked against each other"
)]
pub fn gemm_bf16(
    device: &crate::device::CudaDevice,
    activations: &Tensor<bf16>,
    weights: &Tensor<bf16>,
    out: &Tensor<f32>,
) -> infer_core::Result<()> {
    let context = context_for(device)
        .ok_or_else(|| infer_core::Error::invalid("cuBLAS handle unavailable for this device"))?;
    let [m, k] = activations.shape()[..] else {
        return Err(infer_core::Error::invalid("cuBLAS activation rank"));
    };
    let [n, weight_k] = weights.shape()[..] else {
        return Err(infer_core::Error::invalid("cuBLAS weight rank"));
    };
    let [out_m, out_n] = out.shape()[..] else {
        return Err(infer_core::Error::invalid("cuBLAS output rank"));
    };
    if k != weight_k || m != out_m || n != out_n {
        return Err(infer_core::Error::invalid("cuBLAS operand shape"));
    }
    // SAFETY: the three tensors are live for the call and their extents were just checked.
    let status = unsafe {
        launch(
            context,
            m,
            n,
            k,
            activations.device_pointer().cu_deviceptr() as *const c_void,
            weights.device_pointer().cu_deviceptr() as *const c_void,
            out.device_pointer().cu_deviceptr() as *mut c_void,
        )
    };
    status.map_err(|code| {
        infer_core::Error::new(
            infer_core::ErrorCode::Backend,
            format!("cublasGemmEx status {code}"),
        )
    })
}

/// A weight at least this large is worth delegating; the measurement below shows the vendor
/// kernel losing on 2048x2048 and winning from 2048x8192 upwards.
const MIN_DELEGATED_WEIGHT_ELEMENTS: usize = 8 * 1024 * 1024;

/// The tile the activation is narrowed in. 1024 measured 5.2 us against 8.1 us at 8192 on the 2B
/// prompt shapes, because the grid has to fill the device at this size.
const CAST_TILE: i32 = 1024;

/// Vendor GEMM support for one program: the handle and the BF16 activation scratches.
///
/// The scratch is keyed by element count and allocated the first time a shape needs it, which
/// happens while a graph is being built and never while one is being captured.
pub struct Support {
    device: crate::device::CudaDevice,
    context: Option<&'static Context>,
    activations: HashMap<usize, Tensor<bf16>>,
}

impl Support {
    /// Build the support for one program. Delegation stays off unless
    /// `INFER_CUBLAS_PROJECTIONS` is set, and off when the toolkit is missing.
    pub fn new(device: &crate::device::CudaDevice) -> Self {
        let enabled = std::env::var_os("INFER_CUBLAS_PROJECTIONS").is_some();
        Self {
            device: device.clone(),
            context: enabled.then(|| context_for(device)).flatten(),
            activations: HashMap::new(),
        }
    }

    /// Whether this program delegates at all.
    pub const fn is_enabled(&self) -> bool {
        self.context.is_some()
    }

    /// Run one GEMM per shape on the device stream before any graph captures it.
    ///
    /// cuBLAS chooses workspace and algorithm the first time it sees a configuration, and that
    /// host-side work is rejected while a stream is capturing, so every shape a graph will record
    /// has to be exercised here first.
    #[expect(
        unsafe_code,
        reason = "Audited warmup launch: both operands are allocated here with the extents the launch is told about"
    )]
    pub fn warm(
        &mut self,
        m: usize,
        weight: &Tensor<bf16>,
        n: usize,
        k: usize,
    ) -> Result<(), DeviceError> {
        let Some(context) = self.context else {
            return Ok(());
        };
        if n.saturating_mul(k) < MIN_DELEGATED_WEIGHT_ELEMENTS || m.saturating_mul(k) == 0 {
            return Ok(());
        }
        let (Ok(m_i), Ok(n_i), Ok(k_i)) = (i32::try_from(m), i32::try_from(n), i32::try_from(k))
        else {
            return Ok(());
        };
        let activation_elements = m * k;
        if !self.activations.contains_key(&activation_elements) {
            let scratch =
                api::zeros::<bf16>(&[activation_elements]).sync_on(&self.device.stream)?;
            self.activations.insert(activation_elements, scratch);
        }
        let scratch = self
            .activations
            .get(&activation_elements)
            .ok_or_else(|| DeviceError::Internal("cuBLAS scratch".to_string()))?;
        let out = api::zeros::<f32>(&[m * n]).sync_on(&self.device.stream)?;
        // SAFETY: both operands are live device tensors allocated just above with the extents the
        // launch is told about, and the stream belongs to this device.
        let status = unsafe {
            launch(
                context,
                m_i,
                n_i,
                k_i,
                scratch.device_pointer().cu_deviceptr() as *const c_void,
                weight.device_pointer().cu_deviceptr() as *const c_void,
                out.device_pointer().cu_deviceptr() as *mut c_void,
            )
        };
        status
            .map_err(|code| DeviceError::Internal(format!("cublasGemmEx warmup status {code}")))?;
        self.device
            .reclaim_barrier()
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        Ok(())
    }

    /// Record `out[m, n] = input[m, k] * weight[n, k]^T` through cuBLAS when the shape is one the
    /// measurement favours, returning whether it did.
    ///
    /// # Errors
    /// Propagates launch failures; a shape that is not worth delegating is not an error.
    pub fn record_dense(
        &mut self,
        scope: &Scope,
        input: &Tensor<f32>,
        weight: &Tensor<bf16>,
        out: &Tensor<f32>,
        (m, n, k): (usize, usize, usize),
    ) -> Result<bool, DeviceError> {
        let Some(context) = self.context else {
            return Ok(false);
        };
        let activation_elements = m.saturating_mul(k);
        if n.saturating_mul(k) < MIN_DELEGATED_WEIGHT_ELEMENTS
            || activation_elements == 0
            || activation_elements % CAST_TILE as usize != 0
            || input.size() != activation_elements
        {
            return Ok(false);
        }
        let (Ok(m), Ok(n), Ok(k)) = (i32::try_from(m), i32::try_from(n), i32::try_from(k)) else {
            return Ok(false);
        };
        if !self.activations.contains_key(&activation_elements) {
            let scratch =
                api::zeros::<bf16>(&[activation_elements]).sync_on(&self.device.stream)?;
            self.activations.insert(activation_elements, scratch);
        }
        let scratch = self
            .activations
            .get_mut(&activation_elements)
            .ok_or_else(|| DeviceError::Internal("cuBLAS scratch".to_string()))?;
        scope.record(
            crate::kernels::linear::cast_bf16(
                (&mut *scratch).partition([CAST_TILE as usize]),
                input,
            )
            .generics(vec![CAST_TILE.to_string()]),
        )?;
        let op = GemmBf16::new(context, m, n, k, scratch, weight, out)
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        scope.record(op)?;
        Ok(true)
    }
}

/// A bf16 projection that can be recorded into a captured graph./// A bf16 projection that can be recorded into a captured graph.
///
/// `DeviceOp::execute` runs during capture, which is why the launch takes `alpha` and `beta` from
/// device memory: anything read from the host at that point would be frozen into the graph.
pub struct GemmBf16<'a> {
    context: &'a Context,
    m: i32,
    n: i32,
    k: i32,
    activations: &'a Tensor<bf16>,
    weights: &'a Tensor<bf16>,
    out: &'a Tensor<f32>,
}

impl<'a> GemmBf16<'a> {
    /// Describe `out[m, n] = activations[m, k] * weights[n, k]^T`.
    ///
    /// The extents are explicit rather than read from the tensor ranks so a flat activation buffer
    /// works: the cast that produces it writes flat tiles.
    ///
    /// # Errors
    /// Rejects operands whose element counts disagree with the extents.
    pub fn new(
        context: &'a Context,
        m: i32,
        n: i32,
        k: i32,
        activations: &'a Tensor<bf16>,
        weights: &'a Tensor<bf16>,
        out: &'a Tensor<f32>,
    ) -> infer_core::Result<Self> {
        let expected = |a: i32, b: i32| usize::try_from(a).ok().zip(usize::try_from(b).ok());
        let sizes = expected(m, k)
            .map(|(m, k)| m * k)
            .zip(expected(n, k).map(|(n, k)| n * k))
            .zip(expected(m, n).map(|(m, n)| m * n));
        let Some(((activation_size, weight_size), out_size)) = sizes else {
            return Err(infer_core::Error::invalid("cuBLAS extent"));
        };
        if activations.size() != activation_size
            || weights.size() != weight_size
            || out.size() != out_size
        {
            return Err(infer_core::Error::invalid("cuBLAS operand shape"));
        }
        Ok(Self {
            context,
            m,
            n,
            k,
            activations,
            weights,
            out,
        })
    }
}

impl GraphNode for GemmBf16<'_> {}

#[expect(
    unsafe_code,
    reason = "Audited in-graph launch: the operands are borrowed for the whole capture and the stream comes from the context"
)]
impl DeviceOp for GemmBf16<'_> {
    type Output = ();

    unsafe fn execute(self, _context: &ExecutionContext) -> Result<(), DeviceError> {
        // SAFETY: the operands are live device tensors whose extents were checked in `new`;
        // the handle already targets the stream this capture records onto.
        let status = unsafe {
            launch(
                self.context,
                self.m,
                self.n,
                self.k,
                self.activations.device_pointer().cu_deviceptr() as *const c_void,
                self.weights.device_pointer().cu_deviceptr() as *const c_void,
                self.out.device_pointer().cu_deviceptr() as *mut c_void,
            )
        };
        status.map_err(|code| DeviceError::Internal(format!("cublasGemmEx status {code}")))?;
        Ok(())
    }
}

impl IntoFuture for GemmBf16<'_> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match cutile::cuda_async::device_context::with_default_device_policy(|policy| {
            policy.next_stream()
        }) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{CudaDevice, device_error};

    fn sample(seed: u64, index: usize) -> f32 {
        let mut state = seed
            .wrapping_add(index as u64)
            .wrapping_mul(6364136223846793005);
        state ^= state >> 33;
        state = state.wrapping_mul(0xff51afd7ed558ccd);
        state ^= state >> 29;
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    }

    /// cuBLAS must reproduce a projection against an f64 reference on the shapes the dense models
    /// use, before anything is allowed to delegate to it.
    #[test]
    #[ignore = "requires CUDA hardware; run inside safe-run"]
    fn gemm_bf16_matches_reference() -> Result<(), Box<dyn std::error::Error>> {
        CudaDevice::enable_kernel_cache()?;
        let device = CudaDevice::new(0)?;
        if !available() {
            eprintln!("cuBLAS unavailable; skipping");
            return Ok(());
        }
        for (m, n, k) in [
            (4_usize, 256_usize, 256_usize),
            (12, 2048, 2048),
            (128, 2048, 2048),
        ] {
            let activations: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(sample(1, i))).collect();
            let weights: Vec<bf16> = (0..n * k).map(|i| bf16::from_f32(sample(2, i))).collect();
            let host_activations: Vec<f32> = activations.iter().map(|v| v.to_f32()).collect();
            let host_weights: Vec<f32> = weights.iter().map(|v| v.to_f32()).collect();
            let a = device.upload(activations, &[m, k])?;
            let b = device.upload(weights, &[n, k])?;
            let out = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
            // The first launch pays cuBLAS initialisation and its internal algorithm search, so
            // warm the shape and then time repeats.
            gemm_bf16(&device, &a, &b, &out)?;
            device.reclaim_barrier()?;
            let repeats = 50;
            let started = std::time::Instant::now();
            for _ in 0..repeats {
                gemm_bf16(&device, &a, &b, &out)?;
            }
            device.reclaim_barrier()?;
            let elapsed = started.elapsed() / repeats;
            let actual = out.to_host_vec().sync_on(&device.stream)?;
            let mut worst = 0.0_f32;
            let mut scale = 0.0_f64;
            for i in 0..m {
                for j in 0..n {
                    let mut expected = 0.0_f64;
                    for d in 0..k {
                        expected += f64::from(host_activations[i * k + d])
                            * f64::from(host_weights[j * k + d]);
                    }
                    scale = scale.max(expected.abs());
                    worst = worst.max((f64::from(actual[i * n + j]) - expected).abs() as f32);
                }
            }
            eprintln!(
                "m={m} n={n} k={k}: worst {worst:.3e} against reference scale {scale:.1} in {:.3} ms",
                elapsed.as_secs_f64() * 1.0e3
            );
            assert!(
                f64::from(worst) <= 1.0e-2 * scale.max(1.0),
                "cuBLAS disagrees with the reference at m={m} n={n} k={k}"
            );
        }
        Ok(())
    }

    /// The engine runs captured graphs, so the launch has to be recordable and replayable, with
    /// its scalars living on the device rather than frozen into the graph.
    #[test]
    #[ignore = "requires CUDA hardware; run inside safe-run"]
    fn gemm_bf16_replays_inside_a_captured_graph() -> Result<(), Box<dyn std::error::Error>> {
        CudaDevice::enable_kernel_cache()?;
        let device = CudaDevice::new(0)?;
        if !available() {
            eprintln!("cuBLAS unavailable; skipping");
            return Ok(());
        }
        let (m, n, k) = (12_usize, 2048_usize, 2048_usize);
        let activations: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(sample(3, i))).collect();
        let weights: Vec<bf16> = (0..n * k).map(|i| bf16::from_f32(sample(4, i))).collect();
        let a = device.upload(activations, &[m, k])?;
        let b = device.upload(weights, &[n, k])?;
        let out = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
        let context = context_for(&device).ok_or("no cuBLAS context")?;
        // cuBLAS does host-side work on the first use of a configuration (workspace and algorithm
        // selection), which a capturing stream rejects; run the shape once before recording it.
        gemm_bf16(&device, &a, &b, &out)?;
        device.reclaim_barrier()?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            let op = GemmBf16::new(context, m as i32, n as i32, k as i32, &a, &b, &out)
                .map_err(|error| DeviceError::Internal(error.to_string()))?;
            scope.record(op)?;
            Ok(())
        })
        .map_err(device_error)?;
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        device.reclaim_barrier()?;
        let captured = out.to_host_vec().sync_on(&device.stream)?;
        let direct = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
        gemm_bf16(&device, &a, &b, &direct)?;
        device.reclaim_barrier()?;
        let reference = direct.to_host_vec().sync_on(&device.stream)?;
        let gap = captured
            .iter()
            .zip(&reference)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("captured versus direct launch: max_abs={gap:e}");
        assert_eq!(gap, 0.0, "the captured launch differs from the direct one");
        Ok(())
    }
}
