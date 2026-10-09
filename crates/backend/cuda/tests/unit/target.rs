use super::*;

#[test]
fn native_fp4_is_a_capability_not_a_storage_policy() {
    assert!(CudaTarget::BlackwellSm120.supports_compute(DType::Fp4E2M1));
    assert!(!CudaTarget::HopperSm90.supports_compute(DType::Fp4E2M1));
    assert!(!CudaTarget::Other("sm_999".into()).supports_compute(DType::Fp4E2M1));
    assert!(CudaTarget::HopperSm90.supports_compute(DType::Fp8E4M3));
    assert!(CudaTarget::Other("sm_999".into()).supports_compute(DType::Bf16));
}

#[test]
fn production_and_test_devices_do_not_share_fp4_capability() {
    assert!(
        CudaTarget::from_sm_name("sm_90")
            .require_native_nvfp4()
            .is_err()
    );
    assert!(
        CudaTarget::from_sm_name("sm_120")
            .require_native_nvfp4()
            .is_ok()
    );
    assert!(
        CudaTarget::from_sm_name("sm_999")
            .require_native_nvfp4()
            .is_err()
    );
}
