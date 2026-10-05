use infer_scheduler::service_charge;

#[test]
fn service_charge_handles_zero_weight_and_saturates_extreme_costs() {
    assert_eq!(service_charge(1, 0), u64::MAX);
    assert_eq!(service_charge(u64::MAX, 1), u64::MAX);
    assert_eq!(service_charge(1, 3), 333_334);
    assert_eq!(service_charge(0, 1), 0);
}
