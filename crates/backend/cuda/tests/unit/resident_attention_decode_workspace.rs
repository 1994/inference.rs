use super::partitions;
#[test]
fn split_plan_obeys_parallelism_and_short_window_limits() {
    assert_eq!(partitions(16, 512, 170), 1);
    assert_eq!(partitions(16, 4096, 8), 1);
    assert_eq!(partitions(16, 4096, 64), 4);
    assert_eq!(partitions(16, 4096, 170), 8);
    assert_eq!(partitions(8, 16384, 170), 16);
}
