use super::partitions;
#[test]
fn partition_tree_obeys_hardware_and_window_without_storage_capacity() {
    assert_eq!(partitions(16, Some(512), 170), 1);
    assert_eq!(partitions(16, None, 8), 1);
    assert_eq!(partitions(16, None, 64), 4);
    assert_eq!(partitions(16, None, 170), 8);
    assert_eq!(partitions(8, None, 170), 16);
    assert_eq!(partitions(24, None, 170), 8);
}
