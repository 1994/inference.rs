use super::*;

#[test]
fn sizes_stay_readable() {
    assert_eq!(human_bytes(0), "0 MiB");
    assert_eq!(
        human_bytes(crate::constants::GIB_U64 + crate::constants::GIB_U64 / 2),
        "1.5 GiB"
    );
}
