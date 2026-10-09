use super::*;
#[test]
fn checkpoint_kv_storage_is_declared_independently_of_weights() -> Result<()> {
    assert_eq!(declared_kv_dtype(&serde_json::Value::Null)?, None);
    let mut scheme = serde_json::json!({"num_bits":8,"type":"float","strategy":"tensor","dynamic":false,"symmetric":true});
    assert_eq!(declared_kv_dtype(&scheme)?, Some(TensorDtype::F8E4m3));
    scheme["dynamic"] = true.into();
    assert!(declared_kv_dtype(&scheme).is_err());
    scheme["dynamic"] = false.into();
    scheme["type"] = "int".into();
    assert!(declared_kv_dtype(&scheme).is_err());
    Ok(())
}
