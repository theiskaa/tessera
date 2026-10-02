use super::*;

#[test]
fn no_schema_keeps_the_fixed_historical_1900_support_contract() {
    assert_eq!(supports(&json!({}), &json!({})).unwrap(), [9817, 1900, 600]);
    assert_eq!(
        supports(
            &json!({"model_gold_support":{"real":{"org":1904}}}),
            &json!({})
        )
        .unwrap(),
        [9817, 1900, 600]
    );
}

#[test]
fn null_and_unknown_versions_are_refused_without_metadata_dependencies() {
    for schema in [
        Value::Null,
        json!("training_seen_evaluation_v4_1905_v1"),
        json!(4),
    ] {
        let error = supports(&json!({"schema":schema}), &json!({})).unwrap_err();
        assert_eq!(error.to_string(), "unsupported training-seen plan schema");
    }
}
