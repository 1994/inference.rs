use infer_core::Result;
use infer_ir::{ExecutionInput, OutputReadout, TokenBuffer};

#[test]
fn pair_input_clones_share_tokens_and_document_headers_without_changing_json() -> Result<()> {
    let wire = serde_json::json!({"Pairs": {"query": [1, 2], "documents": [[3, 4], [5]]}});
    let input: infer_ir::RequestInput = serde_json::from_value(wire.clone())
        .map_err(|e| infer_core::Error::invalid(e.to_string()))?;
    let cloned = input.clone();
    let (
        infer_ir::RequestInput::Pairs { query, documents },
        infer_ir::RequestInput::Pairs {
            query: other_query,
            documents: other_documents,
        },
    ) = (&input, &cloned)
    else {
        return Err(infer_core::Error::invariant("pair input changed variant"));
    };
    assert!(query.shares_storage(other_query));
    assert!(documents.shares_storage(other_documents));
    assert!(
        documents
            .iter()
            .zip(other_documents)
            .all(|(a, b)| a.shares_storage(b))
    );
    drop(input);
    assert_eq!(
        serde_json::to_value(cloned).map_err(|e| infer_core::Error::invalid(e.to_string()))?,
        wire
    );
    Ok(())
}
#[test]
fn spans_share_payload_and_decode_borrows_only_the_new_token() -> Result<()> {
    let prompt: TokenBuffer = (0..65536).collect();
    let clone = prompt.clone();
    assert!(prompt.shares_storage(&clone));
    let span = prompt.span(100..200)?;
    assert_eq!(span.as_ptr(), prompt[100..].as_ptr());
    let input = ExecutionInput::Prefill {
        span,
        readout: OutputReadout::None,
    };
    assert_eq!(input.delta(100)?.len(), 100);
    assert!(input.delta(99).is_err());
    let decode = ExecutionInput::Decode {
        position: 65536,
        token: 7,
    };
    assert_eq!(size_of_val(decode.delta(65536)?), 4);
    assert_eq!(decode.validate(&prompt, 65537, 65536)?, [7]);
    assert!(decode.validate(&prompt[..65535], 65537, 65536).is_err());
    Ok(())
}
#[test]
fn shared_prompt_is_immutable_when_generation_appends() {
    let prompt: TokenBuffer = vec![1, 2, 3].into();
    let mut context = prompt.clone();
    context.push(4);
    assert_eq!(prompt.as_slice(), [1, 2, 3]);
    assert_eq!(context.as_slice(), [1, 2, 3, 4]);
    assert!(!context.shares_storage(&prompt));
    assert!(context.span(4..5).is_err());
}

#[test]
fn generated_suffix_publishes_intermediate_page_boundaries() -> Result<()> {
    let generated: TokenBuffer = vec![7, 8, 9, 10].into();
    let input = ExecutionInput::Prefill {
        span: generated.span_at(0..4, 3)?,
        readout: OutputReadout::None,
    };
    assert_eq!(input.delta(3)?.as_ptr(), generated.as_ptr());
    assert_eq!(input.prefix(&[1, 2, 3], 5)?.as_ref(), [1, 2, 3, 7, 8]);
    assert_eq!(
        input.prefix(&[1, 2, 3], 7)?.as_ref(),
        [1, 2, 3, 7, 8, 9, 10]
    );
    assert_eq!(input.prefix(&[1, 2, 3], 2)?.as_ref(), [1, 2]);
    assert!(input.prefix(&[1, 2, 3], 8).is_err());
    assert!(generated.span_at(0..4, usize::MAX).is_err());
    Ok(())
}
#[test]
fn shared_results_keep_byte_credit_and_wire_shape_until_last_reader() -> Result<()> {
    let pool = infer_core::credits::CreditPool::new(1, 16)?;
    let mut output = infer_ir::SharedOutput::from(vec![1.0f32, 2.0]);
    output.attach_credit(pool.reserve(8)?);
    let other = output.clone();
    assert!(other.shares_storage(&output));
    assert_eq!(
        serde_json::to_value(&other).map_err(|e| infer_core::Error::invalid(e.to_string()))?,
        serde_json::json!([1.0, 2.0])
    );
    drop(output);
    assert_eq!(pool.used(), 8);
    assert!(pool.reserve(1).is_err());
    drop(other);
    assert_eq!(pool.used(), 0);
    assert!(pool.reserve(16).is_ok());
    Ok(())
}
