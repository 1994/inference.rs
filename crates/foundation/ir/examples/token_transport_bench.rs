//! Isolated token handoff measurement; this is not engine throughput or GPU performance.
use infer_core::Result;
use infer_ir::{ExecutionInput, TokenBuffer};
use std::{hint::black_box, time::Instant};

fn main() -> Result<()> {
    let repeats = 50_000u32;
    println!(
        "context_tokens,iterations,prefix_copy_ns,shared_clone_ns,incremental_decode_ns,old_payload_bytes,new_payload_bytes"
    );
    for count in [512, 8192, 65536] {
        let tokens: TokenBuffer = vec![7; count].into();
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(black_box(tokens.as_slice()).to_vec());
        }
        let copy = start.elapsed().as_nanos() / u128::from(repeats);
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(black_box(&tokens).clone());
        }
        let shared = start.elapsed().as_nanos() / u128::from(repeats);
        let start = Instant::now();
        for _ in 0..repeats {
            let input = black_box(ExecutionInput::Decode {
                position: count,
                token: 7,
            });
            black_box(input.delta(black_box(count))?);
        }
        let decode = start.elapsed().as_nanos() / u128::from(repeats);
        println!("{count},{repeats},{copy},{shared},{decode},{},4", count * 4);
    }
    Ok(())
}
