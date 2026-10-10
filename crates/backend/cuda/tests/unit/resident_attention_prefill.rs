use super::kernels;
use crate::device::CudaDevice;
use cutile::prelude::*;

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 32;
const CAPACITY: usize = 128;
const LANES: usize = 32;

fn data(size: usize, phase: u16) -> Vec<f32> {
    (0..size)
        .map(|i| {
            (f32::from((u16::try_from(i % 97).expect("bounded") * 7 + phase) % 31) - 15.0) / 16.0
        })
        .collect()
}

fn reference(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    head: usize,
    position: usize,
    window: usize,
) -> Vec<f64> {
    let start = if window == 0 {
        0
    } else {
        (position + 1).saturating_sub(window)
    };
    let offset = head / (HEADS / KV_HEADS) * CAPACITY * DIM;
    let mut scores: Vec<f64> = (start..=position)
        .map(|p| {
            query
                .iter()
                .zip(&keys[offset + p * DIM..offset + (p + 1) * DIM])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                / f64::from(u32::try_from(DIM).expect("dimension")).sqrt()
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for score in &mut scores {
        *score = (*score - maximum).exp();
    }
    let total: f64 = scores.iter().sum();
    (0..DIM)
        .map(|d| {
            scores
                .iter()
                .enumerate()
                .map(|(i, s)| s * f64::from(values[offset + (start + i) * DIM + d]))
                .sum::<f64>()
                / total
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA; run inside safe-run with --test-threads 1"]
fn chunk_attention_matches_reference_with_tails_windows_and_draft_offset()
-> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (start, count, offset, window) in [
        (0, 32, 0, 0),
        (3, 17, 0, 7),
        (31, 32, 0, 0),
        (0, 17, -1, 0),
        (32, 1, -1, 7),
        (3, 117, 0, 0),
        (1, 117, -1, 7),
    ] {
        check_case(&device, (start, count, offset, window))?;
    }
    Ok(())
}

fn check_case(
    device: &CudaDevice,
    case: (i32, i32, i32, i32),
) -> Result<(), Box<dyn std::error::Error>> {
    let (start, count, offset, window) = case;
    let lanes = if count > 32 { 128 } else { LANES };
    let query = data(lanes * HEADS * DIM, 1);
    let key = data(lanes * KV_HEADS * DIM, 3);
    let value = data(lanes * KV_HEADS * DIM, 11);
    let mut expected_keys = data(KV_HEADS * CAPACITY * DIM, 13);
    let mut expected_values = data(KV_HEADS * CAPACITY * DIM, 17);
    let mut keys = api::copy_host_vec_to_device(&Arc::new(expected_keys.clone()))
        .sync_on(&device.stream)?
        .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    let mut values = api::copy_host_vec_to_device(&Arc::new(expected_values.clone()))
        .sync_on(&device.stream)?
        .reshape(&[KV_HEADS, CAPACITY, DIM])?;
    let query_device = device.upload(query.clone(), &[lanes * HEADS, DIM])?;
    let key_device = device.upload(key.clone(), &[lanes, KV_HEADS, DIM])?;
    let value_device = device.upload(value.clone(), &[lanes, KV_HEADS, DIM])?;
    let metadata = device.upload(vec![start, count, offset, 0], &[4])?;
    kernels::append(
        (&mut keys).partition([1, CAPACITY, DIM]),
        (&mut values).partition([1, CAPACITY, DIM]),
        &key_device,
        &value_device,
        &metadata,
        1.0,
        1.0,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        CAPACITY.to_string(),
        "0".into(),
    ])
    .sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[lanes * HEADS, DIM]).sync_on(&device.stream)?;
    kernels::decode(
        (&mut output).partition([1, DIM]),
        &query_device,
        &keys,
        &values,
        &metadata,
        window,
        1.0,
        1.0,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        (HEADS / KV_HEADS).to_string(),
        HEADS.to_string(),
    ])
    .sync_on(&device.stream)?;
    for lane in 0..usize::try_from(count)? {
        let position = start + i32::try_from(lane)? + offset;
        if position < 0 {
            continue;
        }
        let position = usize::try_from(position)?;
        for head in 0..KV_HEADS {
            let src = (lane * KV_HEADS + head) * DIM;
            let dst = (head * CAPACITY + position) * DIM;
            expected_keys[dst..dst + DIM].copy_from_slice(&key[src..src + DIM]);
            expected_values[dst..dst + DIM].copy_from_slice(&value[src..src + DIM]);
        }
    }
    // Same inputs and same reference through the tensor-core twin. q and out are bound as
    // `[lanes, HEADS*DIM]` partitioned `[QT, DIM]` so the launch grid is (lanes/QT, HEADS) and
    // grid axis 1 is the head; binding them as `[lanes*HEADS, DIM]` collapses that axis, pid.1
    // becomes zero and every CTA reads KV head 0. Its compensated BF16 mma is held to the
    // relative tolerance the attention gate uses for the same product.
    let mut tiled = api::zeros::<f32>(&[lanes, HEADS * DIM]).sync_on(&device.stream)?;
    let query_tiled = query_device.view(&[lanes, HEADS * DIM])?;
    let scale = 1.0 / f32::from(u16::try_from(DIM).expect("dim fits in u16")).sqrt();
    kernels::decode_tiled(
        (&mut tiled).partition([LANES, DIM]),
        &query_tiled,
        &keys,
        &values,
        &metadata,
        window,
        1.0,
        1.0,
        scale,
    )
    .generics(vec![
        "f32".into(),
        DIM.to_string(),
        (HEADS / KV_HEADS).to_string(),
        LANES.to_string(),
        LANES.to_string(),
    ])
    .sync_on(&device.stream)?;
    let tiled_actual = tiled.to_host_vec().sync_on(&device.stream)?;
    for lane in 0..lanes {
        let position = start + i32::try_from(lane)? + offset;
        for head in 0..HEADS {
            let base = (lane * HEADS + head) * DIM;
            let expected = if lane < usize::try_from(count)? && position >= 0 {
                reference(
                    &query[base..base + DIM],
                    &expected_keys,
                    &expected_values,
                    head,
                    usize::try_from(position)?,
                    usize::try_from(window)?,
                )
            } else {
                vec![0.0; DIM]
            };
            for (dim, expected) in expected.into_iter().enumerate() {
                let error = (f64::from(tiled_actual[base + dim]) - expected).abs();
                assert!(
                    error <= 1e-4 * expected.abs().max(1.0),
                    "tiled start={start} count={count} offset={offset} window={window} lane={lane} head={head} dim={dim} error={error}"
                );
            }
        }
    }
    assert_eq!(
        keys.to_host_vec().sync_on(&device.stream)?,
        expected_keys,
        "append changed inactive rows or head strides"
    );
    assert_eq!(
        values.to_host_vec().sync_on(&device.stream)?,
        expected_values
    );
    let actual = output.to_host_vec().sync_on(&device.stream)?;
    for lane in 0..lanes {
        let position = start + i32::try_from(lane)? + offset;
        for head in 0..HEADS {
            let base = (lane * HEADS + head) * DIM;
            let expected = if lane < usize::try_from(count)? && position >= 0 {
                reference(
                    &query[base..base + DIM],
                    &expected_keys,
                    &expected_values,
                    head,
                    usize::try_from(position)?,
                    usize::try_from(window)?,
                )
            } else {
                vec![0.0; DIM]
            };
            for (dim, expected) in expected.into_iter().enumerate() {
                assert!(
                    (f64::from(actual[base + dim]) - expected).abs() < 2e-5,
                    "start={start} count={count} offset={offset} window={window} lane={lane} head={head} dim={dim}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; run inside safe-run with --test-threads 1"]
fn chunk_rotary_uses_each_token_position_and_preserves_unrotated_tail()
-> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 128;
    const HALF: usize = 32;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let input = data(LANES * HEADS * WIDTH, 5);
    let frequencies: Vec<_> = (0..HALF)
        .map(|i| 0.01 / f32::from(u16::try_from(i + 1).expect("rotary width")))
        .collect();
    let input_device = device.upload(input.clone(), &[LANES * HEADS, WIDTH])?;
    let frequency_device = device.upload(frequencies.clone(), &[HALF])?;
    let metadata = device.upload(vec![37_i32, 17, 0, 0], &[4])?;
    let mut output = api::zeros::<f32>(&[LANES * HEADS, WIDTH]).sync_on(&device.stream)?;
    crate::resident::kernels::aux::prefill_rope(
        (&mut output).partition([1, HALF]),
        &input_device,
        &frequency_device,
        &metadata,
    )
    .generics(vec![WIDTH.to_string(), HALF.to_string(), HEADS.to_string()])
    .sync_on(&device.stream)?;
    let actual = output.to_host_vec().sync_on(&device.stream)?;
    for row in 0..LANES * HEADS {
        let position = f32::from(u16::try_from(37 + row / HEADS)?);
        for d in 0..WIDTH {
            let base = row * WIDTH;
            let expected = if d < HALF * 2 {
                let angle = position * frequencies[d % HALF];
                let other = if d < HALF {
                    -input[base + d + HALF]
                } else {
                    input[base + d - HALF]
                };
                input[base + d].mul_add(angle.cos(), other * angle.sin())
            } else {
                input[base + d]
            };
            assert!(
                (actual[base + d] - expected).abs() < 2e-5,
                "rotary row={row} dimension={d}"
            );
        }
    }
    Ok(())
}
