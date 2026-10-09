use super::*;
use std::time::Instant;

type Tensors = Vec<Tensor<f32>>;

fn fixtures(device: &CudaDevice) -> Result<(Tensors, Tensors)> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for lane in 0..33_u16 {
        let size = if lane == 32 { 248_320 } else { 5120 };
        let input = device.upload(vec![f32::from(lane); size], &[size])?;
        inputs.push(Arc::try_unwrap(input).map_err(|_| Error::invariant("fixture ownership"))?);
        outputs.push(
            api::zeros::<f32>(&[size])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        );
    }
    Ok((inputs, outputs))
}

#[test]
#[ignore = "requires NVIDIA GPU; run under tools/bench/safe-run.sh"]
fn pinned_readbacks_ordering_and_overhead() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let (mut inputs, mut outputs) = fixtures(&device)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for (output, input) in outputs.iter_mut().zip(&inputs) {
            scope.record(api::memcpy(output, input))?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    let sources: Vec<_> = outputs
        .into_iter()
        .map(|output| Some(ReadbackSource::whole(Arc::new(output))))
        .collect();
    let mut readbacks = Readbacks::default();
    for token in 0..8_u16 {
        let data = device.upload(vec![f32::from(token); 5120], &[5120])?;
        graph
            .update(api::memcpy(&mut inputs[0], &data))
            .map_err(device_error)?;
        let mut selected = sources.clone();
        if token % 2 == 0 {
            selected[1] = None;
        }
        let results = readbacks.run(&device, &graph, &selected)?;
        for (lane, row) in results.iter().enumerate() {
            if selected[lane].is_none() {
                assert_eq!(row.as_slice(), &[0.0_f32; 0]);
                continue;
            }
            let expected = if lane == 0 {
                token
            } else {
                u16::try_from(lane).map_err(device_error)?
            };
            assert!(
                row.iter()
                    .all(|value| value.to_bits() == f32::from(expected).to_bits())
            );
            assert_eq!(row.len(), if lane == 32 { 248_320 } else { 5120 });
        }
    }
    let retained = Arc::clone(
        readbacks.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("pinned reuse"))?,
    );
    let _ = readbacks.run(&device, &graph, &sources)?;
    assert!(Arc::ptr_eq(
        &retained,
        readbacks.buffers[0]
            .as_ref()
            .ok_or_else(|| Error::invariant("pinned reuse"))?
    ));
    let over_budget = vec![sources[32].clone(); 68];
    assert_eq!(
        readbacks
            .run(&device, &graph, &over_budget)
            .err()
            .map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    let other = CudaDevice::new(0)?;
    assert!(readbacks.run(&other, &graph, &sources).is_err());
    measure(&device, &graph, &sources, &mut readbacks)?;
    let mut single = Readbacks::default();
    let _ = single.run(&device, &graph, &sources[32..])?;
    measure(&device, &graph, &sources[32..], &mut single)?;
    let empty = readbacks.run(&device, &graph, &[None, None])?;
    assert_eq!(empty, vec![Vec::<f32>::new(), Vec::new()]);
    partial_readback(&device, &graph, &mut readbacks)?;
    readbacks.poisoned = true;
    assert!(readbacks.run(&device, &graph, &sources).is_err());
    Ok(())
}

/// A mid-tensor range copies exactly those elements; out-of-bounds is rejected.
fn partial_readback(
    device: &CudaDevice,
    graph: &CudaGraph<()>,
    readbacks: &mut Readbacks,
) -> Result<()> {
    let ramp = device.upload((0..8192_u16).map(f32::from).collect::<Vec<f32>>(), &[8192])?;
    let partial = [Some(ReadbackSource {
        tensor: Arc::clone(&ramp),
        skip: 999,
        len: 2345,
    })];
    let rows = readbacks.run(device, graph, &partial)?;
    let expected: Vec<f32> = (999..3344_u16).map(f32::from).collect();
    assert_eq!(rows.len(), 1);
    assert!(
        rows[0]
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    let overflow = [Some(ReadbackSource {
        tensor: ramp,
        skip: 8000,
        len: 193,
    })];
    assert!(readbacks.run(device, graph, &overflow).is_err());
    Ok(())
}

fn measure(
    device: &CudaDevice,
    graph: &CudaGraph<()>,
    sources: &[Option<ReadbackSource>],
    readbacks: &mut Readbacks,
) -> Result<()> {
    let mut legacy = Vec::new();
    let mut pinned = Vec::new();
    for round in 0..6 {
        for offset in 0..2 {
            let old = (round + offset) % 2 == 0;
            let start = Instant::now();
            for _ in 0..100 {
                let rows = if old {
                    let mut reads = DeviceOpVec::with_capacity(sources.len());
                    for source in sources {
                        let source = source
                            .as_ref()
                            .ok_or_else(|| Error::invariant("benchmark source"))?;
                        reads.push(Arc::clone(&source.tensor).to_host_vec());
                    }
                    graph
                        .launch()
                        .then(move |()| reads)
                        .sync_on(&device.stream)
                        .map_err(device_error)?
                } else {
                    readbacks.run(device, graph, sources)?
                };
                std::hint::black_box(rows);
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / 100.0;
            if old {
                legacy.push(us);
            } else {
                pinned.push(us);
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "device": device.name()?, "rows": sources.len(), "iterations_per_round": 100,
            "legacy_wall_us": legacy, "pinned_wall_us": pinned,
            "scope": "33 graph D2D copies plus selected output D2H, including host Vec materialization; rows=33 is prefill Full, rows=1 is logits only; not model throughput"
        })
    );
    Ok(())
}
