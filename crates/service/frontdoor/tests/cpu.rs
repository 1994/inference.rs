use infer_core::{Error, Result};
use infer_frontdoor::{cpu::CpuConfig, cpu::CpuPool};
use std::{sync::Arc, sync::Barrier, sync::mpsc, time::Duration};

fn config() -> CpuConfig {
    CpuConfig {
        workers: 2,
        max_jobs: 4,
        max_bytes: 4096,
        ..Default::default()
    }
}
#[tokio::test]
async fn preparation_workers_run_concurrently_and_a_blocked_job_does_not_block_delivery()
-> Result<()> {
    let pool = CpuPool::new(config())?;
    let barrier = Arc::new(Barrier::new(2));
    let first = pool.clone();
    let a = barrier.clone();
    let one = tokio::spawn(async move {
        first
            .run(32, move |_| {
                a.wait();
                Ok(1)
            })
            .await
    });
    let second = pool.clone();
    let two = tokio::spawn(async move {
        second
            .run(32, move |_| {
                barrier.wait();
                Ok(2)
            })
            .await
    });
    assert_eq!(one.await.map_err(|e| Error::invariant(e.to_string()))??, 1);
    assert_eq!(two.await.map_err(|e| Error::invariant(e.to_string()))??, 2);
    assert_eq!(pool.inspect().peak_workers, 2);
    let (release, gate) = mpsc::channel();
    let worker = pool.clone();
    let blocked = tokio::spawn(async move {
        worker
            .run(32, move |_| {
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                Ok(())
            })
            .await
    });
    assert_eq!(pool.run(32, |_| Ok(3)).await?, 3);
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    blocked
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(pool.inspect().retained_bytes, 0);
    Ok(())
}
#[tokio::test]
async fn timeout_keeps_byte_credit_until_the_actual_cpu_job_exits() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 32,
        job_timeout_ms: 20,
        ..config()
    })?;
    let (release, gate) = mpsc::channel();
    let result = pool
        .run(32, move |_| {
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
            Ok(())
        })
        .await;
    assert!(result.is_err());
    assert_eq!(pool.inspect().retained_bytes, 32);
    assert!(pool.run(1, |_| Ok(())).await.is_err());
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    for _ in 0..100 {
        if pool.inspect().retained_jobs == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(pool.inspect().retained_jobs, 0);
    assert_eq!(pool.run(1, |_| Ok(7)).await?, 7);
    Ok(())
}
#[tokio::test]
async fn failed_provider_does_not_destroy_worker_or_leak_admission_credit() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        ..config()
    })?;
    let result = pool
        .run::<()>(32, |_| std::panic::resume_unwind(Box::new(())))
        .await;
    assert!(result.is_err());
    assert_eq!(pool.run(32, |_| Ok(9)).await?, 9);
    assert_eq!(pool.inspect().retained_bytes, 0);
    Ok(())
}
