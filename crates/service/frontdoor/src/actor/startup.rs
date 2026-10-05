//! Startup responsibilities.
use super::{RuntimeActor, RuntimeHandle, mpsc, observation_bytes};
use infer_core::Result;
use infer_runtime::Engine;
use infer_spi::BackendProvider;
use std::{sync::Arc, sync::atomic::AtomicBool, time::Duration, time::Instant};

impl RuntimeHandle {
    ///
    /// # Errors
    /// Returns a backend/state error if stale completions cannot be drained or an I/O error if the runtime thread cannot start.
    pub fn start<B: BackendProvider + Send + 'static>(engine: Engine<B>) -> Result<Self>
    where
        B::Ticket: Send,
    {
        Self::start_with_config(engine, crate::cpu::CpuConfig::default())
    }
    /// # Errors
    /// Rejects invalid CPU limits, unforkable preparation providers or startup failures.
    pub fn start_with_config<B: BackendProvider + Send + 'static>(
        engine: Engine<B>,
        config: crate::cpu::CpuConfig,
    ) -> Result<Self>
    where
        B::Ticket: Send,
    {
        config.validate()?;
        let preparer = engine.request_preparer()?;
        let cpu = crate::cpu::CpuPool::new(config.clone())?;
        let delivery = crate::cpu::CpuPool::named(
            crate::cpu::CpuConfig {
                workers: 1,
                max_jobs: config.max_jobs,
                ..config.clone()
            },
            "delivery",
        )?;
        let observation = crate::cpu::CpuPool::named(
            crate::cpu::CpuConfig {
                workers: 1,
                max_jobs: 2,
                ..config.clone()
            },
            "observe",
        )?;
        let observation_bytes = observation_bytes(engine.config())?;
        let engine =
            engine.into_threaded(32, Duration::from_millis(config.device_control_timeout_ms))?;
        let (sender, receiver) = mpsc::sync_channel(256);
        let (control, control_rx) = mpsc::sync_channel(64);
        let stop = Arc::new(AtomicBool::new(false));
        let actor_stop = stop.clone();
        let clock_started = Instant::now();
        let clock_origin_us = engine.now_us();
        let placement = engine.config().cpu.placement.scheduler.clone();
        let thread = placement.spawn("infer-scheduler".into(), move |_| {
            RuntimeActor::new(engine, receiver, control_rx, actor_stop, clock_started).run();
        })?;
        Ok(Self {
            commands: Arc::new(sender),
            control: Arc::new(control),
            wake: thread.thread().clone(),
            stop,
            clock_started,
            clock_origin_us,
            preparer,
            cpu,
            delivery,
            observation,
            observation_bytes,
            ingress: crate::ingress::Registry::new(config.max_jobs, config.max_bytes),
            config,
        })
    }
}
