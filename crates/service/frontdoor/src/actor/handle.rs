//! Handle responsibilities.
use super::{Command, RuntimeHandle, async_mpsc, mpsc, oneshot, preparation_bytes};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_observe::{ObservationQuery, trace::TraceContext};
use infer_runtime::{EngineOutput, RuntimeInspection};
use std::{time::Duration, time::Instant};

impl RuntimeHandle {
    #[must_use]
    pub fn cpu_pool(&self) -> crate::cpu::CpuPool {
        self.cpu.clone()
    }
    #[must_use]
    pub fn delivery_pool(&self) -> crate::cpu::CpuPool {
        self.delivery.clone()
    }
    #[must_use]
    pub fn cpu_inspection(&self) -> crate::cpu::CpuInspection {
        self.cpu.inspect()
    }
    pub(super) async fn reply<T>(&self, response: oneshot::Receiver<T>) -> Result<T> {
        tokio::time::timeout(
            Duration::from_millis(self.config.command_timeout_ms),
            response,
        )
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::Backend,
                "runtime command acknowledgement timed out",
            )
        })?
        .map_err(|_| Error::new(ErrorCode::Backend, "engine actor stopped"))
    }
    pub(super) fn send(&self, command: Command) -> Result<()> {
        let sender = if matches!(command, Command::Submit { .. }) {
            &self.commands
        } else {
            &self.control
        };
        sender.try_send(command).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => {
                Error::new(ErrorCode::Capacity, "frontdoor command queue full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                Error::new(ErrorCode::Backend, "engine actor stopped")
            }
        })?;
        self.wake.unpark();
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an invalid-input, capacity, or conflict error if the runtime rejects admission or its command channel is closed.
    pub async fn submit(
        &self,
        request: CanonicalRequest,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        self.submit_with_trace(request, None).await
    }
    /// # Errors
    /// Returns admission, capacity, conflict, or actor lifecycle errors.
    pub async fn submit_with_trace(
        &self,
        request: CanonicalRequest,
        trace: Option<TraceContext>,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let bytes = preparation_bytes(&request)?;
        let id = request.id;
        self.submit_preparing_until(id, bytes, trace, request.qos.deadline_us, move |_| {
            Ok(request)
        })
        .await
    }
    /// Perform text/workload preparation under the same request identity and bounded ingress credit.
    /// # Errors
    /// Returns duplicate, capacity, cancellation, preparation or engine admission errors.
    pub async fn submit_preparing(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        work: impl FnOnce(&crate::cpu::CpuContext) -> Result<CanonicalRequest> + Send + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        self.submit_preparing_until(id, bytes, trace, None, work)
            .await
    }
    /// # Errors
    /// Returns identity, byte budget, deadline, preparation or admission errors.
    pub async fn submit_preparing_until(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        deadline_us: Option<u64>,
        work: impl FnOnce(&crate::cpu::CpuContext) -> Result<CanonicalRequest> + Send + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let now_us = self.clock_origin_us.saturating_add(
            u64::try_from(self.clock_started.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
        let deadline = deadline_us.map(|deadline| {
            Instant::now() + Duration::from_micros(deadline.saturating_sub(now_us))
        });
        let pending = self.ingress.reserve(id, bytes)?;
        let _abandon = crate::ingress::Abandon(pending.clone());
        let preparing = pending.clone();
        let preparer = self.preparer.clone();
        let request = self
            .cpu
            .run_until(bytes, deadline, move |context| {
                preparing.check()?;
                let request = work(context)?;
                if request.id != id {
                    return Err(Error::invalid("CPU preparation changed request identity"));
                }
                preparing.check()?;
                preparer.prepare(request)
            })
            .await?;
        pending.check()?;
        let (output, receiver) = async_mpsc::channel(16);
        let (reply, response) = oneshot::channel();
        self.send(Command::Submit {
            request: Box::new(request),
            pending,
            trace,
            output,
            reply,
        })?;
        self.reply(response).await??;
        Ok(receiver)
    }
    ///
    /// # Errors
    /// Returns a not-found or conflict error if the request is unknown or the runtime channel is closed.
    pub async fn cancel(&self, request: RequestId) -> Result<()> {
        let (reply, response) = oneshot::channel();
        let preparing = self.ingress.cancel(request);
        self.send(Command::Cancel {
            request,
            preparing,
            reply,
        })?;
        self.reply(response).await?
    }
    ///
    /// # Errors
    /// Returns a conflict error if the runtime channel is closed or inspection cannot be delivered.
    pub async fn inspect(&self) -> Result<RuntimeInspection> {
        let (reply, response) = oneshot::channel();
        self.send(Command::Inspect { reply })?;
        self.reply(response).await
    }
    /// # Errors
    /// Returns a capacity, query-validation, or stopped-actor error.
    pub async fn observe(&self, query: ObservationQuery) -> Result<serde_json::Value> {
        let bytes = match query {
            ObservationQuery::Metrics => 64 << 10,
            ObservationQuery::Diagnostics => 2 << 20,
            ObservationQuery::Events { limit, .. } => limit
                .checked_mul(1024)
                .and_then(|bytes| bytes.checked_add(256 << 10))
                .ok_or_else(|| Error::invalid("observation response budget overflow"))?,
            _ => self.observation_bytes,
        };
        let credit = self.observation.reserve(bytes)?;
        let (reply, response) = oneshot::channel();
        self.send(Command::Observe {
            query,
            credit,
            reply,
        })?;
        let (snapshot, credit) = self.reply(response).await??;
        self.observation
            .run_reserved(credit, None, move |_| snapshot.render())
            .await
    }
    ///
    /// # Errors
    /// Returns a conflict error if the runtime channel is closed or shutdown cannot be acknowledged.
    pub async fn shutdown(&self) -> Result<()> {
        let (reply, response) = oneshot::channel();
        self.send(Command::Stop { reply })?;
        tokio::time::timeout(Duration::from_secs(5),response)
            .await.map_err(|_|Error::new(ErrorCode::Backend,"shutdown acknowledgement timed out; in-flight resources remain owned by the actor"))?
            .map_err(|_| Error::new(ErrorCode::Backend, "engine actor stopped"))?
    }
}
