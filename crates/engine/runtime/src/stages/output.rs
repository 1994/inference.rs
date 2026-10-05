//! Numerical validation, sampling and projection execute outside the scheduler owner.
use infer_core::{Error, Result};
use infer_ir::{CanonicalRequest, ModelOutput, WorkloadOutput};
use infer_spi::WorkloadProvider;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct OutputShape {
    pub logits: usize,
    pub rows: usize,
    pub width: usize,
}
pub struct OutputJob {
    pub request: Arc<CanonicalRequest>,
    pub output: ModelOutput,
    pub shape: OutputShape,
    pub sample: Option<usize>,
    pub generated: infer_ir::TokenBuffer,
    pub project: Option<Vec<ModelOutput>>,
}
pub struct OutputAcknowledgement {
    pub result: Result<ProcessedOutput>,
    pub elapsed_us: u64,
}
pub struct ProcessedOutput {
    pub output: Option<ModelOutput>,
    pub token: Option<u32>,
    pub projection: Option<WorkloadOutput>,
}
impl OutputJob {
    pub fn process(
        mut self,
        workloads: &dyn WorkloadProvider,
        scratch: &mut infer_workloads::SamplingWorkspace,
    ) -> Result<ProcessedOutput> {
        self.shape.validate(&self.output)?;
        if let Some(position) = self.sample {
            let token = infer_workloads::sample_with_history(
                &self.output.logits,
                &self.request.sampling,
                self.request.id.get(),
                position,
                infer_workloads::SamplingHistory {
                    prompt: match &self.request.input {
                        infer_ir::RequestInput::Sequence { tokens, .. } => tokens,
                        _ => &[],
                    },
                    generated: &self.generated,
                },
                scratch,
            )?;
            return Ok(ProcessedOutput {
                output: Some(self.output),
                token: Some(token),
                projection: None,
            });
        }
        if let Some(mut outputs) = self.project.take() {
            outputs.push(self.output);
            let projection = workloads.postprocess(&self.request, &outputs)?;
            return Ok(ProcessedOutput {
                output: None,
                token: None,
                projection: Some(projection),
            });
        }
        Ok(ProcessedOutput {
            output: Some(self.output),
            token: None,
            projection: None,
        })
    }
}
impl OutputShape {
    fn validate(self, output: &ModelOutput) -> Result<()> {
        if output.logits.len() != self.logits
            || output.hidden.len() != self.rows
            || output
                .hidden
                .iter()
                .any(|row| row.len() != self.width || row.iter().any(|value| !value.is_finite()))
            || output.logits.iter().any(|value| !value.is_finite())
        {
            return Err(Error::invariant("backend output shape/numerics mismatch"));
        }
        Ok(())
    }
}
