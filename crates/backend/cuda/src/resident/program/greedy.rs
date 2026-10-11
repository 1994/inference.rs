//! Draft readout and external-hidden reuse on the program's owned CUDA stream.
use super::{CudaDevice, DeviceProgram, ExternalInput, ProgramWeights, device_error};
use crate::resident::greedy::Greedy;
use cutile::prelude::*;
use infer_core::{Error, Result};

pub(super) struct ProgramReadouts {
    pub greedy: Option<DeviceGreedy>,
    pub token_readout: Option<crate::resident::token_readout::TokenReadout>,
    pub hidden: Arc<Tensor<f32>>,
    pub logits: Arc<Tensor<f32>>,
}
impl ProgramReadouts {
    pub(super) fn new(
        device: &CudaDevice,
        arena: &mut crate::resident::ActivationArena,
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        vocabulary: usize,
    ) -> Result<Self> {
        let hidden = crate::resident::batch::take_result(arena, graph.hidden)?;
        let logits = crate::resident::batch::take_result(arena, graph.logits)?;
        let greedy = DeviceGreedy::new(device, &logits, vocabulary, weights)?;
        let token_readout = if logits.size() == vocabulary {
            crate::resident::token_readout::TokenReadout::contiguous(device, &logits, 1, weights)?
        } else {
            None
        };
        Ok(Self {
            greedy,
            token_readout,
            hidden,
            logits,
        })
    }
}

pub(super) struct DeviceGreedy {
    graph: CudaGraph<()>,
    selector: Greedy,
}
impl DeviceGreedy {
    pub(super) fn new(
        device: &CudaDevice,
        logits: &Tensor<f32>,
        vocabulary: usize,
        weights: &ProgramWeights,
    ) -> Result<Option<Self>> {
        if weights.fusion.is_none() || logits.size() != vocabulary {
            return Ok(None);
        }
        // A diagnostic switch permits matched baseline/candidate runs with one binary.
        if std::env::var("INFER_CUDA_DEVICE_GREEDY").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let tile = crate::tuning::greedy::select(
            device,
            1,
            vocabulary,
            weights.autotune,
            &weights.resident_tuning,
        );
        let mut selector = Greedy::new(device, 1, vocabulary, tile)?;
        let graph = CudaGraph::scope(&device.stream, |scope| selector.record(scope, logits))
            .map_err(device_error)?;
        tracing::info!(target:"infer::capture", device_greedy=true, rows=1, vocabulary, tile, "draft GPU token readout captured");
        Ok(Some(Self { graph, selector }))
    }
}
impl DeviceProgram {
    pub(crate) const fn has_device_greedy(&self) -> bool {
        self.greedy.is_some()
    }
    /// The first draft consumes a target hidden row; later drafts reuse their own resident output.
    pub(crate) fn step_greedy(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
    ) -> Result<u32> {
        if self.greedy.is_none() {
            return Err(Error::unsupported("device greedy readout unavailable"));
        }
        let previous = Arc::clone(&self.hidden);
        let external =
            external.map_or_else(|| ExternalInput::Device(&previous), ExternalInput::Host);
        self.prepare_step(
            token,
            [position; infer_models::mrope::ROTARY_AXES],
            kv_position,
            Some(external),
            true,
        )?;
        let greedy = self
            .greedy
            .as_ref()
            .ok_or_else(|| Error::invariant("device greedy readout"))?;
        self.graph
            .launch()
            .then(|()| greedy.graph.launch())
            .sync_on(&self.device.stream)
            .map_err(device_error)?;
        let tokens = greedy.selector.read(&self.device)?;
        let token = *tokens
            .first()
            .ok_or_else(|| Error::invariant("device greedy token readback"))?;
        self.last_batch = None;
        self.next_position = kv_position + 1;
        Ok(token)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_program_greedy.rs"]
mod tests;
