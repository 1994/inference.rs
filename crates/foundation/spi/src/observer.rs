//! Observer extension contract.
use infer_core::event::SemanticEvent;

pub trait Observer {
    fn observe(&mut self, events: &[SemanticEvent]);
}
