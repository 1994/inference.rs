mod diagnostics;
mod observability;
mod quality;
mod runtime;
mod state;
use crate::{
    AgentBackend, AgentContext, protocol::PROTOCOL_VERSION, registry::CommandDescriptor,
    registry::CommandEffect, registry::CommandOutput, registry::CommandRegistry,
};
use infer_core::Result;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoParams {}

fn object(properties: Value, required: &[&str]) -> Value {
    let mut schema = serde_json::Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), properties);
    schema.insert("required".into(), json!(required));
    schema.insert("additionalProperties".into(), Value::Bool(false));
    Value::Object(schema)
}

fn add<B: AgentBackend, P: DeserializeOwned + 'static, R: CommandOutput + 'static>(
    registry: &mut CommandRegistry<AgentContext<B>>,
    method: &'static str,
    aliases: &[&'static str],
    description: &'static str,
    effect: CommandEffect,
    params: Value,
    execute: fn(&mut AgentContext<B>, P) -> R,
) -> Result<()> {
    registry.register(
        CommandDescriptor {
            method,
            aliases: aliases.to_vec(),
            description,
            effect,
            params,
        },
        execute,
    )
}

pub fn registry<B: AgentBackend>() -> Result<CommandRegistry<AgentContext<B>>> {
    let mut registry = CommandRegistry::default();
    add(
        &mut registry,
        "agent.discover",
        &[],
        "Discover versioned commands, aliases, effects and parameter contracts",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        discover,
    )?;
    add(
        &mut registry,
        "agent.inspect",
        &[],
        "Inspect bounded control-session audit and experiment identities",
        CommandEffect::ReadOnly,
        object(json!({}), &[]),
        inspect,
    )?;
    runtime::register(&mut registry)?;
    diagnostics::register(&mut registry)?;
    state::register(&mut registry)?;
    quality::register(&mut registry)?;
    observability::register(&mut registry)?;
    Ok(registry)
}

fn discover<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"protocol_version": PROTOCOL_VERSION,"commands":context.descriptors,"limits":{"frame_bytes":crate::transport::MAX_FRAME_BYTES,"batch_requests":64,"call_history":256,"experiment_history":16},"backend":context.engine.inspect().backend_kind})
}

fn inspect<B: AgentBackend>(context: &mut AgentContext<B>, _: NoParams) -> Value {
    json!({"session":context.session,"calls":context.calls,"dropped_calls":context.dropped_calls,"experiments":context.experiments.iter().map(|e|e.id).collect::<Vec<_>>()})
}
