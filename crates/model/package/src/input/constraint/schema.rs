//! Explicit supported schema subset; unknown keywords never silently weaken strict mode.
use infer_core::{Error, Result};
use serde_json::Value;
use std::collections::BTreeSet;

const MAX_SCHEMA_DEPTH: usize = 16;
const MAX_SCHEMA_NODES: usize = 256;
const MAX_PROPERTY_NAME_BYTES: usize = 256;

pub(super) fn validate(schema: &Value, strict: bool) -> Result<()> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(Error::invalid("tool parameters must have an object root"));
    }
    visit(schema, strict, 0, &mut 0)
}
fn visit(schema: &Value, strict: bool, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    if depth > MAX_SCHEMA_DEPTH || *nodes > MAX_SCHEMA_NODES {
        return Err(Error::invalid("tool schema exceeds depth or node budget"));
    }
    let object = schema
        .as_object()
        .ok_or_else(|| Error::invalid("schema must be an object"))?;
    for key in object.keys() {
        if ![
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "anyOf",
            "description",
            "title",
        ]
        .contains(&key.as_str())
        {
            return Err(Error::unsupported(format!(
                "unsupported tool schema keyword: {key}"
            )));
        }
    }
    if let Some(variants) = object.get("anyOf") {
        let variants = variants
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| Error::invalid("anyOf requires schemas"))?;
        for variant in variants {
            visit(variant, strict, depth + 1, nodes)?;
        }
    }
    let types: Vec<&str> = match object.get("type") {
        Some(Value::String(kind)) => vec![kind],
        Some(Value::Array(types)) if !types.is_empty() => types
            .iter()
            .map(|kind| {
                kind.as_str()
                    .ok_or_else(|| Error::invalid("schema type must be a string"))
            })
            .collect::<Result<_>>()?,
        None if object.contains_key("anyOf") => vec![],
        _ => return Err(Error::invalid("schema requires type or anyOf")),
    };
    if types.iter().any(|t| {
        ![
            "object", "array", "string", "number", "integer", "boolean", "null",
        ]
        .contains(t)
    }) {
        return Err(Error::invalid("unknown schema type"));
    }
    if types.contains(&"object") {
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::invalid("object schema requires properties"))?;
        let required = object.get("required").and_then(Value::as_array);
        if strict
            && (object.get("additionalProperties") != Some(&Value::Bool(false))
                || required.is_none_or(|r| r.len() != properties.len()))
        {
            return Err(Error::invalid(
                "strict objects require additionalProperties=false and every property in required",
            ));
        }
        if let Some(required) = required {
            let mut names = BTreeSet::new();
            for name in required {
                let name = name
                    .as_str()
                    .ok_or_else(|| Error::invalid("required entries must be strings"))?;
                if !properties.contains_key(name) || !names.insert(name) {
                    return Err(Error::invalid("invalid or duplicate required property"));
                }
            }
        } else if object.contains_key("required") {
            return Err(Error::invalid("required must be an array"));
        }
        for (name, child) in properties {
            if name.len() > MAX_PROPERTY_NAME_BYTES {
                return Err(Error::invalid("schema property name too long"));
            }
            visit(child, strict, depth + 1, nodes)?;
        }
    }
    if types.contains(&"array") {
        visit(
            object
                .get("items")
                .ok_or_else(|| Error::invalid("array schema requires items"))?,
            strict,
            depth + 1,
            nodes,
        )?;
    }
    Ok(())
}
