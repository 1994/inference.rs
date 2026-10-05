//! Inference commands.
#[cfg(feature = "test-backends")]
use super::{DemoOptions, examples};
use super::{
    RunOptions, backend, config, print, read_json, results, run_requests, selected_engine,
    write_json,
};
use infer_core::Result;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_ir::CanonicalRequest;
#[cfg(feature = "test-backends")]
use infer_runtime::RuntimeConfig;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn run(options: RunOptions, backend_choice: backend::Selection) -> Result<()> {
    let RunOptions {
        requests,
        config: config_path,
        model,
        package,
        device_memory_mib,
        op_trace,
        journal,
        snapshot,
        events,
    } = options;
    let input: Vec<CanonicalRequest> = read_json(&requests)?;
    let ids = input.iter().map(|r| r.id).collect::<Vec<_>>();
    let mut engine = selected_engine(
        config(config_path.as_deref())?,
        model.as_deref(),
        package.as_deref(),
        device_memory_mib,
        backend_choice,
    )?;
    let trace = run_requests(&mut engine, input)?;
    if let Some(path) = journal {
        write_json(&path, &engine.journal()?)?;
    }
    if let Some(path) = snapshot {
        write_json(&path, &engine.snapshot()?)?;
    }
    if let Some(path) = events {
        write_json(&path, &trace)?;
    }
    if let Some(path) = op_trace {
        write_json(&path, &engine.backend().trace())?;
    }
    print(
        &serde_json::json!({"runtime":engine.inspect(),"execution":engine.backend().inspection(),"results":results(&engine,&ids)?}),
    )?;

    Ok(())
}
#[cfg(feature = "test-backends")]
pub fn demo(options: DemoOptions, backend_choice: backend::Selection) -> Result<()> {
    let DemoOptions { journal, snapshot } = options;
    let mut engine = selected_engine(RuntimeConfig::default(), None, None, 512, backend_choice)?;
    let input = examples()?;
    let ids = input.iter().map(|r| r.id).collect::<Vec<_>>();
    run_requests(&mut engine, input)?;
    if let Some(path) = journal {
        write_json(&path, &engine.journal()?)?;
    }
    if let Some(path) = snapshot {
        write_json(&path, &engine.snapshot()?)?;
    }
    print(&serde_json::json!({"runtime":engine.inspect(),"results":results(&engine,&ids)?}))?;

    Ok(())
}
