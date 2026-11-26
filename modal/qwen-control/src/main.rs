use serde_json::Value;
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Output},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const APP: &str = "qwen38-27b";
const SERVER: &str = "qwen38-27b/Server";
const CONTEXT: u64 = 131072;
const IDLE_SECONDS: u64 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Q8,
    Iq4,
}

impl Variant {
    fn parse(arg: Option<&str>) -> Result<Self, String> {
        match arg {
            None | Some("q8") => Ok(Self::Q8),
            Some("iq4") => Ok(Self::Iq4),
            Some(other) => Err(format!("Unknown model {other}. Choose q8 or iq4.")),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Iq4 => "iq4",
        }
    }
    fn alias(self) -> &'static str {
        match self {
            Self::Q8 => "qwen38-hauhau-q8kp",
            Self::Iq4 => "qwen38-hauhau-iq4xs",
        }
    }
    fn gpu(self) -> &'static str {
        match self {
            Self::Q8 => "L40S",
            Self::Iq4 => "L4",
        }
    }
}

fn state_path() -> Result<PathBuf, String> {
    let home = env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(PathBuf::from(home).join(".local/state/qwen-modal/variant"))
}

fn project_root() -> Result<PathBuf, String> {
    if let Some(root) = env::var_os("LOCAL_LLM_ROOT") {
        return Ok(PathBuf::from(root));
    }
    let home = env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(PathBuf::from(home).join("local-llm"))
}

fn write_state(path: &Path, state: &str) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("Missing state directory"))?;
    fs::create_dir_all(dir)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp = dir.join(format!(".variant-{}-{stamp}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    let result = (|| {
        writeln!(file, "{state}")?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        fs::File::open(dir)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

fn set_gateway_state(variant: Option<Variant>) -> Result<(), String> {
    let path = state_path()?;
    write_state(&path, variant.map_or("stopped", Variant::name))
        .map_err(|e| format!("Cannot update gateway state {}: {e}", path.display()))
}

fn show_gateway_state(expected: Option<Variant>) {
    let actual = state_path()
        .ok()
        .and_then(|path| fs::read_to_string(path).ok());
    let state = match actual.as_deref().map(str::trim) {
        Some("q8") => "q8",
        Some("iq4") => "iq4",
        Some("stopped") => "stopped",
        _ => "unavailable (requests blocked)",
    };
    println!("Gateway state: {state}");
    if state != expected.map_or("stopped", Variant::name) {
        eprintln!(
            "Gateway state differs from the Modal deployment. Run qwen-modal start or stop to reconcile it."
        );
    }
}

fn modal_command(args: &[&str], variant: Option<Variant>) -> Result<Output, String> {
    let root = project_root()?;
    // Bound control-plane calls as well as HTTP requests; a network hiccup must
    // not leave start/status/stop hanging indefinitely.
    let limit = if args.first() == Some(&"deploy") {
        "120s"
    } else {
        "45s"
    };
    let mut command = Command::new("timeout");
    command.args(["--kill-after=5s", limit]);
    command.arg(root.join(".venv-modal/bin/modal"));
    if let Some(variant) = variant {
        command.env("QWEN_VARIANT", variant.name());
    }
    if args.first() == Some(&"curl") {
        command
            .arg("curl")
            .args(["--profile", "local-llm"])
            .args(&args[1..]);
    } else {
        command.args(args).args(["--profile", "local-llm"]);
    }
    let output = command
        .output()
        .map_err(|e| format!("Cannot run Modal client: {e}"))?;
    if output.status.success() {
        Ok(output)
    } else if output.status.code() == Some(124) {
        Err(format!("Modal control-plane call timed out after {limit}."))
    } else {
        Err(format!(
            "Modal command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn modal(args: &[&str]) -> Result<Output, String> {
    modal_command(args, None)
}

fn json_command(args: &[&str]) -> Result<Value, String> {
    let output = modal(args)?;
    serde_json::from_slice(&output.stdout).map_err(|e| format!("Invalid Modal response: {e}"))
}

fn target_app(apps: &Value) -> Result<Option<(&str, &str)>, String> {
    let apps = apps.as_array().ok_or("Unexpected Modal app list")?;
    Ok(apps
        .iter()
        .filter(|app| app["description"] == APP)
        .filter_map(|app| Some((app["app_id"].as_str()?, app["state"].as_str()?)))
        .find(|(_, state)| matches!(*state, "deployed" | "stopping")))
}

fn containers(app_id: &str) -> Result<usize, String> {
    let result = json_command(&["container", "list", "--json"])?;
    let items = result.as_array().ok_or("Unexpected Modal container list")?;
    Ok(items.iter().filter(|item| item["app_id"] == app_id).count())
}

fn server_info() -> Result<Value, String> {
    json_command(&["server", "info", SERVER, "--json"])
}

fn deployed_variant(info: &Value) -> Result<Variant, String> {
    let gpus = info["gpus"].as_array().ok_or("Missing GPU configuration")?;
    match gpus.as_slice() {
        [gpu] if gpu[1].as_u64() == Some(1) => match gpu[0].as_str() {
            Some("L40S") => Ok(Variant::Q8),
            Some("L4") => Ok(Variant::Iq4),
            _ => Err("Unexpected server GPU".into()),
        },
        _ => Err("Expected exactly one server GPU".into()),
    }
}

fn server_url(info: &Value) -> Result<&str, String> {
    info["server_url"]
        .as_str()
        .filter(|url| url.starts_with("https://") && url.ends_with(".modal.direct"))
        .ok_or_else(|| "Missing Modal server URL".into())
}

fn validate_server(info: &Value, expected: Variant) -> Result<(), String> {
    let variant = deployed_variant(info)?;
    if variant != expected {
        return Err(format!(
            "Already deployed with {} ({}). Run qwen-modal stop, then qwen-modal start {}.",
            variant.name(),
            variant.gpu(),
            expected.name()
        ));
    }
    // Refuse to wake stale or changed resource settings instead of silently paying more.
    let cpu = &info["cpu"];
    if cpu[0].as_f64() != Some(1.0)
        || cpu[1].as_f64() != Some(2.0)
        || info["memory_mib"].as_u64() != Some(2048)
        || info["min_containers"].as_u64().unwrap_or(0) != 0
        || info["max_containers"].as_u64() != Some(1)
        || info["buffer_containers"].as_u64().unwrap_or(0) != 0
        || info["scaledown_window"].as_u64() != Some(IDLE_SECONDS)
        || info["target_concurrency"].as_f64() != Some(1.0)
        || info["unauthenticated"] != false
    {
        return Err(format!(
            "Deployment has stale or unexpected resource/autoscaling/auth settings. Run qwen-modal stop, then qwen-modal start {}.",
            expected.name()
        ));
    }
    Ok(())
}

fn has_alias(model: &Value, alias: &str) -> bool {
    model["id"] == alias
        || model["aliases"]
            .as_array()
            .is_some_and(|aliases| aliases.iter().any(|candidate| candidate == alias))
}

fn model_ready(response: &Value, expected: Variant) -> Result<bool, String> {
    let models = response["data"].as_array().ok_or("Invalid model list")?;
    let [model] = models.as_slice() else {
        return Err("Expected exactly one model".into());
    };
    // /v1/models may be available before model loading completes.
    let Some(context) = model["meta"]["n_ctx"].as_u64() else {
        return Ok(false);
    };
    if context != CONTEXT {
        return Err(format!("Wrong context: {context}, expected {CONTEXT}."));
    }
    let actual = if has_alias(model, Variant::Q8.alias()) {
        Some(Variant::Q8)
    } else if has_alias(model, Variant::Iq4.alias()) {
        Some(Variant::Iq4)
    } else {
        None
    };
    if actual != Some(expected) {
        return Err(format!("Loaded model does not match {}.", expected.alias()));
    }
    Ok(true)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let arg = args.first().map_or("help", String::as_str);
    if args.len() > (if arg == "start" { 2 } else { 1 }) {
        return Err("Usage: qwen-modal start [q8|iq4] | stop | status".into());
    }
    match arg {
        "start" => {
            let variant = Variant::parse(args.get(1).map(String::as_str))?;
            match target_app(&json_command(&["app", "list", "--json"])?)? {
                None => {
                    let deployment = project_root()?.join("modal/modal_qwen.py");
                    let deployment = deployment.to_str().ok_or("Checkout path must be UTF-8")?;
                    println!("Deploying {} on {}...", variant.name(), variant.gpu());
                    modal_command(&["deploy", deployment], Some(variant))?;
                }
                Some((_, "stopping")) => {
                    return Err("The app is still stopping; try again shortly.".into());
                }
                Some(_) => {}
            }
            let info = server_info()?;
            validate_server(&info, variant)?;
            let url = server_url(&info)?;
            let models_url = format!("{url}/v1/models");
            println!("Waking the model; GPU automatically sleeps when idle...");
            // Requests return 503 during zero-to-one scaling, but also trigger cold start.
            // Never pin min_containers=1: cancellation cannot leave a GPU billing indefinitely.
            let started = Instant::now();
            let deadline = started + Duration::from_secs(600);
            let mut last_progress = started;
            let mut last_error = String::from("model warming up");
            while Instant::now() < deadline {
                match modal(&[
                    "curl",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "5",
                    "--max-time",
                    "15",
                    "--write-out",
                    "\n%{http_code}",
                    &models_url,
                ]) {
                    Ok(output) => {
                        let body = String::from_utf8_lossy(&output.stdout);
                        let (json, status) = body.rsplit_once('\n').ok_or("Missing HTTP status")?;
                        let status = status.trim();
                        match status {
                            "200" => {
                                let response: Value = serde_json::from_str(json)
                                    .map_err(|e| format!("Invalid model response: {e}"))?;
                                let ready = model_ready(&response, variant).inspect_err(|_| {
                                    let _ = set_gateway_state(None);
                                })?;
                                if ready {
                                    set_gateway_state(Some(variant))?;
                                    println!(
                                        "Ready in {:.1}s.\nAPI: {url}/v1\nModel: {}\nContext: {CONTEXT} tokens\nGPU: {}. Sleeps after {IDLE_SECONDS}s idle.\nUse `qwen-modal stop` to disable the app.",
                                        started.elapsed().as_secs_f64(),
                                        variant.alias(),
                                        variant.gpu()
                                    );
                                    return Ok(());
                                }
                            }
                            "401" | "403" => {
                                return Err(format!(
                                    "Modal authentication failed (HTTP {status})."
                                ));
                            }
                            "404" => return Err("Modal server URL was not found.".into()),
                            _ => last_error = format!("HTTP {status}"),
                        }
                    }
                    Err(e) => last_error = e,
                }
                if last_progress.elapsed() >= Duration::from_secs(15) {
                    eprintln!(
                        "Waiting for GPU/model ({:.0}s): {last_error}",
                        started.elapsed().as_secs_f64()
                    );
                    last_progress = Instant::now();
                }
                thread::sleep(Duration::from_secs(2));
            }
            return Err(format!(
                "Model not ready after 10 minutes ({last_error}). The app remains enabled with automatic shutdown; run qwen-modal stop to disable it."
            ));
        }
        "stop" => {
            // Disable new gateway requests first, including during a Modal outage.
            set_gateway_state(None)?;
            let apps = json_command(&["app", "list", "--json"])?;
            if let Some((id, state)) = target_app(&apps)? {
                if state == "deployed" {
                    println!("Stopping the app and any active inference...");
                    modal(&["app", "stop", "--yes", id])?;
                } else {
                    println!("Waiting for the app to finish stopping...");
                }
                let deadline = Instant::now() + Duration::from_secs(60);
                while Instant::now() < deadline {
                    if containers(id)? == 0
                        && target_app(&json_command(&["app", "list", "--json"])?)?.is_none()
                    {
                        println!("Stopped; no GPU containers remain. Model files are preserved.");
                        return Ok(());
                    }
                    thread::sleep(Duration::from_secs(2));
                }
                return Err("Stop was requested but could not be verified within 60 seconds. Run `qwen-modal status`.".into());
            }
            println!("The app is already stopped.");
        }
        "status" => match target_app(&json_command(&["app", "list", "--json"])?)? {
            Some((id, "deployed")) => {
                let info = server_info()?;
                let variant = deployed_variant(&info)?;
                println!(
                    "App: enabled\nActive containers: {}\nAPI: {}/v1\nModel configuration: {} ({})\nCPU cores: {}\nRAM request (MiB): {}\nMin/Max containers: {}/{}\nIdle window: {}s",
                    containers(id)?,
                    server_url(&info)?,
                    variant.alias(),
                    variant.gpu(),
                    info["cpu"],
                    info["memory_mib"],
                    info["min_containers"].as_u64().unwrap_or(0),
                    info["max_containers"],
                    info["scaledown_window"]
                );
                if let Err(e) = validate_server(&info, variant) {
                    eprintln!("Warning: {e}");
                }
                show_gateway_state(Some(variant));
            }
            Some((id, _)) => {
                println!("App: stopping\nActive containers: {}", containers(id)?);
                show_gateway_state(None);
            }
            None => {
                println!("App: stopped");
                show_gateway_state(None);
            }
        },
        "help" | "--help" | "-h" => println!(
            "Usage: qwen-modal start [q8|iq4] | stop | status\n\nStart defaults to Q8 on L40S; IQ4 on L4 is a cheaper fallback.\nStop disables the app and interrupts active requests; stop first to change variants.\nThe GPU scales to zero automatically after 30 seconds idle."
        ),
        _ => {
            return Err(format!(
                "Unknown command: {arg}. Use start, stop, or status."
            ));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_is_replaced_without_leaving_temp_files() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("qwen-modal-test-{}-{stamp}", std::process::id()));
        let path = dir.join("variant");
        write_state(&path, "q8").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "q8\n");
        write_state(&path, "iq4").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "iq4\n");
        write_state(&path, "stopped").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "stopped\n");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_file(path).unwrap();
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn picks_only_the_active_target_app() {
        let apps = serde_json::json!([
            {"description": "qwen38-27b", "state": "stopped", "app_id": "old"},
            {"description": "other", "state": "deployed", "app_id": "wrong"},
            {"description": "qwen38-27b", "state": "deployed", "app_id": "right"}
        ]);
        assert_eq!(target_app(&apps).unwrap(), Some(("right", "deployed")));
        assert_eq!(target_app(&serde_json::json!([])).unwrap(), None);
        assert!(target_app(&serde_json::json!({})).is_err());
        assert_eq!(
            target_app(
                &serde_json::json!([{"description": APP, "state": "stopping", "app_id": "old"}])
            )
            .unwrap(),
            Some(("old", "stopping"))
        );
    }

    #[test]
    fn validates_variant_and_cost_settings() {
        let mut info = serde_json::json!({"gpus": [["L40S", 1]], "cpu": [1.0, 2.0], "memory_mib": 2048, "min_containers": null, "max_containers": 1, "buffer_containers": null, "scaledown_window": 30, "target_concurrency": 1.0, "unauthenticated": false});
        assert!(validate_server(&info, Variant::Q8).is_ok());
        assert!(validate_server(&info, Variant::Iq4).is_err());
        info["min_containers"] = serde_json::json!(1);
        assert!(validate_server(&info, Variant::Q8).is_err());
        info["min_containers"] = serde_json::json!(0);
        info["memory_mib"] = serde_json::json!(16384);
        assert!(validate_server(&info, Variant::Q8).is_err());
    }

    #[test]
    fn recognizes_q8_even_if_legacy_alias_is_canonical() {
        let mut response = serde_json::json!({"data": [{"id": Variant::Iq4.alias(), "aliases": [Variant::Q8.alias(), Variant::Iq4.alias()], "meta": {"n_ctx": CONTEXT}}]});
        assert!(model_ready(&response, Variant::Q8).unwrap());
        assert!(model_ready(&response, Variant::Iq4).is_err());
        response["data"][0]["meta"] = Value::Null;
        assert!(!model_ready(&response, Variant::Q8).unwrap());
        response["data"][0]["meta"] = serde_json::json!({"n_ctx": 4096});
        assert!(model_ready(&response, Variant::Q8).is_err());
    }
}
