use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::Command,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_MODEL: &str = "claude-sonnet-5-5";
const MAX_STEPS: usize = 15; // safety cap on tool-use iterations per request
const MAX_OUTPUT_CHARS: usize = 8000;

const SYSTEM_PROMPT: &str = "You are Coding Buddy, a careful coding assistant working inside a \
project directory. Use the provided tools to explore files, write code, and run commands. \
Read before you edit, make small focused changes, and explain what you did briefly.";

fn tool_definitions() -> Value {
    json!([
        {
            "name": "list_dir",
            "description": "List files and folders in a directory (relative to project root).",
            "input_schema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Relative dir path, '.' for root" } },
                "required": ["path"]
            }
        },
        {
            "name": "read_file",
            "description": "Read a UTF-8 text file (relative to project root).",
            "input_schema": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "input_schema": {
                "type": "object",
                "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
                "required": ["path", "content"]
            }
        },
        {
            "name": "run_command",
            "description": "Run a shell command in the project root (e.g. 'cargo test'). The user must approve it.",
            "input_schema": {
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }
        }
    ])
}

/// Resolve a relative path inside the workspace; reject absolute paths and `..`.
fn safe_path(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let p = Path::new(rel);
    if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("path '{rel}' is outside the workspace"));
    }
    Ok(root.join(p))
}

fn truncate(s: String) -> String {
    if s.chars().count() > MAX_OUTPUT_CHARS {
        let cut: String = s.chars().take(MAX_OUTPUT_CHARS).collect();
        format!("{cut}\n...[truncated]")
    } else {
        s
    }
}

fn confirm(prompt: &str) -> bool {
    print!("{prompt} [y/N]: ");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).is_ok() && buf.trim().eq_ignore_ascii_case("y")
}

/// Execute one tool call. Returns (output, is_error).
fn execute_tool(name: &str, input: &Value, root: &Path) -> (String, bool) {
    let result: Result<String, String> = (|| match name {
        "list_dir" => {
            let dir = safe_path(root, input["path"].as_str().unwrap_or("."))?;
            let mut names: Vec<String> = fs::read_dir(&dir)
                .map_err(|e| e.to_string())?
                .filter_map(|e| e.ok())
                .map(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    if e.path().is_dir() { format!("{n}/") } else { n }
                })
                .collect();
            names.sort();
            Ok(names.join("\n"))
        }
        "read_file" => {
            let p = safe_path(root, input["path"].as_str().ok_or("missing 'path'")?)?;
            fs::read_to_string(p).map_err(|e| e.to_string())
        }
        "write_file" => {
            let p = safe_path(root, input["path"].as_str().ok_or("missing 'path'")?)?;
            let content = input["content"].as_str().ok_or("missing 'content'")?;
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(&p, content).map_err(|e| e.to_string())?;
            Ok(format!("wrote {} bytes to {}", content.len(), p.display()))
        }
        "run_command" => {
            let cmd = input["command"].as_str().ok_or("missing 'command'")?;
            if !confirm(&format!("\n⚠️  Run command `{cmd}`?")) {
                return Err("user denied the command".into());
            }
            let out = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(root)
                .output()
                .map_err(|e| e.to_string())?;
            Ok(format!(
                "exit: {}\nstdout:\n{}\nstderr:\n{}",
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ))
        }
        other => Err(format!("unknown tool '{other}'")),
    })();

    match result {
        Ok(s) => (truncate(s), false),
        Err(e) => (e, true),
    }
}

/// The agent loop: call the LLM, run requested tools, feed results back.
fn run_agent(
    client: &Client,
    api_key: &str,
    model: &str,
    root: &Path,
    messages: &mut Vec<Value>,
) -> Result<(), String> {
    for _ in 0..MAX_STEPS {
        let body = json!({
            "model": model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "tools": tool_definitions(),
            "messages": messages,
        });

        let resp = client
            .post(API_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        let v: Value = resp.json().map_err(|e| format!("bad response: {e}"))?;
        if !status.is_success() {
            return Err(format!("API error {status}: {v}"));
        }

        let content = v["content"].as_array().cloned().unwrap_or_default();
        messages.push(json!({ "role": "assistant", "content": content }));

        let mut tool_results: Vec<Value> = Vec::new();
        for block in &content {
            match block["type"].as_str() {
                Some("text") => println!("\n🤖 {}", block["text"].as_str().unwrap_or("")),
                Some("tool_use") => {
                    let name = block["name"].as_str().unwrap_or("");
                    let id = block["id"].as_str().unwrap_or("");
                    println!("🔧 {name} {}", block["input"]);
                    let (output, is_error) = execute_tool(name, &block["input"], root);
                    tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": output,
                        "is_error": is_error,
                    }));
                }
                _ => {}
            }
        }

        // No tool calls requested -> the model is done with this task.
        if tool_results.is_empty() {
            return Ok(());
        }
        messages.push(json!({ "role": "user", "content": tool_results }));
    }
    Err(format!("stopped after {MAX_STEPS} steps without finishing"))
}

fn main() {
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=\"your-key\"");
            std::process::exit(1);
        }
    };
    let model = env::var("CODING_BUDDY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let root = PathBuf::from(env::args().nth(1).unwrap_or_else(|| ".".into()))
        .canonicalize()
        .expect("workspace directory not found");

    let client = Client::new();
    let mut messages: Vec<Value> = Vec::new();

    println!("Coding Buddy ready. Workspace: {}  Model: {model}", root.display());
    println!("Type a task, or 'exit' to quit.\n");

    loop {
        print!("you> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        messages.push(json!({ "role": "user", "content": line }));
        if let Err(e) = run_agent(&client, &api_key, &model, &root, &mut messages) {
            eprintln!("❌ {e}");
        }
        println!();
    }
}
