use ai_work::{compatible_cuda, scaffold, templates, Os, ProjectConfig};

use clap::Parser;
use console::style;
use dialoguer::{Confirm, FuzzySelect, Input};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::thread;

#[derive(Parser)]
#[command(name = "aiw", version, about = "🧬 AI Project Template Scaffolding CLI")]
struct Cli {
    #[arg(short, long)]
    name: Option<String>,
    #[arg(short, long, default_value = ".")]
    output: String,
    #[arg(long)]
    python: Option<String>,
    #[arg(long)]
    torch: Option<String>,
    #[arg(long)]
    cuda: Option<String>,
    #[arg(long)]
    wandb: Option<bool>,
    #[arg(long)]
    transformers: Option<bool>,
}

fn detect_os() -> Os {
    if cfg!(target_os = "macos") { Os::Mac }
    else if cfg!(target_os = "windows") { Os::Windows }
    else { Os::Linux }
}

fn detect_gpu() -> bool {
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn banner() {
    let lines = [
        "",
        "    ╭──────────────────────────────────────────────────╮",
        "    │                                                  │",
        "    │   🧬  aiw — AI Work                             │",
        "    │   ──────────────────                             │",
        "    │   Scaffold production-ready AI research projects │",
        "    │                                                  │",
        "    ╰──────────────────────────────────────────────────╯",
        "",
    ];
    for line in &lines {
        println!("{}", style(line).cyan().bold());
        thread::sleep(Duration::from_millis(30));
    }
}

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Run a uv command with live output streaming and a spinner.
fn run_uv_live(args: &[&str], cwd: &PathBuf, label: &str) -> Result<(), String> {
    let sp = ProgressBar::new_spinner();
    sp.set_style(
        ProgressStyle::with_template("  {spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"]),
    );
    sp.set_message(label.to_string());
    sp.enable_steady_tick(Duration::from_millis(80));

    let mut child = Command::new("uv")
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run uv: {e}"))?;

    // Stream stderr (uv writes progress to stderr)
    let stderr = child.stderr.take().unwrap();
    let sp_clone = sp.clone();
    let label_owned = label.to_string();
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut last_lines = Vec::new();
        for line in reader.lines() {
            if let Ok(line) = line {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    sp_clone.set_message(format!("{} │ {}", label_owned, style(&trimmed).dim()));
                    last_lines.push(trimmed);
                }
            }
        }
        last_lines
    });

    // Capture stdout
    let stdout = child.stdout.take().unwrap();
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut out = String::new();
        for line in reader.lines() {
            if let Ok(line) = line {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    });

    let status = child.wait().map_err(|e| format!("uv process error: {e}"))?;
    let stderr_lines = stderr_handle.join().unwrap_or_default();
    let _stdout = stdout_handle.join().unwrap_or_default();

    if status.success() {
        sp.finish_with_message(format!("{} {}", style("✔").green(), label));
        Ok(())
    } else {
        sp.finish_with_message(format!("{} {}", style("✘").red(), label));
        let err_msg = stderr_lines.into_iter().rev().take(5).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        Err(err_msg)
    }
}

fn gather_config(cli: &Cli) -> Result<ProjectConfig, Box<dyn std::error::Error>> {
    let os = detect_os();
    let has_gpu = detect_gpu();

    let os_label = match os {
        Os::Linux => "🐧 Linux",
        Os::Mac => "🍎 macOS",
        Os::Windows => "🪟 Windows",
    };
    println!(
        "  {} Detected: {}  │  GPU: {}",
        style("ℹ").blue(),
        style(os_label).green(),
        if has_gpu { style("✔ NVIDIA GPU found").green().to_string() }
        else { style("✘ No GPU").yellow().to_string() }
    );
    println!();

    let name: String = match &cli.name {
        Some(n) => n.clone(),
        None => Input::new()
            .with_prompt(format!("  {} Project name", style("📁").bold()))
            .interact_text()?,
    };

    let python_version = if let Some(ref p) = cli.python {
        p.clone()
    } else {
        let opts = vec!["3.12", "3.11", "3.10", "3.13"];
        let idx = FuzzySelect::new()
            .with_prompt(format!("  {} Python version", style("🐍").bold()))
            .items(&opts)
            .default(0)
            .interact()?;
        opts[idx].to_string()
    };

    let pytorch_version = if let Some(ref t) = cli.torch {
        t.clone()
    } else {
        let opts = vec!["2.8.0", "2.7.1", "2.6.0", "2.5.1", "2.4.1"];
        let idx = FuzzySelect::new()
            .with_prompt(format!("  {} PyTorch version", style("🔥").bold()))
            .items(&opts)
            .default(0)
            .interact()?;
        opts[idx].to_string()
    };

    let cuda_version = if os == Os::Linux && has_gpu {
        let cuda_opts = compatible_cuda(&pytorch_version);
        if let Some(ref c) = cli.cuda {
            if !cuda_opts.contains(&c.as_str()) {
                return Err(format!(
                    "CUDA {} is not compatible with PyTorch {}. Valid options: {}",
                    c, pytorch_version, cuda_opts.join(", ")
                ).into());
            }
            Some(c.clone())
        } else {
            let opts: Vec<&str> = cuda_opts.to_vec();
            let idx = FuzzySelect::new()
                .with_prompt(format!("  {} CUDA version (compatible with PyTorch {})", style("🎮").bold(), pytorch_version))
                .items(&opts)
                .default(0)
                .interact()?;
            Some(opts[idx].to_string())
        }
    } else {
        None
    };

    let use_wandb = cli.wandb.unwrap_or_else(|| {
        Confirm::new()
            .with_prompt(format!("  {} Install Weights & Biases (wandb)?", style("📊").bold()))
            .default(true)
            .interact()
            .unwrap_or(true)
    });

    let use_transformers = cli.transformers.unwrap_or_else(|| {
        Confirm::new()
            .with_prompt(format!("  {} Install HuggingFace Transformers?", style("🤗").bold()))
            .default(false)
            .interact()
            .unwrap_or(false)
    });

    Ok(ProjectConfig {
        name, output: PathBuf::from(&cli.output),
        python_version, pytorch_version, cuda_version,
        use_wandb, use_transformers, os, has_gpu,
    })
}

fn create_project(cfg: &ProjectConfig) -> Result<(), Box<dyn std::error::Error>> {
    let root = cfg.output.join(&cfg.name);

    // 1. Create folder structure + write templates
    let sp = spinner("Creating project structure...");
    scaffold::create_dirs(&root)?;
    templates::write_all(&root, cfg)?;
    sp.finish_with_message(format!("{} Project structure created", style("✔").green()));

    // 2. Init uv project
    run_uv_live(
        &["init", "--python", &cfg.python_version, "--no-readme"],
        &root,
        &format!("Initializing uv project (Python {})", cfg.python_version),
    )?;
    for f in &["hello.py", "main.py"] {
        let _ = std::fs::remove_file(root.join(f));
    }

    // 3. Configure PyTorch index in pyproject.toml
    let torch_spec = format!("torch=={}", cfg.pytorch_version);
    let pyproject = root.join("pyproject.toml");
    let content = std::fs::read_to_string(&pyproject)?;

    if cfg.os == Os::Linux && cfg.has_gpu && cfg.cuda_version.is_some() {
        let cu = cfg.cuda_version.as_ref().unwrap().replace('.', "");
        let idx_url = format!("https://download.pytorch.org/whl/cu{cu}");
        let idx_name = format!("pytorch-cu{cu}");
        let index_config = format!("\
\n[[tool.uv.index]]\n\
name = \"{idx_name}\"\n\
url = \"{idx_url}\"\n\
explicit = true\n\
\n\
[tool.uv.sources]\n\
torch = [\n\
  {{ index = \"{idx_name}\", marker = \"sys_platform == 'linux'\" }},\n\
]\n\
torchvision = [\n\
  {{ index = \"{idx_name}\", marker = \"sys_platform == 'linux'\" }},\n\
]\n\
torchaudio = [\n\
  {{ index = \"{idx_name}\", marker = \"sys_platform == 'linux'\" }},\n\
]\n");
        std::fs::write(&pyproject, format!("{content}{index_config}"))?;
    } else {
        let index_config = "\
\n[[tool.uv.index]]\n\
name = \"pytorch-cpu\"\n\
url = \"https://download.pytorch.org/whl/cpu\"\n\
explicit = true\n\
\n\
[tool.uv.sources]\n\
torch = [\n\
  { index = \"pytorch-cpu\" },\n\
]\n\
torchvision = [\n\
  { index = \"pytorch-cpu\" },\n\
]\n\
torchaudio = [\n\
  { index = \"pytorch-cpu\" },\n\
]\n";
        std::fs::write(&pyproject, format!("{content}{index_config}"))?;
    }

    // 4. Install torch
    let cuda_label = cfg.cuda_version.as_ref()
        .map(|c| format!(" + CUDA {c}"))
        .unwrap_or_default();
    run_uv_live(
        &["add", &torch_spec],
        &root,
        &format!("Installing PyTorch {}{}", cfg.pytorch_version, cuda_label),
    )?;

    // 5. Install runtime packages
    let mut packages: Vec<String> = vec![
        "hydra-core".into(),
        "omegaconf".into(),
        "watermark".into(),
        "rdkit".into(),
        "lightning".into(),
    ];
    if cfg.use_wandb { packages.push("wandb".into()); }
    if cfg.use_transformers { packages.push("transformers".into()); }

    let pkg_refs: Vec<&str> = packages.iter().map(|s| s.as_str()).collect();
    let mut args: Vec<&str> = vec!["add"];
    args.extend(&pkg_refs);
    run_uv_live(&args, &root, "Installing runtime packages")?;

    // 6. Install dev dependencies
    run_uv_live(
        &["add", "--dev", "ipykernel", "ruff"],
        &root,
        "Installing dev dependencies (ipykernel, ruff)",
    )?;

    // Done
    println!();
    println!("  {} Project {} created successfully!", style("🎉").bold(), style(&cfg.name).green().bold());
    println!();
    println!("  {} Next steps:", style("→").cyan());
    println!("    {}", style(format!("cd {}", cfg.name)).dim());
    println!("    {}", style("uv run python src/main.py").dim());
    println!();

    Ok(())
}

fn main() {
    banner();
    let cli = Cli::parse();

    let cfg = match gather_config(&cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  {} {}", style("✘").red(), e);
            std::process::exit(1);
        }
    };

    println!();
    println!("  {} Configuration Summary:", style("📋").bold());
    println!("  ├─ Project:      {}", style(&cfg.name).green());
    println!("  ├─ Python:       {}", style(&cfg.python_version).cyan());
    println!("  ├─ PyTorch:      {}", style(&cfg.pytorch_version).cyan());
    if let Some(ref cu) = cfg.cuda_version {
        println!("  ├─ CUDA:         {}", style(cu).cyan());
    }
    println!("  ├─ WandB:        {}", if cfg.use_wandb { style("Yes").green() } else { style("No").yellow() });
    println!("  └─ Transformers: {}", if cfg.use_transformers { style("Yes").green() } else { style("No").yellow() });
    println!();

    if let Err(e) = create_project(&cfg) {
        eprintln!("  {} Error: {}", style("✘").red(), e);
        std::process::exit(1);
    }
}
