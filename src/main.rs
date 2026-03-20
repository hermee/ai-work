use ai_work::{compatible_cuda, scaffold, templates, Os, ProjectConfig};

use clap::Parser;
use console::style;
use dialoguer::{Confirm, FuzzySelect, Input};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

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
    println!("{}", style(r#"
  ╔══════════════════════════════════════════════════════╗
  ║                                                      ║
  ║   🧬  aiw — AI Work  v0.1.0                       ║
  ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━               ║
  ║   Scaffold production-ready AI research projects     ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝
"#).cyan().bold());
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

fn progress_bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("  {msg} [{bar:30.cyan/dim}] {pos}/{len}")
            .unwrap()
            .progress_chars("█▓░"),
    );
    pb.set_message(msg.to_string());
    pb
}

fn run_cmd(cmd: &str, args: &[&str], cwd: &PathBuf) -> Result<String, String> {
    let output = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to run {cmd}: {e}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
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
    let sp = spinner("Initializing uv project...");
    run_cmd("uv", &["init", "--python", &cfg.python_version, "--no-readme"], &root)?;
    for f in &["hello.py", "main.py"] {
        let _ = std::fs::remove_file(root.join(f));
    }
    sp.finish_with_message(format!(
        "{} uv project initialized (Python {})",
        style("✔").green(), cfg.python_version
    ));

    // 3. Install packages
    let mut packages: Vec<String> = vec![
        "hydra-core".into(),
        "omegaconf".into(),
        "watermark".into(),
        "rdkit".into(),
        "lightning".into(),
    ];
    let torch_spec = format!("torch=={}", cfg.pytorch_version);
    if cfg.use_wandb { packages.push("wandb".into()); }
    if cfg.use_transformers { packages.push("transformers".into()); }

    let total = packages.len() as u64 + 1;
    let pb = progress_bar(total, "📦 Installing packages");

    if cfg.os == Os::Linux && cfg.has_gpu && cfg.cuda_version.is_some() {
        let cu = cfg.cuda_version.as_ref().unwrap().replace('.', "");
        let idx_url = format!("https://download.pytorch.org/whl/cu{cu}");
        // Configure a named PyTorch index scoped only to torch
        let pyproject = root.join("pyproject.toml");
        let content = std::fs::read_to_string(&pyproject)
            .map_err(|e| format!("Failed to read pyproject.toml: {e}"))?;
        let index_config = format!(
            "\n[[tool.uv.index]]\nname = \"pytorch-cu{cu}\"\nurl = \"{idx_url}\"\nexplicit = true\n\n[tool.uv.sources]\ntorch = {{ index = \"pytorch-cu{cu}\" }}\n"
        );
        std::fs::write(&pyproject, format!("{content}{index_config}"))
            .map_err(|e| format!("Failed to write pyproject.toml: {e}"))?;
        run_cmd("uv", &["add", &torch_spec], &root)?;
    } else {
        run_cmd("uv", &["add", &torch_spec], &root)?;
    }
    pb.set_position(1);

    let pkg_refs: Vec<&str> = packages.iter().map(|s| s.as_str()).collect();
    if !pkg_refs.is_empty() {
        let mut args: Vec<&str> = vec!["add"];
        args.extend(&pkg_refs);
        run_cmd("uv", &args, &root)?;
    }
    pb.set_position(total);
    pb.finish_with_message(format!("{} All packages installed", style("✔").green()));

    // 4. Install dev dependencies
    let sp = spinner("Installing dev dependencies...");
    run_cmd("uv", &["add", "--dev", "ipykernel", "ruff"], &root)?;
    sp.finish_with_message(format!("{} Dev dependencies installed (ipykernel, ruff)", style("✔").green()));

    // Done
    println!();
    println!("  {} Project {} created successfully!", style("🎉").bold(), style(&cfg.name).green().bold());
    println!();
    println!("  {} Next steps:", style("→").cyan());
    println!("    cd {}", cfg.name);
    println!("    uv run python src/main.py");
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
