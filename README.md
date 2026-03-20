# 🧬 AI Project Template CLI

A CLI tool that scaffolds production-ready AI/ML research projects with [Hydra](https://hydra.cc/) config management, [PyTorch Lightning](https://lightning.ai/), and [uv](https://docs.astral.sh/uv/) for dependency management.

## Features

- Interactive or fully non-interactive project setup
- Auto-detects OS (Linux / macOS / Windows) and NVIDIA GPU
- Scaffolds a structured project with Hydra configs, Lightning boilerplate, and data directories
- Installs PyTorch with the correct CUDA index (Linux + GPU) or CPU/MPS wheel (macOS)
- Optional [Weights & Biases](https://wandb.ai/) and [HuggingFace Transformers](https://huggingface.co/transformers/) integration

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (to build from source)
- [uv](https://docs.astral.sh/uv/) — must be available on `PATH`

## Installation

```bash
cargo build --release
# Binary is at target/release/project-template-cli
```

## Usage

### Interactive mode

Run with no arguments to get an interactive wizard:

```bash
project-template-cli
```

You'll be prompted for:

1. Project name
2. Python version (3.10 – 3.13)
3. PyTorch version (2.4.1 – 2.8.0)
4. CUDA version (only on Linux with a detected GPU)
5. Whether to install WandB
6. Whether to install HuggingFace Transformers

### Non-interactive mode

Pass all options as flags to skip prompts entirely:

```bash
project-template-cli \
  --name my-project \
  --output ~/projects \
  --python 3.12 \
  --torch 2.8.0 \
  --cuda 12.8 \
  --wandb true \
  --transformers false
```

### CLI Options

| Flag               | Short | Default | Description                                      |
|--------------------|-------|---------|--------------------------------------------------|
| `--name <NAME>`    | `-n`  | —       | Project name (prompted if omitted)               |
| `--output <DIR>`   | `-o`  | `.`     | Parent directory for the new project              |
| `--python <VER>`   |       | —       | Python version (e.g. `3.12`)                     |
| `--torch <VER>`    |       | —       | PyTorch version (e.g. `2.8.0`)                   |
| `--cuda <VER>`     |       | —       | CUDA version (e.g. `12.8`); Linux + GPU only     |
| `--wandb <BOOL>`   |       | —       | Install Weights & Biases (`true` / `false`)      |
| `--transformers <BOOL>` |  | —       | Install HuggingFace Transformers (`true`/`false`) |
| `--help`           | `-h`  |         | Print help                                        |
| `--version`        | `-V`  |         | Print version                                     |

## Generated Project Structure

```
my-project/
├── config/
│   ├── callbacks/
│   │   └── default.yaml      # Checkpoint & early stopping
│   ├── datamodule/
│   │   └── default.yaml      # Data paths, splits, batch size
│   ├── experiment/
│   │   ├── default.yaml      # Baseline training config
│   │   └── debug.yaml        # Fast debug run (2 epochs, CPU)
│   ├── general/
│   │   └── default.yaml      # Project-wide settings
│   ├── logger/
│   │   └── default.yaml      # WandB logger (or null)
│   └── config.yaml           # Hydra root config
├── src/
│   ├── __init__.py
│   └── main.py               # Entry point with seed, timing, watermark
├── data/
│   └── .gitkeep
├── notes/
│   └── README.md
├── .gitignore
├── pyproject.toml             # Managed by uv
└── README.md
```

## Installed Packages

Every generated project includes:

| Package        | Purpose                          |
|----------------|----------------------------------|
| PyTorch        | Deep learning framework          |
| Lightning      | Training loop abstraction        |
| Hydra          | Config management                |
| OmegaConf      | Structured configs               |
| watermark      | Environment reproducibility info |
| rdkit          | Cheminformatics toolkit          |

Plus optionally: `wandb`, `transformers`.

## Running a Generated Project

```bash
cd my-project

# Default run
uv run python src/main.py

# Debug experiment (2 epochs, CPU)
uv run python src/main.py experiment=debug

# Override config values
uv run python src/main.py general.seed=123 trainer.max_epochs=50
```

## License

MIT
