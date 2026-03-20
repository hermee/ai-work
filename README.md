<div align="center">

# 🧬 AI Work

**Production-ready AI/ML research project scaffolding — in one command.**

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen?logo=github-actions&logoColor=white)](#testing)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4--2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89CFF0)](https://hydra.cc/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

<br/>

<img width="680" alt="aiw demo" src="https://github.com/user-attachments/assets/placeholder-demo.gif"/>

<br/>

[Installation](#installation) · [Quick Start](#quick-start) · [CLI Reference](#cli-reference) · [Project Structure](#generated-project-structure) · [Experiments](#experiment-templates)

</div>

---

## Why AI Work?

Setting up a new ML research project means wiring together PyTorch, Lightning, Hydra configs, data directories, logging, and dependency management — every single time. **`aiw`** does it all in seconds:

- 🔍 **Auto-detects** OS (Linux / macOS / Windows) and NVIDIA GPU
- ⚡ **Installs PyTorch** with the correct CUDA index automatically
- 📁 **Scaffolds** a clean, opinionated project structure with Hydra config groups
- 🧪 **Includes example experiments** — Lightning MNIST classification & HuggingFace LLM finetuning
- 📦 **Manages dependencies** via [uv](https://docs.astral.sh/uv/) with dev tools (ruff, ipykernel) out of the box

---

## Installation

### Prerequisites

| Tool | Purpose |
|------|---------|
| [Rust toolchain](https://rustup.rs/) | Build from source |
| [uv](https://docs.astral.sh/uv/) | Python package management (must be on `PATH`) |

### Build

```bash
git clone https://github.com/hermee/ai-work.git
cd ai-work
cargo build --release
```

The binary is at `target/release/aiw`. Optionally, copy it to your PATH:

```bash
cp target/release/aiw ~/.local/bin/
```

---

## Quick Start

### Interactive mode

```bash
aiw
```

You'll be guided through project name, Python version, PyTorch version, CUDA (if applicable), and optional integrations.

### Non-interactive mode

```bash
aiw \
  --name my-project \
  --output ~/research \
  --python 3.12 \
  --torch 2.8.0 \
  --cuda 12.8 \
  --wandb true \
  --transformers false
```

---

## CLI Reference

```
aiw [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--name <NAME>` | `-n` | — | Project name (prompted if omitted) |
| `--output <DIR>` | `-o` | `.` | Parent directory for the new project |
| `--python <VER>` | | — | Python version (`3.10` – `3.13`) |
| `--torch <VER>` | | — | PyTorch version (`2.4.1` – `2.8.0`) |
| `--cuda <VER>` | | — | CUDA version (`11.8` – `12.8`); Linux + GPU only |
| `--wandb <BOOL>` | | — | Install Weights & Biases |
| `--transformers <BOOL>` | | — | Install HuggingFace Transformers |
| `--help` | `-h` | | Print help |
| `--version` | `-V` | | Print version |

> Any omitted flag triggers an interactive prompt.

---

## Generated Project Structure

```
my-project/
├── config/
│   ├── callbacks/
│   │   └── default.yaml          # ModelCheckpoint & EarlyStopping
│   ├── datamodule/
│   │   └── default.yaml          # Data paths, splits, batch size
│   ├── experiment/
│   │   ├── default.yaml          # Baseline training config
│   │   ├── debug.yaml            # Fast debug run (2 epochs, CPU)
│   │   ├── lg/
│   │   │   └── mnist_classify.yaml   # ⚡ Lightning MNIST example
│   │   └── hf/
│   │       └── llm_finetune.yaml     # 🤗 HuggingFace LoRA finetune example
│   ├── general/
│   │   └── default.yaml          # Project-wide settings
│   ├── logger/
│   │   └── default.yaml          # WandB logger (or null)
│   └── config.yaml               # Hydra root config
├── src/
│   ├── __init__.py
│   └── main.py                   # Entry point with seed, timing, watermark
├── data/
│   └── .gitkeep
├── notes/
│   └── README.md
├── .gitignore
├── pyproject.toml                # Managed by uv
└── README.md
```

---

## Installed Packages

### Runtime

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| Lightning | Training loop abstraction |
| Hydra | Config management |
| OmegaConf | Structured configs |
| watermark | Environment reproducibility info |
| rdkit | Cheminformatics toolkit |

### Dev

| Package | Purpose |
|---------|---------|
| ipykernel | Jupyter notebook support |
| ruff | Fast Python linter & formatter |

### Optional

| Package | Flag | Purpose |
|---------|------|---------|
| wandb | `--wandb true` | Experiment tracking |
| transformers | `--transformers true` | HuggingFace models |

---

## Experiment Templates

### ⚡ Lightning — MNIST Classification

A complete config showing all trainer, model, and datamodule options:

```bash
uv run python src/main.py experiment=lg/mnist_classify
```

<details>
<summary>Config preview</summary>

```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  precision: 16-mixed
  gradient_clip_val: 1.0

model:
  _target_: src.models.MNISTClassifier
  input_dim: 784
  hidden_dim: 256
  output_dim: 10
  learning_rate: 1e-3
  dropout: 0.2
  scheduler: cosine
```

</details>

### 🤗 HuggingFace — LLM Finetuning (LoRA)

A reference config for parameter-efficient finetuning:

```bash
uv run python src/main.py experiment=hf/llm_finetune
```

<details>
<summary>Config preview</summary>

```yaml
model:
  pretrained_model_name: meta-llama/Llama-3.2-1B
  load_in_4bit: true

peft:
  method: lora
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, v_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  lr_scheduler_type: cosine
```

</details>

---

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

---

## Environment Detection

| OS | GPU | PyTorch Install |
|----|-----|-----------------|
| Linux | ✅ NVIDIA | `torch` from `pytorch.org/whl/cu<ver>` |
| Linux | ❌ | CPU wheel via `uv add` |
| macOS | — | CPU/MPS wheel via `uv add` |
| Windows | — | CPU wheel via `uv add` |

---

## Testing

The project includes 17 tests covering scaffold creation, template generation, and conditional config logic.

```bash
cargo test
```

```
running 14 tests  (unit)    ... ok
running  3 tests  (integration) ... ok

test result: ok. 17 passed; 0 failed
```

---

## Examples

**Minimal CPU project:**

```bash
aiw -n quick-test --python 3.12 --torch 2.8.0 --wandb false --transformers false
```

**Full GPU project with all integrations:**

```bash
aiw \
  --name drug-discovery \
  --output ~/research \
  --python 3.12 \
  --torch 2.8.0 \
  --cuda 12.8 \
  --wandb true \
  --transformers true
```

---

## License

MIT

---

<div align="center">
<sub>Built with 🦀 Rust — scaffolds 🐍 Python AI projects in seconds.</sub>
</div>
