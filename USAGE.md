# Usage Guide

## Quick Start

### Build the CLI

```bash
cargo build --release
```

The binary is at `target/release/project-template-cli`.

### Create a Project

```bash
# Interactive — prompts for every option
project-template-cli

# Non-interactive — supply all flags
project-template-cli \
  --name my-project \
  --output ~/projects \
  --python 3.12 \
  --torch 2.8.0 \
  --cuda 12.8 \
  --wandb true \
  --transformers false
```

---

## CLI Reference

```
project-template-cli [OPTIONS]
```

| Flag                    | Short | Default | Description                                                  |
|-------------------------|-------|---------|--------------------------------------------------------------|
| `--name <NAME>`         | `-n`  | —       | Project name. Prompted interactively if omitted.             |
| `--output <DIR>`        | `-o`  | `.`     | Parent directory where the project folder will be created.   |
| `--python <VER>`        |       | —       | Python version. Choices: `3.10`, `3.11`, `3.12`, `3.13`.    |
| `--torch <VER>`         |       | —       | PyTorch version. Choices: `2.4.1`, `2.5.1`, `2.6.0`, `2.7.1`, `2.8.0`. |
| `--cuda <VER>`          |       | —       | CUDA version. Choices: `11.8`, `12.1`, `12.4`, `12.6`, `12.8`. Only applies on Linux with an NVIDIA GPU. |
| `--wandb <BOOL>`        |       | —       | Install Weights & Biases. `true` or `false`.                 |
| `--transformers <BOOL>` |       | —       | Install HuggingFace Transformers. `true` or `false`.         |
| `--help`                | `-h`  |         | Print help.                                                  |
| `--version`             | `-V`  |         | Print version.                                               |

Any flag that is omitted in non-interactive mode will trigger an interactive prompt for that value.

---

## Environment Detection

On startup the CLI automatically detects:

- **Operating system** — Linux, macOS, or Windows.
- **NVIDIA GPU** — runs `nvidia-smi` to check for a GPU.

These determine:

| OS      | GPU   | PyTorch install method                                      |
|---------|-------|-------------------------------------------------------------|
| Linux   | Yes   | `uv pip install torch==<ver> --index-url https://download.pytorch.org/whl/cu<ver>` |
| Linux   | No    | `uv add torch==<ver>` (CPU wheel)                           |
| macOS   | —     | `uv add torch==<ver>` (CPU/MPS wheel resolved by pip)       |
| Windows | —     | `uv add torch==<ver>`                                       |

---

## Working with the Generated Project

### Directory Layout

```
my-project/
├── config/              # Hydra configuration tree
│   ├── callbacks/
│   │   └── default.yaml
│   ├── datamodule/
│   │   └── default.yaml
│   ├── experiment/
│   │   ├── default.yaml
│   │   └── debug.yaml
│   ├── general/
│   │   └── default.yaml
│   ├── logger/
│   │   └── default.yaml
│   └── config.yaml      # Hydra root — composes all groups
├── src/
│   ├── __init__.py
│   └── main.py          # Entry point
├── data/
│   └── .gitkeep
├── notes/
│   └── README.md
├── .gitignore
├── pyproject.toml
└── README.md
```

### Running Experiments

```bash
cd my-project

# Default training run
uv run python src/main.py

# Debug run (2 epochs, CPU, fast_dev_run)
uv run python src/main.py experiment=debug

# Override any config value from the command line
uv run python src/main.py general.seed=123 trainer.max_epochs=50

# Combine experiment preset with overrides
uv run python src/main.py experiment=debug trainer.max_epochs=5
```

### Hydra Config Groups

| Group        | File              | Key settings                                      |
|--------------|-------------------|---------------------------------------------------|
| `general`    | `default.yaml`    | `project_name`, `seed`, `device`, `num_workers`   |
| `datamodule` | `default.yaml`    | `data_dir`, `batch_size`, train/val/test splits    |
| `experiment` | `default.yaml`    | `max_epochs`, `accelerator`, `learning_rate`       |
| `experiment` | `debug.yaml`      | 2 epochs, CPU, `fast_dev_run: true`                |
| `logger`     | `default.yaml`    | WandB config (or `null` if WandB was not selected) |
| `callbacks`  | `default.yaml`    | `ModelCheckpoint`, `EarlyStopping`                 |

To add a new experiment, create a YAML file in `config/experiment/` and run it with:

```bash
uv run python src/main.py experiment=<filename_without_extension>
```

### Adding Packages

The project uses `uv` for dependency management:

```bash
cd my-project
uv add <package-name>
```

### WandB Integration

If WandB was enabled during scaffolding:

- `config/logger/default.yaml` contains the WandB project settings.
- `src/main.py` imports `wandb` and includes a test log call.
- Set your API key with `wandb login` or the `WANDB_API_KEY` environment variable before running.

If WandB was not enabled, the logger config is set to `null` and no WandB code is generated.

---

## Examples

### Minimal CPU project

```bash
project-template-cli \
  --name quick-test \
  --python 3.12 \
  --torch 2.8.0 \
  --wandb false \
  --transformers false
```

### Full GPU project with all integrations

```bash
project-template-cli \
  --name drug-discovery \
  --output ~/research \
  --python 3.12 \
  --torch 2.8.0 \
  --cuda 12.8 \
  --wandb true \
  --transformers true
```

### Scaffold into a specific directory

```bash
project-template-cli --name my-project --output /tmp/experiments
# Creates /tmp/experiments/my-project/
```
