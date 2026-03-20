use crate::{Os, ProjectConfig};
use std::fs;
use std::path::Path;

pub fn write_all(root: &Path, cfg: &ProjectConfig) -> Result<(), Box<dyn std::error::Error>> {
    // ── .gitignore ──
    fs::write(root.join(".gitignore"), "\
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
data/*
!data/.gitkeep
.idea/
.vscode/
*.swp
.DS_Store
Thumbs.db
*.log
wandb/
outputs/
multirun/
uv.lock
")?;

    // ── config/config.yaml (Hydra root) ──
    fs::write(root.join("config/config.yaml"), "\
defaults:
  - general: default
  - datamodule: default
  - experiment: default
  - logger: default
  - callbacks: default
  - _self_

seed: 42
debug: false
")?;

    // ── config/general/default.yaml ──
    fs::write(root.join("config/general/default.yaml"), format!("\
project_name: {name}
seed: 42
device: auto
precision: 32
num_workers: 4
verbose: true
", name = cfg.name))?;

    // ── config/datamodule/default.yaml ──
    fs::write(root.join("config/datamodule/default.yaml"), "\
data_dir: ${hydra:runtime.cwd}/data
train_split: 0.8
val_split: 0.1
test_split: 0.1
batch_size: 32
num_workers: ${general.num_workers}
pin_memory: true

sources:
  raw: ${.data_dir}/raw
  processed: ${.data_dir}/processed
  external: ${.data_dir}/external
")?;

    // ── config/experiment/default.yaml ──
    fs::write(root.join("config/experiment/default.yaml"), "\
# @package _global_
name: baseline

trainer:
  max_epochs: 100
  accelerator: auto
  devices: auto
  gradient_clip_val: 1.0

model:
  learning_rate: 1e-3
  weight_decay: 1e-5
  scheduler: cosine
")?;

    // ── config/experiment/debug.yaml ──
    fs::write(root.join("config/experiment/debug.yaml"), "\
# @package _global_
name: debug

trainer:
  max_epochs: 2
  accelerator: cpu
  devices: 1
  fast_dev_run: true

model:
  learning_rate: 1e-3
")?;

    // ── config/experiment/lg/mnist_classify.yaml ──
    fs::write(root.join("config/experiment/lg/mnist_classify.yaml"), "\
# @package _global_
# Lightning MNIST classification example
# Usage: uv run python src/main.py experiment=lg/mnist_classify
name: mnist_classify

trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
  precision: 16-mixed
  gradient_clip_val: 1.0
  log_every_n_steps: 50
  check_val_every_n_epoch: 1
  deterministic: true

model:
  _target_: src.models.MNISTClassifier
  input_dim: 784
  hidden_dim: 256
  output_dim: 10
  learning_rate: 1e-3
  weight_decay: 1e-5
  dropout: 0.2
  scheduler: cosine
  warmup_epochs: 1

datamodule:
  _target_: src.datamodules.MNISTDataModule
  data_dir: ${hydra:runtime.cwd}/data
  batch_size: 128
  num_workers: ${general.num_workers}
  pin_memory: true
  train_split: 0.9
  val_split: 0.1
")?;

    // ── config/experiment/hf/llm_finetune.yaml ──
    fs::write(root.join("config/experiment/hf/llm_finetune.yaml"), "\
# @package _global_
# HuggingFace LLM finetuning example
# Usage: uv run python src/main.py experiment=hf/llm_finetune
name: llm_finetune

model:
  pretrained_model_name: meta-llama/Llama-3.2-1B
  task: causal_lm
  load_in_4bit: true
  use_peft: true

peft:
  method: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  max_steps: 1000
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  fp16: true
  logging_steps: 10
  save_steps: 200
  eval_steps: 200
  output_dir: ${hydra:runtime.cwd}/outputs/hf_finetune

dataset:
  name: tatsu-lab/alpaca
  split: train
  max_length: 512
  text_field: text
")?;

    // ── config/logger/default.yaml ──
    if cfg.use_wandb {
        fs::write(root.join("config/logger/default.yaml"), format!("\
wandb:
  project: {name}
  entity: null
  group: null
  tags: []
  mode: online
  log_model: false
  save_dir: ${{hydra:runtime.cwd}}/outputs
", name = cfg.name))?;
    } else {
        fs::write(root.join("config/logger/default.yaml"), "logger: null\n")?;
    }

    // ── config/callbacks/default.yaml ──
    fs::write(root.join("config/callbacks/default.yaml"), "\
model_checkpoint:
  monitor: val_loss
  mode: min
  save_top_k: 3
  save_last: true
  dirpath: ${hydra:runtime.cwd}/outputs/checkpoints

early_stopping:
  monitor: val_loss
  patience: 10
  mode: min
")?;

    // ── src/main.py ──
    let wandb_import = if cfg.use_wandb { "import wandb\n" } else { "" };
    let transformers_watermark = if cfg.use_transformers { ",transformers" } else { "" };
    let wandb_watermark = if cfg.use_wandb { ",wandb" } else { "" };
    let wandb_test = if cfg.use_wandb {
        r#"
    # ── wandb test ──
    wandb.init(project=config.logger.wandb.project, mode="disabled")
    wandb.log({"test_metric": 1.0})
    wandb.finish()
    print("✔ wandb test passed")
"#
    } else { "" };

    let gpu_line = match cfg.os {
        Os::Linux if cfg.has_gpu => r#"os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"#,
        _ => "",
    };

    fs::write(root.join("src/main.py"), format!(r#"import os
import time
import functools
import random

import hydra
import torch
import lightning
import numpy as np
from omegaconf import DictConfig, OmegaConf
from watermark import watermark
{wandb_import}
{gpu_line}
def get_time(func):
    """Decorator that prints execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n⏱  {{func.__name__}} took {{elapsed:.2f}}s")
        return result
    return wrapper


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@get_time
@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(watermark(packages="torch,lightning{transformers_watermark}{wandb_watermark}", python=True))

    # loading configuration
    config = OmegaConf.create(OmegaConf.to_container(cfg))

    # setting random seed for reproducibility
    set_seed(seed=config.general.seed, deterministic=True)

    print(config)
{wandb_test}

if __name__ == "__main__":
    main()
"#))?;

    // ── src/__init__.py ──
    fs::write(root.join("src/__init__.py"), "")?;

    // ── data/.gitkeep ──
    fs::write(root.join("data/.gitkeep"), "")?;

    // ── notes/README.md ──
    fs::write(root.join("notes/README.md"), format!(
        "# {} — Notes\n\nExperiment notes and observations go here.\n", cfg.name
    ))?;

    // ── README.md ──
    fs::write(root.join("README.md"), format!(r#"# {name}

## Setup

```bash
uv sync
```

## Run

```bash
uv run python src/main.py
```

## Experiments

```bash
uv run python src/main.py experiment=debug
```

## Structure

```
{name}/
├── config/
│   ├── general/
│   ├── datamodule/
│   ├── experiment/
│   ├── logger/
│   ├── callbacks/
│   └── config.yaml
├── src/
│   └── main.py
├── data/
├── notes/
└── .gitignore
```
"#, name = cfg.name))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Os, ProjectConfig};
    use tempfile::TempDir;

    fn make_cfg(tmp: &TempDir, wandb: bool, transformers: bool, gpu: bool) -> ProjectConfig {
        ProjectConfig {
            name: "test-proj".into(),
            output: tmp.path().to_path_buf(),
            python_version: "3.12".into(),
            pytorch_version: "2.8.0".into(),
            cuda_version: if gpu { Some("12.8".into()) } else { None },
            use_wandb: wandb,
            use_transformers: transformers,
            os: if gpu { Os::Linux } else { Os::Linux },
            has_gpu: gpu,
        }
    }

    fn setup(wandb: bool, transformers: bool, gpu: bool) -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("test-proj");
        crate::scaffold::create_dirs(&root).unwrap();
        let cfg = make_cfg(&tmp, wandb, transformers, gpu);
        write_all(&root, &cfg).unwrap();
        tmp
    }

    fn read(tmp: &TempDir, path: &str) -> String {
        std::fs::read_to_string(tmp.path().join("test-proj").join(path)).unwrap()
    }

    #[test]
    fn all_base_files_created() {
        let tmp = setup(false, false, false);
        let expected = [
            ".gitignore",
            "README.md",
            "src/__init__.py",
            "src/main.py",
            "data/.gitkeep",
            "notes/README.md",
            "config/config.yaml",
            "config/general/default.yaml",
            "config/datamodule/default.yaml",
            "config/experiment/default.yaml",
            "config/experiment/debug.yaml",
            "config/experiment/lg/mnist_classify.yaml",
            "config/experiment/hf/llm_finetune.yaml",
            "config/logger/default.yaml",
            "config/callbacks/default.yaml",
        ];
        for f in &expected {
            assert!(
                tmp.path().join("test-proj").join(f).exists(),
                "missing file: {f}"
            );
        }
    }

    #[test]
    fn wandb_enabled_logger_config() {
        let tmp = setup(true, false, false);
        let content = read(&tmp, "config/logger/default.yaml");
        assert!(content.contains("wandb:"));
        assert!(content.contains("project: test-proj"));
    }

    #[test]
    fn wandb_disabled_logger_config() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "config/logger/default.yaml");
        assert!(content.contains("logger: null"));
    }

    #[test]
    fn wandb_enabled_main_py() {
        let tmp = setup(true, false, false);
        let content = read(&tmp, "src/main.py");
        assert!(content.contains("import wandb"));
        assert!(content.contains("wandb.init"));
    }

    #[test]
    fn wandb_disabled_main_py() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "src/main.py");
        assert!(!content.contains("import wandb"));
    }

    #[test]
    fn transformers_watermark_included() {
        let tmp = setup(false, true, false);
        let content = read(&tmp, "src/main.py");
        assert!(content.contains(",transformers"));
    }

    #[test]
    fn gpu_sets_cuda_visible_devices() {
        let tmp = setup(false, false, true);
        let content = read(&tmp, "src/main.py");
        assert!(content.contains("CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn no_gpu_no_cuda_env() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "src/main.py");
        assert!(!content.contains("CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn project_name_in_general_config() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "config/general/default.yaml");
        assert!(content.contains("project_name: test-proj"));
    }

    #[test]
    fn lg_experiment_has_trainer_and_model() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "config/experiment/lg/mnist_classify.yaml");
        assert!(content.contains("trainer:"));
        assert!(content.contains("model:"));
        assert!(content.contains("datamodule:"));
        assert!(content.contains("MNISTClassifier"));
    }

    #[test]
    fn hf_experiment_has_peft_and_training() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "config/experiment/hf/llm_finetune.yaml");
        assert!(content.contains("peft:"));
        assert!(content.contains("training:"));
        assert!(content.contains("dataset:"));
        assert!(content.contains("lora"));
    }

    #[test]
    fn readme_contains_project_name() {
        let tmp = setup(false, false, false);
        let content = read(&tmp, "README.md");
        assert!(content.contains("# test-proj"));
    }
}
