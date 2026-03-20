pub mod scaffold;
pub mod templates;

use std::path::PathBuf;

#[derive(Clone)]
pub struct ProjectConfig {
    pub name: String,
    pub output: PathBuf,
    pub python_version: String,
    pub pytorch_version: String,
    pub cuda_version: Option<String>,
    pub use_wandb: bool,
    pub use_transformers: bool,
    pub os: Os,
    pub has_gpu: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Os {
    Linux,
    Mac,
    Windows,
}

impl ProjectConfig {
    /// Create a config suitable for testing (no side effects).
    #[cfg(test)]
    pub fn test_default(name: &str, output: PathBuf) -> Self {
        Self {
            name: name.to_string(),
            output,
            python_version: "3.12".into(),
            pytorch_version: "2.8.0".into(),
            cuda_version: None,
            use_wandb: false,
            use_transformers: false,
            os: Os::Linux,
            has_gpu: false,
        }
    }
}
