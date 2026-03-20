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

/// Returns compatible CUDA versions for a given PyTorch version.
pub fn compatible_cuda(torch: &str) -> &'static [&'static str] {
    match torch {
        "2.10.0" => &["12.8", "12.6", "11.8"],
        "2.9.1" => &["13.0", "12.8", "12.6"],
        "2.8.0" => &["12.9", "12.8", "12.6"],
        "2.7.1" => &["12.8", "12.6", "11.8"],
        "2.6.0" => &["12.6", "12.4", "11.8"],
        "2.5.1" => &["12.4", "12.1", "11.8"],
        "2.4.1" => &["12.4", "12.1", "11.8"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::compatible_cuda;

    #[test]
    fn torch_2100_supports_128_126_118() {
        let opts = compatible_cuda("2.10.0");
        assert_eq!(opts, &["12.8", "12.6", "11.8"]);
    }

    #[test]
    fn torch_291_supports_130_128_126() {
        let opts = compatible_cuda("2.9.1");
        assert_eq!(opts, &["13.0", "12.8", "12.6"]);
    }

    #[test]
    fn torch_280_supports_129_128_126() {
        let opts = compatible_cuda("2.8.0");
        assert_eq!(opts, &["12.9", "12.8", "12.6"]);
    }

    #[test]
    fn torch_271_supports_128_126_118() {
        let opts = compatible_cuda("2.7.1");
        assert_eq!(opts, &["12.8", "12.6", "11.8"]);
    }

    #[test]
    fn torch_260_supports_126_124_118() {
        let opts = compatible_cuda("2.6.0");
        assert_eq!(opts, &["12.6", "12.4", "11.8"]);
    }

    #[test]
    fn torch_251_supports_124_121_118() {
        let opts = compatible_cuda("2.5.1");
        assert_eq!(opts, &["12.4", "12.1", "11.8"]);
    }

    #[test]
    fn torch_241_supports_124_121_118() {
        let opts = compatible_cuda("2.4.1");
        assert_eq!(opts, &["12.4", "12.1", "11.8"]);
    }

    #[test]
    fn unknown_torch_returns_empty() {
        assert!(compatible_cuda("1.0.0").is_empty());
    }
}
