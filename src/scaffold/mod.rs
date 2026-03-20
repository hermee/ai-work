use std::fs;
use std::path::Path;

pub const DIRS: &[&str] = &[
    "",
    "src",
    "data",
    "notes",
    "config",
    "config/general",
    "config/datamodule",
    "config/experiment",
    "config/experiment/lg",
    "config/experiment/hf",
    "config/logger",
    "config/callbacks",
];

pub fn create_dirs(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for d in DIRS {
        fs::create_dir_all(root.join(d))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_all_directories() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("test-proj");
        create_dirs(&root).unwrap();

        for d in DIRS {
            assert!(root.join(d).is_dir(), "missing directory: {d}");
        }
    }

    #[test]
    fn is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("test-proj");
        create_dirs(&root).unwrap();
        create_dirs(&root).unwrap(); // should not error
        for d in DIRS {
            assert!(root.join(d).is_dir());
        }
    }
}
