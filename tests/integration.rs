use ai_work::{scaffold, templates, Os, ProjectConfig};
use tempfile::TempDir;

fn full_scaffold(wandb: bool, transformers: bool, gpu: bool) -> TempDir {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("integ-proj");

    let cfg = ProjectConfig {
        name: "integ-proj".into(),
        output: tmp.path().to_path_buf(),
        python_version: "3.12".into(),
        pytorch_version: "2.8.0".into(),
        cuda_version: if gpu { Some("12.8".into()) } else { None },
        use_wandb: wandb,
        use_transformers: transformers,
        os: Os::Linux,
        has_gpu: gpu,
    };

    scaffold::create_dirs(&root).unwrap();
    templates::write_all(&root, &cfg).unwrap();
    tmp
}

#[test]
fn full_project_minimal() {
    let tmp = full_scaffold(false, false, false);
    let root = tmp.path().join("integ-proj");

    // All directories exist
    for d in scaffold::DIRS {
        assert!(root.join(d).is_dir(), "missing dir: {d}");
    }

    // All core files exist and are non-empty
    let files = [
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
    for f in &files {
        let path = root.join(f);
        assert!(path.exists(), "missing file: {f}");
        // __init__.py and .gitkeep are allowed to be empty
        if !f.contains("__init__") && !f.contains(".gitkeep") {
            assert!(
                std::fs::metadata(&path).unwrap().len() > 0,
                "file is empty: {f}"
            );
        }
    }
}

#[test]
fn full_project_all_options_enabled() {
    let tmp = full_scaffold(true, true, true);
    let root = tmp.path().join("integ-proj");

    let main_py = std::fs::read_to_string(root.join("src/main.py")).unwrap();
    assert!(main_py.contains("import wandb"));
    assert!(main_py.contains(",transformers"));
    assert!(main_py.contains("CUDA_VISIBLE_DEVICES"));

    let logger = std::fs::read_to_string(root.join("config/logger/default.yaml")).unwrap();
    assert!(logger.contains("wandb:"));
}

#[test]
fn no_extra_files_outside_template() {
    let tmp = full_scaffold(false, false, false);
    let root = tmp.path().join("integ-proj");

    // Walk the tree and ensure every file is one we expect
    let mut count = 0;
    for entry in walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        count += 1;
        let rel = entry.path().strip_prefix(&root).unwrap();
        let rel_str = rel.to_string_lossy();
        // Just ensure it's under a known top-level dir
        let valid_prefixes = ["src/", "config/", "data/", "notes/", ".gitignore", "README.md"];
        assert!(
            valid_prefixes.iter().any(|p| rel_str.starts_with(p)),
            "unexpected file: {rel_str}"
        );
    }
    assert!(count > 0, "no files found");
}
