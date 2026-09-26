from pathlib import Path


def test_required_project_directories_exist():
    root = Path(__file__).parents[1]
    required = [
        "src/ingestion", "src/segmentation", "src/audio", "src/context",
        "src/break_scoring", "src/optimization", "src/brand_matching",
        "src/manifest", "src/api", "configs", "docs", "outputs"
    ]
    for directory in required:
        assert (root / directory).is_dir(), directory
