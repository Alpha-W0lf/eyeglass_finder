from pathlib import Path

import pytest

from src.utils.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["config.yaml", "production.yaml"])
def test_load_config_parses_every_repo_config(name):
    config = load_config(REPO_ROOT / "config" / name)
    assert isinstance(config, AppConfig)
    assert config.data_processing.file_pattern.endswith("train-*.parquet")
    assert config.model_params.face_detection.min_face_size > 0
    assert 0 < config.model_params.classification.eyewear_prob_threshold <= 1


def test_production_config_matches_showcase_run():
    config = load_config(REPO_ROOT / "config" / "production.yaml")
    assert config.paths.input_dir == "data/raw"
    assert config.data_processing.file_pattern == "train-*.parquet"
    assert config.data_processing.chunk_size == 512
    assert config.hardware.max_workers == 10
    assert config.model_params.classification.eyewear_prob_threshold == 0.9
