from pathlib import Path

from src.utils.config import load_config
from src.utils.device import get_best_available_device

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config():
    return load_config(REPO_ROOT / "config" / "production.yaml")


def test_force_cpu_only_wins_over_everything(monkeypatch):
    monkeypatch.setenv("FORCE_CPU_ONLY", "true")
    config = _config()
    config.hardware.device_override = "cuda"
    assert get_best_available_device(config) == "cpu"


def test_device_override_is_used(monkeypatch):
    monkeypatch.delenv("FORCE_CPU_ONLY", raising=False)
    config = _config()
    config.hardware.device_override = "cuda"
    assert get_best_available_device(config) == "cuda"


def test_falls_back_to_cpu_without_accelerators(monkeypatch):
    import torch

    monkeypatch.delenv("FORCE_CPU_ONLY", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config()
    config.hardware.use_mps_acceleration = False
    assert get_best_available_device(config) == "cpu"
