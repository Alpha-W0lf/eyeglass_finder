from __future__ import annotations

from typing import Literal

import torch
import os

from src.utils.config import AppConfig


DeviceType = Literal["cuda", "mps", "cpu"]


def get_best_available_device(config: AppConfig | None = None) -> DeviceType:
    if os.environ.get("FORCE_CPU_ONLY", "false").lower() == "true":
        return "cpu"

    if config and config.hardware.device_override:
        return config.hardware.device_override

    if torch.cuda.is_available():
        return "cuda"
        
    use_mps = config.hardware.use_mps_acceleration if config else True
    if use_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
        
    return "cpu"


def torch_device(config: AppConfig) -> torch.device:
    return torch.device(get_best_available_device(config))
