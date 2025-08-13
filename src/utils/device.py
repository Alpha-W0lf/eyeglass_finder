from __future__ import annotations

from typing import Literal

import torch


DeviceType = Literal["cuda", "mps", "cpu"]


def get_best_available_device() -> DeviceType:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_device() -> torch.device:
    return torch.device(get_best_available_device())
