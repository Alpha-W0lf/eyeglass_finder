from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PathsConfig:
    data_dir: str
    outputs_dir: str
    models_dir: str


@dataclass
class FaceDetectionConfig:
    model_path: str
    min_confidence: float
    det_size: int
    max_detections: int


@dataclass
class ModelParamsConfig:
    face_detection: FaceDetectionConfig


@dataclass
class AppConfig:
    paths: PathsConfig
    model_params: ModelParamsConfig


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    paths = PathsConfig(**raw["paths"])  # type: ignore[arg-type]
    fd = FaceDetectionConfig(**raw["model_params"]["face_detection"])  # type: ignore[index]
    model_params = ModelParamsConfig(face_detection=fd)
    return AppConfig(paths=paths, model_params=model_params)
