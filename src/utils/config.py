from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PathsConfig:
    input_dir: str
    output_dir: str
    output_filename: str


@dataclass
class DataProcessingConfig:
    file_pattern: str
    chunk_size: int


@dataclass
class FaceDetectionConfig:
    model_path: str
    min_face_size: int
    min_confidence: float
    keep_all: bool
    target_size: List[int]


@dataclass
class ClassificationConfig:
    present_label: str
    output_image_format: str


@dataclass
class ModelParamsConfig:
    face_detection: FaceDetectionConfig
    classification: ClassificationConfig


@dataclass
class ExecutionConfig:
    num_workers: int
    diagnostic_serial_mode: bool
    inference_batch_size: int


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class ReportGenerationConfig:
    qualitative_analysis_sample_size: int


@dataclass
class AppConfig:
    paths: PathsConfig
    data_processing: DataProcessingConfig
    model_params: ModelParamsConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    report_generation: ReportGenerationConfig


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    paths = PathsConfig(**raw["paths"])  # type: ignore[arg-type]
    data_processing = DataProcessingConfig(**raw["data_processing"])  # type: ignore[arg-type]

    fd = FaceDetectionConfig(**raw["model_params"]["face_detection"])  # type: ignore[index]
    cls = ClassificationConfig(**raw["model_params"]["classification"])  # type: ignore[index]
    model_params = ModelParamsConfig(face_detection=fd, classification=cls)

    execution = ExecutionConfig(**raw["execution"])  # type: ignore[arg-type]
    logging = LoggingConfig(**raw["logging"])  # type: ignore[arg-type]
    report_generation = ReportGenerationConfig(**raw["report_generation"])  # type: ignore[arg-type]

    return AppConfig(
        paths=paths,
        data_processing=data_processing,
        model_params=model_params,
        execution=execution,
        logging=logging,
        report_generation=report_generation,
    )
