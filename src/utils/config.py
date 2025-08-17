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
    # Optional at runtime; set by scripts when a run starts
    logs_dir: str | None = None


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
class HardwareConfig:
    max_workers: int
    worker_memory_limit_mb: int
    use_mps_acceleration: bool
    device_override: str | None


@dataclass
class PerformanceConfig:
    """Settings related to performance optimization."""
    prefetch_chunks: int
    gc_frequency: int
    thread_pool_workers: int
    batch_optimization: bool
    inference_batch_size: int


@dataclass
class PerformanceAlertsConfig:
    cpu_threshold: int
    memory_threshold: int
    min_throughput: float


@dataclass
class ObservabilityConfig:
    detailed_metrics: bool
    memory_profiling: bool
    benchmark_mode: bool
    performance_alerts: PerformanceAlertsConfig


@dataclass
class AppConfig:
    paths: PathsConfig
    data_processing: DataProcessingConfig
    model_params: ModelParamsConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    report_generation: ReportGenerationConfig
    hardware: HardwareConfig
    performance: PerformanceConfig
    observability: ObservabilityConfig
    # Optional at runtime; set by scripts
    run_id: str | None = None


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
    hardware = HardwareConfig(**raw["hardware"])  # type: ignore[arg-type]
    performance = PerformanceConfig(**raw["performance"])  # type: ignore[arg-type]

    alerts = PerformanceAlertsConfig(**raw["observability"]["performance_alerts"])
    observability = ObservabilityConfig(
        detailed_metrics=raw["observability"]["detailed_metrics"],
        memory_profiling=raw["observability"]["memory_profiling"],
        benchmark_mode=raw["observability"]["benchmark_mode"],
        performance_alerts=alerts,
    )

    return AppConfig(
        paths=paths,
        data_processing=data_processing,
        model_params=model_params,
        execution=execution,
        logging=logging,
        report_generation=report_generation,
        hardware=hardware,
        performance=performance,
        observability=observability,
    )


def config_from_dict(raw: Dict[str, Any]) -> AppConfig:
    """Reconstructs an AppConfig object from a dictionary."""
    paths = PathsConfig(**raw["paths"])
    data_processing = DataProcessingConfig(**raw["data_processing"])
    
    fd = FaceDetectionConfig(**raw["model_params"]["face_detection"])
    cls = ClassificationConfig(**raw["model_params"]["classification"])
    model_params = ModelParamsConfig(face_detection=fd, classification=cls)
    
    execution = ExecutionConfig(**raw["execution"])
    logging = LoggingConfig(**raw["logging"])
    report_generation = ReportGenerationConfig(**raw["report_generation"])
    hardware = HardwareConfig(**raw["hardware"])
    performance = PerformanceConfig(**raw["performance"])
    
    alerts = PerformanceAlertsConfig(**raw["observability"]["performance_alerts"])
    observability = ObservabilityConfig(
        detailed_metrics=raw["observability"]["detailed_metrics"],
        memory_profiling=raw["observability"]["memory_profiling"],
        benchmark_mode=raw["observability"]["benchmark_mode"],
        performance_alerts=alerts,
    )
    
    return AppConfig(
        paths=paths,
        data_processing=data_processing,
        model_params=model_params,
        execution=execution,
        logging=logging,
        report_generation=report_generation,
        hardware=hardware,
        performance=performance,
        observability=observability,
        run_id=raw.get("run_id"),
    )
