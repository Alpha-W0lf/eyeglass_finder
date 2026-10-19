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
    eyewear_prob_threshold: float = 0.5
    kind: str = "eyeglasses"  # glasses-detector classifier kind: anyglasses | eyeglasses | sunglasses | shadows
    enable_sunglasses_rejection: bool = False


@dataclass
class ModelParamsConfig:
    face_detection: FaceDetectionConfig
    classification: ClassificationConfig


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class ReportGenerationConfig:
    qualitative_analysis_sample_size: int
    # The public finds gallery shows only faces scored at or above this value (0 shows every find).
    gallery_min_eyeglasses_proba: float = 0.0
    # Optional CSV (image_id, face_index, glasses_confidence, visual_review); when set, the finds page shows only visual_review == eyeglasses.
    gallery_visual_review_csv: str | None = None


@dataclass
class HardwareConfig:
    max_workers: int
    use_mps_acceleration: bool
    device_override: str | None


@dataclass
class PerformanceConfig:
    """Settings related to performance optimization."""
    prefetch_chunks: int
    inference_batch_size: int
    # Face classification batching
    face_classification_batch_size: int = 16
    # Ramp-up (startup smoothing)
    rampup_enabled: bool = False
    rampup_warmup_chunks: int = 0
    rampup_initial_prefetch_chunks: int | None = None
    rampup_initial_chunk_size_override: int | None = None
    rampup_stagger_worker_submissions_ms: int = 0


@dataclass
class PerformanceAlertsConfig:
    memory_threshold: int


@dataclass
class ObservabilityConfig:
    detailed_metrics: bool
    memory_profiling: bool
    performance_alerts: PerformanceAlertsConfig


@dataclass
class SamplingConfig:
    """Controls how face crops are generated for artifacts and (optionally) classifier input."""
    square_crop: bool = True
    crop_margin: float = 0.20
    pad_mode: str | None = None
    apply_to_classifier: bool = False


@dataclass
class AppConfig:
    paths: PathsConfig
    data_processing: DataProcessingConfig
    model_params: ModelParamsConfig
    logging: LoggingConfig
    report_generation: ReportGenerationConfig
    hardware: HardwareConfig
    performance: PerformanceConfig
    observability: ObservabilityConfig
    sampling: SamplingConfig | None = None
    # Optional at runtime; set by scripts
    run_id: str | None = None


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r") as f:
        return config_from_dict(yaml.safe_load(f))


def config_from_dict(raw: Dict[str, Any]) -> AppConfig:
    """Reconstructs an AppConfig object from a dictionary."""
    paths = PathsConfig(**raw["paths"])
    data_processing = DataProcessingConfig(**raw["data_processing"])
    
    fd = FaceDetectionConfig(**raw["model_params"]["face_detection"])
    cls = ClassificationConfig(**raw["model_params"]["classification"])
    model_params = ModelParamsConfig(face_detection=fd, classification=cls)
    
    logging = LoggingConfig(**raw["logging"])
    report_generation = ReportGenerationConfig(**raw["report_generation"])
    hardware = HardwareConfig(**raw["hardware"])
    performance = PerformanceConfig(**raw["performance"])
    
    alerts = PerformanceAlertsConfig(**raw["observability"]["performance_alerts"])
    observability = ObservabilityConfig(
        detailed_metrics=raw["observability"]["detailed_metrics"],
        memory_profiling=raw["observability"]["memory_profiling"],
        performance_alerts=alerts,
    )
    
    return AppConfig(
        paths=paths,
        data_processing=data_processing,
        model_params=model_params,
        logging=logging,
        report_generation=report_generation,
        hardware=hardware,
        performance=performance,
        observability=observability,
        sampling=SamplingConfig(**raw.get("sampling", {})) if isinstance(raw.get("sampling", {}), dict) else SamplingConfig(),
        run_id=raw.get("run_id"),
    )
