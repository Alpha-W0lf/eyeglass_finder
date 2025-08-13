"""Defines a centralized class for managing and reporting pipeline metrics.

This module provides the `MetricsManager` dataclass, which serves as a single,
consistent structure for collecting all relevant metrics during a pipeline run.
It captures:
- Run metadata (e.g., git commit, configuration)
- Performance metrics (e.g., runtime, throughput)
- A detailed data funnel (e.g., images processed, faces detected, targets found)

This centralized approach simplifies metrics collection across different parts
of the pipeline and ensures a consistent structure for the final report.
"""
import time
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
import numpy as np


@dataclass
class MetricsManager:
    """
    A stateful dataclass to manage and aggregate metrics throughout a pipeline run.
    An instance of this class is created at the start of a run and updated
    incrementally as data is processed.
    """

    # Run metadata
    run_id: str = "not_set"
    git_commit_hash: str = "not_set"
    run_command: str = "not_set"
    config_snapshot: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    # Performance metrics
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_runtime_seconds: float = 0.0
    images_per_second: float = 0.0
    total_detection_time_seconds: float = 0.0
    total_classification_time_seconds: float = 0.0

    # Worker performance
    worker_processing_times: List[float] = field(default_factory=list)
    worker_time_avg: float = 0.0
    worker_time_std: float = 0.0
    worker_time_min: float = 0.0
    worker_time_max: float = 0.0

    # Data funnel metrics
    total_input_files: int = 0
    total_images_processed: int = 0
    images_with_decoding_errors: int = 0
    images_with_no_faces: int = 0
    total_faces_detected: int = 0
    faces_above_size_threshold: int = 0
    faces_classified: int = 0
    faces_with_eyeglasses: int = 0
    faces_rejected_as_sunglasses: int = 0
    faces_passing_confidence_thresholds: int = 0
    final_target_count: int = 0
    
    # Failure tracking metrics for robustness diagnostics
    failed_inference_batches: int = 0
    corrupted_batches: int = 0
    corrupted_batch_images: int = 0

    # Model quality metrics (for generating plots)
    face_confidence_scores: List[float] = field(default_factory=list)
    
    # Face count diagnostics - Track face detection patterns to identify potential issues
    # This helps investigate the unexpectedly high face detection count (24,499 faces)
    # and determine if we have group photos, false positives, or other patterns
    faces_per_image_distribution: dict = field(default_factory=dict)  # {num_faces: count}
    high_face_count_images: List[dict] = field(default_factory=list)  # Images with many faces
    max_faces_in_single_image: int = 0
    images_with_multiple_faces: int = 0  # Count of images with >1 face

    def finalize(self):
        """
        Calculates and sets the final time-based metrics for the instance.
        This method should be called once at the very end of a run. It mutates
        the instance by setting the `end_time`, `total_runtime_seconds`, and
        `images_per_second` attributes.
        """
        self.end_time = time.time()
        self.total_runtime_seconds = round(self.end_time - self.start_time, 2)
        if self.total_images_processed > 0 and self.total_runtime_seconds > 0:
            self.images_per_second = round(
                self.total_images_processed / self.total_runtime_seconds, 2
            )
        else:
            self.images_per_second = 0
        
        if self.worker_processing_times:
            self.worker_time_avg = round(np.mean(self.worker_processing_times), 2)
            self.worker_time_std = round(np.std(self.worker_processing_times), 2)
            self.worker_time_min = round(np.min(self.worker_processing_times), 2)
            self.worker_time_max = round(np.max(self.worker_processing_times), 2)

    def to_dict(self) -> dict:
        """
        Converts the metrics data into a structured dictionary.
        This is useful for serialization or structured logging.
        """
        self.finalize() # Ensure all calculations are done before saving
        return {
            "run_summary": {
                "run_id": self.run_id,
                "git_commit_hash": self.git_commit_hash,
                "run_command": self.run_command,
                "environment": self.environment,
            },
            "performance": {
                "total_runtime_seconds": self.total_runtime_seconds,
                "images_per_second": self.images_per_second,
                "total_detection_time_seconds": self.total_detection_time_seconds,
                "total_classification_time_seconds": self.total_classification_time_seconds,
            },
            "worker_performance": {
                "avg_chunk_time_seconds": self.worker_time_avg,
                "std_chunk_time_seconds": self.worker_time_std,
                "min_chunk_time_seconds": self.worker_time_min,
                "max_chunk_time_seconds": self.worker_time_max,
            },
            "data_funnel": {
                "total_input_files": self.total_input_files,
                "total_images_processed": self.total_images_processed,
                "images_with_decoding_errors": self.images_with_decoding_errors,
                "images_with_no_faces": self.images_with_no_faces,
                "total_faces_detected": self.total_faces_detected,
                "faces_above_size_threshold": self.faces_above_size_threshold,
                "faces_classified": self.faces_classified,
                "faces_with_eyeglasses": self.faces_with_eyeglasses,
                "faces_rejected_as_sunglasses": self.faces_rejected_as_sunglasses,
                "faces_passing_confidence_thresholds": self.faces_passing_confidence_thresholds,
                "final_target_count": self.final_target_count,
            },
            # Face count diagnostics for investigating detection patterns
            "face_count_diagnostics": {
                "faces_per_image_distribution": self.faces_per_image_distribution,
                "max_faces_in_single_image": self.max_faces_in_single_image,
                "images_with_multiple_faces": self.images_with_multiple_faces,
                "high_face_count_images_summary": {
                    "count": len(self.high_face_count_images),
                    "threshold_used": 5,  # Images with >5 faces are considered "high"
                }
            },
            "config": self.config_snapshot,
        }

    def save_metadata(self, output_dir: Path, aggregated_metrics: dict = None):
        """
        Saves a subset of run metadata to 'run_metadata.json'.
        This file acts as a state-passing mechanism between the main processing
        script and the artifact generation script.
        """
        self.finalize() # Ensure all calculations are done before saving
        metadata = {
            "run_summary": {
                "run_id": self.run_id,
                "git_commit_hash": self.git_commit_hash,
                "run_command": self.run_command,
                "environment": self.environment,
            },
            "worker_performance": {
                "worker_processing_times": self.worker_processing_times,
                "avg_chunk_time_seconds": self.worker_time_avg,
                "std_chunk_time_seconds": self.worker_time_std,
                "min_chunk_time_seconds": self.worker_time_min,
                "max_chunk_time_seconds": self.worker_time_max,
            },
            "config": self.config_snapshot,
            "start_time": self.start_time,
            "aggregated_metrics": aggregated_metrics or {},
        }
        output_path = output_dir / "run_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Run metadata saved to {output_path}")
