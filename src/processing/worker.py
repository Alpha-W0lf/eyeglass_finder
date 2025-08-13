"""Defines the functions that run inside each parallel worker process.

This module contains the logic that is executed by each worker in the
multiprocessing pool. It has two key functions:

1.  `initialize_worker`: Called once per process, this function loads the
    necessary models into the memory of that specific worker, ensuring
    models are not reloaded for every data chunk.

2.  `process_chunk_of_data`: The main processing loop for a worker. It
    takes a chunk of data, extracts the images, and orchestrates the
    calls to the lower-level detection and classification functions
    (defined in `src.processing.pipeline`).
"""
from functools import partial
from pathlib import Path
from typing import Dict, List
import gc
import io
import os
import time
import traceback

import pandas as pd
from loguru import logger
from PIL import Image

from src.modeling.model_loader import load_face_detector, load_glasses_classifiers
from src.processing.pipeline import detect_and_crop_faces, classify_faces
from src.utils.logging import setup_logging
from src.utils.device import get_best_available_device

# Globals per worker process
g_worker_config: Dict = None
g_face_detector = None
g_glasses_classifiers: Dict[str, object] = None
g_inference_lock = None


def initialize_worker(config: Dict, lock):
    """Initializes models and logging for a single worker process."""
    global g_worker_config, g_face_detector, g_glasses_classifiers, g_inference_lock

    g_worker_config = config
    g_inference_lock = lock

    # Configure logging for this worker process
    setup_logging(log_level=config["logging"]["level"])  # file sink optional

    device = get_best_available_device()

    g_face_detector = load_face_detector(config, device=device)
    g_glasses_classifiers = load_glasses_classifiers(device=device)

    logger.info(f"Worker process {os.getpid()} initialized on device '{device}'.")


def process_chunk_of_data(
    chunk: pd.DataFrame, config: dict
) -> tuple[list[dict] | dict, dict, int, float, list, list, list]:
    """
    Main processing function executed by each worker.
    """
    start_time = time.time()
    try:
        pid = os.getpid()
        num_images_in_chunk = len(chunk)
        logger.info(
            f"WORKER_START: Worker {pid} received a chunk of {num_images_in_chunk} rows."
        )

        if not isinstance(chunk, pd.DataFrame):
            chunk = pd.DataFrame.from_records([chunk])

        if not g_face_detector or not g_glasses_classifiers:
            logger.warning("Worker models not initialized, running initializer...")
            initialize_worker(config, None)

        chunk_metrics = {
            "images_processed": len(chunk),
            "decoding_errors": 0,
            "images_with_no_faces": 0,
            "total_faces_detected": 0,
            "confidence_filtered_count": 0,
            "size_filtered_count": 0,
            "faces_classified_count": 0,
            "final_target_count": 0,
            "sunglasses_count": 0,
            "classification_error_count": 0,
        }

        image_bytes_batch = []
        worker_format_errors = 0
        for item in chunk["image"]:
            if isinstance(item, dict) and "bytes" in item:
                image_bytes_batch.append(item["bytes"])  # WIT format
            elif isinstance(item, bytes):
                image_bytes_batch.append(item)  # Direct bytes format
            else:
                logger.warning(
                    f"Skipping an item in 'image' column of unknown type: {type(item)}"
                )
                worker_format_errors += 1

        image_metadatas = chunk[["image_url", "source_file"]].to_dict("records")

        if not image_bytes_batch:
            logger.warning(
                f"Worker {pid}: No valid image bytes found in the chunk. Skipping."
            )
            del image_bytes_batch, image_metadatas
            gc.collect()
            return [], chunk_metrics, num_images_in_chunk, time.time() - start_time, [], [], []

        # Stage 1: Detect and crop faces
        start_time_det = time.time()
        (
            cropped_faces,
            detection_metrics,
            raw_detection_count,
            confidence_scores,
            faces_per_image_stats,
            high_face_count_images,
        ) = detect_and_crop_faces(
            image_bytes_batch, image_metadatas, g_face_detector, config, g_inference_lock
        )
        detection_time = time.time() - start_time_det

        pipeline_decoding_errors = detection_metrics.get("decoding_errors", 0)
        total_decoding_errors = worker_format_errors + pipeline_decoding_errors

        chunk_metrics.update(detection_metrics)
        chunk_metrics["decoding_errors"] = total_decoding_errors
        chunk_metrics["total_faces_detected"] = raw_detection_count

        if not cropped_faces:
            del image_bytes_batch, image_metadatas
            gc.collect()
            return [], chunk_metrics, num_images_in_chunk, time.time() - start_time, [], [], []

        detection_time_per_face = detection_time / len(cropped_faces) if cropped_faces else 0
        for face in cropped_faces:
            face["detection_time_seconds"] = detection_time_per_face

        # Stage 2: Classify
        start_time_cls = time.time()
        classified_faces, classification_metrics = classify_faces(
            cropped_faces,
            g_glasses_classifiers["eyeglasses"],
            g_glasses_classifiers["sunglasses"],
            config,
        )
        classification_time = time.time() - start_time_cls
        chunk_metrics.update(classification_metrics)

        classification_time_per_face = (
            classification_time / len(classified_faces) if classified_faces else 0
        )
        for face in classified_faces:
            face["classification_time_seconds"] = classification_time_per_face

        batch_results = classified_faces
        final_count = sum(1 for f in batch_results if f.get("is_target"))
        chunk_metrics["final_target_count"] = final_count

        processing_time = time.time() - start_time
        del image_bytes_batch, image_metadatas, cropped_faces, classified_faces
        gc.collect()

        return (
            batch_results,
            chunk_metrics,
            num_images_in_chunk,
            processing_time,
            confidence_scores,
            faces_per_image_stats,
            high_face_count_images,
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Worker {os.getpid()} CRASHED: {e}\n{tb_str}")

        error_info = {"error": str(e), "traceback": tb_str, "worker_id": os.getpid()}
        gc.collect()
        return error_info, {}, 0, time.time() - start_time, [], [], []
