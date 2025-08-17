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
from typing import Dict, List, Tuple
import gc
import io
import os
import time
import traceback
import psutil

import pandas as pd
from loguru import logger

from src.processing.pipeline import (
    initialize_pipeline_worker,
    process_images,
)
from src.utils.config import AppConfig, config_from_dict
from src.utils.logging_setup import configure_worker_logging

# A global variable to hold the configuration for the worker process.
# Using a global variable is a common pattern in multiprocessing to avoid
# passing the config object repeatedly for every task.
g_worker_config: AppConfig = None


def _log_worker_memory_usage(stage: str):
    """Logs the current memory usage of the worker process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(
        f"WORKER_MEMORY_PROFILE ({os.getpid()}): Stage: {stage} - RSS: {mem_info.rss / 1024**2:.2f} MB"
    )


def initialize_worker(config_dict: dict):
    """
    Initializes the worker process.
    - Configures logging for the worker.
    - Stores the application config in a global variable for this process.
    - Initializes the pipeline models (FaceDetector, GlassesDetector).
    """
    global g_worker_config
    
    # Set up the logger for this specific worker process.
    # This ensures that logs from different workers can be distinguished.
    configure_worker_logging()
    
    # Store the config in a global variable for this worker process.
    # This avoids having to pass the config object with every task.
    g_worker_config = config_dict
    
    try:
        # Initialize the pipeline components (models) for this worker.
        # This is a potentially time-consuming operation, so it's done once
        # when the worker process is created.
        initialize_pipeline_worker(g_worker_config)
    except Exception as e:
        # If model loading fails, log a critical error and re-raise.
        # This will cause the worker process to fail, which is the desired
        # behavior if it cannot be initialized correctly.
        logger.critical(f"Worker {os.getpid()}: Failed to initialize pipeline worker: {e}", exc_info=True)
        raise


def process_chunk_of_data(image_df: pd.DataFrame) -> Tuple[int, int, List[Dict], List[Dict], List[Dict], float, float]:
    """
    Processes a chunk of image data.
    This function is the main entry point for a worker process task.
    
    Args:
        image_df: A pandas DataFrame containing image data.
    
    Returns:
        A tuple with statistics and results for the processed chunk.
    """
    if g_worker_config is None:
        raise RuntimeError("Worker config not initialized.")
        
    chunk_id = f"Worker-{os.getpid()}"
    num_images = len(image_df)
    _log_worker_memory_usage(f"Chunk Start")
    logger.info(f"WORKER_START: Worker {os.getpid()} received a chunk of {num_images} rows.")

    try:
        # The core processing logic is delegated to the pipeline module.
        valid_faces, diagnostics, high_face_images, detect_time_s, classify_time_s = process_images(image_df)
        
        # After processing, calculate some basic statistics.
        total_faces_detected = len(valid_faces)
        images_with_faces = len(set(d["image_id"] for d in valid_faces))

        _log_worker_memory_usage(f"Chunk End")
        
        # Return statistics and detailed results for aggregation in the main process.
        return num_images, total_faces_detected, valid_faces, diagnostics, high_face_images, detect_time_s, classify_time_s
    
    except Exception:
        logger.error(f"Worker {os.getpid()} CRASHED.", exc_info=True)
        # In case of a crash, return empty results and stats so the main
        # process can continue aggregating results from other workers.
        return num_images, 0, [], [], [], 0.0, 0.0
    finally:
        # Perform garbage collection at the end of each task to release memory.
        gc.collect()
