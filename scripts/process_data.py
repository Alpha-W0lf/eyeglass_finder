import argparse
import os
from tqdm import tqdm
from functools import partial
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
from dataclasses import asdict

# Ensure src on path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import MetricsManager
from src.data_processing.loader import stream_data_generator
from src.data_processing.utils import get_total_chunks
from src.processing.worker import initialize_worker, process_chunk_of_data
from src.utils.config import load_config, AppConfig
from src.utils.logging_setup import setup_logging, get_logger
from src.modeling.model_loader import load_glasses_classifiers, load_face_detector
from src.utils.monitoring import ResourceMonitor


def get_git_commit_hash() -> str:
    return os.environ.get("GIT_COMMIT_HASH", "unknown")


def get_run_command() -> str:
    return " ".join(sys.argv)


def get_environment() -> dict:
    return {
        "python_version": sys.version.split(" ")[0],
    }


def process_images(config: AppConfig, logger, metrics: MetricsManager):
    logger.info("Starting image processing pipeline (Phase 1: Artifact Generation)...")

    input_dir = Path(config.paths.input_dir)
    file_pattern = config.data_processing.file_pattern
    num_workers = config.hardware.max_workers

    logger.info(f"Streaming data from {input_dir} using pattern '{file_pattern}'...")
    data_files = sorted(list(input_dir.glob(f"**/{file_pattern}")))
    total_images_processed = sum(pq.ParquetFile(f).metadata.num_rows for f in data_files)

    chunk_size = config.data_processing.chunk_size
    data_generator = stream_data_generator(data_files=data_files, chunk_size=chunk_size)

    logger.info(f"Setting up multiprocessing pool with {num_workers} workers.")
    
    # The initializer function is called once for each worker process when it starts.
    # We use a partial function to pass the config dictionary to the initializer.
    initializer = partial(initialize_worker, asdict(config))
    
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=initializer,
        mp_context=mp.get_context("spawn")
    )
    
    total_worker_input_images = 0
    total_faces_detected = 0
    total_failed_chunks = 0
    valid_faces = []
    diagnostics = []
    
    try:
        futures = {
            executor.submit(process_chunk_of_data, chunk)
            for chunk in data_generator
        }
        
        with tqdm(total=total_images_processed, desc="Processing Chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                try:
                    num_processed, num_faces, faces, diags = future.result()
                    
                    total_worker_input_images += num_processed
                    total_faces_detected += num_faces
                    valid_faces.extend(faces)
                    diagnostics.extend(diags)
                    
                except Exception as e:
                    logger.error(f"A worker process failed: {e}", exc_info=True)
                    total_failed_chunks += 1
                finally:
                    pbar.update(1)
                
    finally:
        logger.info("Main process is shutting down. Waiting for workers to terminate...")
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("All worker processes have been terminated.")

    logger.info(f"--- Phase 1: Artifact Generation Summary ---")
    logger.info(f"Total images processed: {total_images_processed}")
    logger.info(f"Total faces detected: {total_faces_detected}")
    logger.info(f"Number of valid faces found: {len(valid_faces)}")
    logger.info(f"Number of diagnostic entries: {len(diagnostics)}")
    logger.info(f"Number of failed chunks: {total_failed_chunks}")

    # Save the collected data to a parquet file
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.paths.output_filename

    if valid_faces:
        results_df = pd.DataFrame(valid_faces)
        results_df.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved {len(valid_faces)} results to {output_path}")
    else:
        logger.warning("No valid faces were found. Skipping parquet file creation.")
        
    # Finalize metrics and save metadata
    metrics.total_images_processed = total_images_processed
    metrics.total_faces_detected = total_faces_detected
    metrics.final_target_count = len(valid_faces)
    
    metrics.save_metadata(output_dir)
    
    logger.info("Phase 1: Artifact Generation finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Process images to find faces with eyeglasses.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{run_timestamp}"
    output_root = Path(config.paths.output_dir) / run_id

    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    config.paths.output_dir = str(output_root)
    config.run_id = run_id
    setattr(config.paths, 'logs_dir', str(logs_dir))

    setup_logging(log_level="INFO", log_file=logs_dir / "run.log")
    logger = get_logger()

    logger.info("Pre-warming model cache in main process...")
    load_face_detector(config, device="cpu")
    load_glasses_classifiers(device="cpu")
    logger.info("Model cache is ready.")

    metrics = MetricsManager(
        run_id=run_id,
        run_command=get_run_command(),
        git_commit_hash=get_git_commit_hash(),
        environment=get_environment(),
        config_snapshot=asdict(config),
    )

    monitor = ResourceMonitor(interval=1)
    try:
        monitor.start()
        process_images(config, logger, metrics)
    except Exception as e:
        logger.critical(f"An unhandled error occurred in the main process: {e}")
        raise e
    finally:
        monitor.stop()
        monitor.save_to_json(Path(config.paths.output_dir) / "resource_utilization.json")


if __name__ == "__main__":
    main()
