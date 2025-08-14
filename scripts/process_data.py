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
from src.utils.logging import setup_logging, get_logger
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


def process_images(config: AppConfig, logger, lock, metrics: MetricsManager):
    logger.info("Starting image processing pipeline (Phase 1: Artifact Generation)...")

    input_dir = Path(config.paths.input_dir)
    file_pattern = config.data_processing.file_pattern
    num_workers = config.execution.num_workers

    logger.info(f"Streaming data from {input_dir} using pattern '{file_pattern}'...")
    data_files = sorted(list(input_dir.glob(f"**/{file_pattern}")))
    total_images_processed = sum(pq.ParquetFile(f).metadata.num_rows for f in data_files)

    chunk_size = config.data_processing.chunk_size
    data_generator = stream_data_generator(data_files=data_files, chunk_size=chunk_size)

    logger.info(f"Setting up multiprocessing pool with {num_workers} workers.")
    processing_func = partial(process_chunk_of_data, config=config)

    all_results = []
    worker_times = []
    all_confidence_scores = []
    aggregated_metrics = {}
    all_faces_per_image_stats = []
    all_high_face_count_images = []
    total_worker_input_images = 0
    total_diagnostic_entries_received = 0

    total_chunks = get_total_chunks(data_files, chunk_size)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(config, lock),
    ) as executor:
        futures = []
        for chunk in data_generator:
            futures.append(executor.submit(processing_func, chunk))

        with tqdm(total=total_chunks, desc="Processing Chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result[0], dict) and 'error' in result[0]:
                    error_info = result[0]
                    logger.error(f"--- FATAL ERROR IN WORKER PROCESS {error_info['worker_id']} ---")
                    logger.error(f"Error: {error_info['error']}")
                    logger.error("--- Full Traceback ---")
                    logger.error(f"\n{error_info['traceback']}")
                    logger.error("----------------------")
                    raise Exception(f"Worker process {error_info['worker_id']} failed. See logs for details.")

                chunk_results, chunk_metrics, num_processed, worker_time, confidence_scores, faces_per_image_stats, high_face_count_images = result
                total_worker_input_images += num_processed
                total_diagnostic_entries_received += len(faces_per_image_stats)
                if chunk_results:
                    all_results.extend(chunk_results)
                worker_times.append(worker_time)
                all_confidence_scores.extend(confidence_scores)
                all_faces_per_image_stats.extend(faces_per_image_stats)
                all_high_face_count_images.extend(high_face_count_images)
                for key, value in chunk_metrics.items():
                    aggregated_metrics[key] = aggregated_metrics.get(key, 0) + value
                pbar.update(1)
                pbar.set_postfix_str(f"Last chunk took {worker_time:.2f}s")

    output_dir = config.paths.output_dir
    output_path = Path(output_dir) / "annotated_faces.parquet"
    logger.info(f"All images processed. Found a total of {aggregated_metrics.get('final_target_count', 0)} target faces after filtering.")

    logger.info(f"MAIN_TRACKING: FINAL VERIFICATION - Total worker input images: {total_worker_input_images}, Total diagnostic entries: {total_diagnostic_entries_received}")
    if total_diagnostic_entries_received != total_worker_input_images:
        logger.error(f"MAIN_TRACKING: DATA INTEGRITY ERROR - Missing {total_worker_input_images - total_diagnostic_entries_received} diagnostic entries!")
        logger.error(f"MAIN_TRACKING: Expected: {total_worker_input_images}, Got: {total_diagnostic_entries_received}")
    else:
        logger.info("MAIN_TRACKING: DATA INTEGRITY OK - All worker images tracked in diagnostics")

    logger.info(f"MAIN_TRACKING: Dataset verification - Expected total images: {total_images_processed}, Worker reported: {total_worker_input_images}")
    if total_worker_input_images != total_images_processed:
        logger.error(f"MAIN_TRACKING: DATASET SIZE MISMATCH - Expected {total_images_processed}, workers processed {total_worker_input_images}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.drop(columns=['cropped_face'], errors='ignore')
        logger.info(f"Saving results to {output_path}...")
        results_df.to_parquet(output_path, index=False)
    else:
        logger.warning("No faces were processed. Skipping parquet file creation.")

    aggregated_metrics["total_images_processed"] = total_images_processed
    aggregated_metrics["face_confidence_scores"] = all_confidence_scores

    face_count_distribution = {}
    max_faces_in_image = 0
    images_with_multiple_faces = 0
    for image_stats in all_faces_per_image_stats:
        num_faces = image_stats['num_faces']
        face_count_distribution[num_faces] = face_count_distribution.get(num_faces, 0) + 1
        max_faces_in_image = max(max_faces_in_image, num_faces)
        if num_faces > 1:
            images_with_multiple_faces += 1
    aggregated_metrics["face_count_diagnostics"] = {
        "faces_per_image_distribution": face_count_distribution,
        "max_faces_in_single_image": max_faces_in_image,
        "images_with_multiple_faces": images_with_multiple_faces,
        "high_face_count_images_count": len(all_high_face_count_images),
        "faces_per_image_stats": all_faces_per_image_stats,
        "high_face_count_images": [
            {k: v for k, v in img.items() if k != 'image_bytes'} for img in all_high_face_count_images
        ],
    }

    if all_high_face_count_images:
        import pickle
        high_face_images_path = Path(output_dir) / "high_face_count_images.pkl"
        with open(high_face_images_path, 'wb') as f:
            pickle.dump(all_high_face_count_images, f)
        logger.info(f"Saved {len(all_high_face_count_images)} high face count images to {high_face_images_path}")

    logger.info(
        f"Face count diagnostics: Max faces in single image: {max_faces_in_image}, Images with multiple faces: {images_with_multiple_faces}, High face count images (>5): {len(all_high_face_count_images)}"
    )

    metrics.worker_processing_times = worker_times
    metrics.save_metadata(Path(output_dir), aggregated_metrics)
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

    manager = mp.Manager()
    lock = manager.Lock()

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
        process_images(config, logger, lock, metrics)
    except Exception as e:
        logger.critical(f"An unhandled error occurred in the main process: {e}")
        raise e
    finally:
        monitor.stop()
        monitor.save_to_json(Path(config.paths.output_dir) / "resource_utilization.json")


if __name__ == "__main__":
    main()
