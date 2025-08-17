import argparse
import os
from tqdm import tqdm
from functools import partial
from pathlib import Path
from datetime import datetime
import sys
import time
import pandas as pd
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
from dataclasses import asdict

# Ensure src on path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import MetricsManager
from src.reporting.report_generator import generate_report
from src.utils.visualizations import (
    plot_face_size_distribution,
    plot_confidence_vs_face_size,
    plot_image_mode_distribution,
    plot_confidence_histogram,
    plot_resource_utilization,
    plot_worker_performance_histogram,
    plot_face_count_distribution,
)
from scripts.generate_run_artifacts import (
    create_qualitative_samples,
    create_high_face_count_samples,
    generate_final_dataset,
)
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
    # Ramp-up: optionally override initial chunk size during warmup
    ramp_en = getattr(config.performance, 'rampup_enabled', False)
    warmup_k = int(getattr(config.performance, 'rampup_warmup_chunks', 0) or 0)
    init_chunk_override = getattr(config.performance, 'rampup_initial_chunk_size_override', None)

    data_generator = stream_data_generator(
        data_files=data_files,
        chunk_size=(init_chunk_override if (ramp_en and warmup_k > 0 and init_chunk_override) else chunk_size)
    )

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
    high_face_images_all = []
    
    try:
        futures = {}
        submitted = 0
        completed = 0
        stagger_ms = int(getattr(config.performance, 'rampup_stagger_worker_submissions_ms', 0) or 0)
        prefetch_steady = int(getattr(config.performance, 'prefetch_chunks', 4) or 1)
        prefetch_warm = int(getattr(config.performance, 'rampup_initial_prefetch_chunks', prefetch_steady) or prefetch_steady)

        chunks_iter = iter(data_generator)

        def current_window() -> int:
            if ramp_en and warmup_k > 0 and submitted < warmup_k:
                return max(1, prefetch_warm)
            return max(1, prefetch_steady)

        # Prime initial window
        try:
            while len(futures) < current_window():
                chunk = next(chunks_iter)
                fut = executor.submit(process_chunk_of_data, chunk)
                futures[fut] = time.perf_counter()
                submitted += 1
                if ramp_en and warmup_k > 0 and submitted <= warmup_k and stagger_ms > 0:
                    time.sleep(stagger_ms / 1000.0)
        except StopIteration:
            pass

        with tqdm(total=total_images_processed, desc="Processing Chunks", unit="chunk") as pbar:
            while futures:
                # Wait for any future to complete
                for future in as_completed(list(futures.keys()), timeout=None):
                    start_ts = futures.pop(future, None)
                    try:
                        num_processed, num_faces, faces, diags, high_face_images, detect_time_s, classify_time_s = future.result()
                        total_worker_input_images += num_processed
                        total_faces_detected += num_faces
                        valid_faces.extend(faces)
                        diagnostics.extend(diags)
                        if high_face_images:
                            high_face_images_all.extend(high_face_images)
                        metrics.total_detection_time_seconds += float(detect_time_s)
                        metrics.total_classification_time_seconds += float(classify_time_s)
                        if start_ts is not None:
                            duration = time.perf_counter() - start_ts
                            metrics.worker_processing_times.append(duration)
                    except Exception as e:
                        logger.error(f"A worker process failed: {e}", exc_info=True)
                        total_failed_chunks += 1
                    finally:
                        pbar.update(1)
                        completed += 1

                    # After consuming one, try to keep window filled
                    try:
                        while len(futures) < current_window():
                            chunk = next(chunks_iter)
                            fut = executor.submit(process_chunk_of_data, chunk)
                            futures[fut] = time.perf_counter()
                            submitted += 1
                            if ramp_en and warmup_k > 0 and submitted <= warmup_k and stagger_ms > 0:
                                time.sleep(stagger_ms / 1000.0)
                    except StopIteration:
                        pass
                    break
                try:
                    num_processed, num_faces, faces, diags, high_face_images, detect_time_s, classify_time_s = future.result()
                    
                    total_worker_input_images += num_processed
                    total_faces_detected += num_faces
                    valid_faces.extend(faces)
                    diagnostics.extend(diags)
                    if high_face_images:
                        high_face_images_all.extend(high_face_images)
                    # accumulate timing
                    metrics.total_detection_time_seconds += float(detect_time_s)
                    metrics.total_classification_time_seconds += float(classify_time_s)
                    # Record worker processing time
                    start_ts = futures.get(future)
                    if start_ts is not None:
                        duration = time.perf_counter() - start_ts
                        metrics.worker_processing_times.append(duration)
                    
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

    # Save the collected data to the annotated faces artifact
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated_faces.parquet"

    if valid_faces:
        results_df = pd.DataFrame(valid_faces)
        results_df.to_parquet(annotated_path, index=False)
        logger.info(f"Successfully saved {len(valid_faces)} annotated faces to {annotated_path}")
    else:
        logger.warning("No valid faces were found. Skipping annotated_faces parquet creation.")
        
    # Finalize metrics and save metadata
    metrics.total_images_processed = total_images_processed
    metrics.total_faces_detected = total_faces_detected
    metrics.final_target_count = len([f for f in valid_faces if f.get("is_target")])

    # Build aggregated metrics expected by artifact generator
    try:
        # Diagnostics-based counts
        reason_counts = {}
        for d in diagnostics:
            r = d.get("reason")
            if r:
                reason_counts[r] = reason_counts.get(r, 0) + 1

        images_with_decoding_errors = reason_counts.get("image_decoding_error", 0)
        images_with_no_faces = reason_counts.get("no_faces_detected", 0)

        # Faces per image distribution
        faces_per_image = {}
        if valid_faces:
            df_faces = pd.DataFrame(valid_faces)
            if "image_id" in df_faces.columns:
                counts = df_faces.groupby("image_id").size().astype(int)
                # include zeros for images with no faces
                all_ids = set(range(total_images_processed))
                present_ids = set(counts.index.tolist())
                zero_ids = all_ids - present_ids
                zero_series = pd.Series(0, index=sorted(list(zero_ids)))
                combined = pd.concat([counts, zero_series])
                # build histogram: value (num_faces) -> frequency
                hist = combined.value_counts().sort_index()
                faces_per_image = {str(int(k)): int(v) for k, v in hist.items()}

        faces_above_size_threshold = len(valid_faces)
        faces_classified = len(valid_faces)
        faces_with_eyeglasses = sum(1 for f in valid_faces if f.get("is_target"))
        faces_rejected_as_sunglasses = sum(1 for f in valid_faces if f.get("sunglasses_prediction"))

        face_count_diagnostics = {
            "faces_per_image_distribution": faces_per_image,
            "max_faces_in_single_image": max([int(v) for v in faces_per_image.values()], default=0),
            "images_with_multiple_faces": sum(1 for v in faces_per_image.values() if int(v) > 1),
            "high_face_count_images": [],
        }

        aggregated_metrics = {
            "total_images_processed": int(total_images_processed),
            "images_with_decoding_errors": int(images_with_decoding_errors),
            "images_with_no_faces": int(images_with_no_faces),
            "total_faces_detected": int(total_faces_detected),
            "faces_above_size_threshold": int(faces_above_size_threshold),
            "faces_classified": int(faces_classified),
            "faces_with_eyeglasses": int(faces_with_eyeglasses),
            "faces_rejected_as_sunglasses": int(faces_rejected_as_sunglasses),
            "final_target_count": int(metrics.final_target_count),
            "face_count_diagnostics": face_count_diagnostics,
        }
    except Exception as e:
        logger.error(f"Failed to build aggregated metrics: {e}")
        aggregated_metrics = {}

    metrics.save_metadata(output_dir, aggregated_metrics=aggregated_metrics)

    # Persist high face count images for diagnostic sampling
    try:
        if high_face_images_all:
            import pickle
            with open(output_dir / "high_face_count_images.pkl", 'wb') as f:
                pickle.dump(high_face_images_all, f)
            logger.info(f"Saved high face count image records: {len(high_face_images_all)}")
    except Exception as e:
        logger.warning(f"Failed to save high face count images pkl: {e}")

    # Auto-generate rich artifacts (Stage 2) inline for a complete run folder
    try:
        if valid_faces:
            df = pd.DataFrame(valid_faces)
            # Final dataset (targets only)
            generate_final_dataset(df, output_dir)
            # Qualitative samples
            create_qualitative_samples(df, output_dir, config)
            # High face count samples (optional, requires pkl)
            create_high_face_count_samples({}, output_dir)

            # Visualizations
            visualizations_dir = output_dir / "visualizations"
            visualizations_dir.mkdir(exist_ok=True)
            # Face-related plots
            if "face_score" in df.columns:
                plot_confidence_histogram(df["face_score"].tolist(), visualizations_dir / "face_confidence_histogram.png")
            plot_face_size_distribution(df, visualizations_dir / "face_size_distribution.png")
            plot_confidence_vs_face_size(df, visualizations_dir / "confidence_vs_face_size.png")
            plot_image_mode_distribution(df, visualizations_dir / "image_mode_distribution.png")

            # Face count distribution from aggregated metrics
            fdist = aggregated_metrics.get("face_count_diagnostics", {}).get("faces_per_image_distribution", {})
            if fdist:
                plot_face_count_distribution(fdist, visualizations_dir / "face_count_distribution.png")

            # Resource utilization plot
            resource_data_path = output_dir / "resource_utilization.json"
            if resource_data_path.exists():
                with open(resource_data_path, 'r') as f:
                    resource_data = json.load(f)
                plot_resource_utilization(resource_data, visualizations_dir / "resource_utilization.png")

            # Worker performance histogram
            if metrics.worker_processing_times:
                plot_worker_performance_histogram(metrics.worker_processing_times, visualizations_dir / "worker_performance.png")

            # Hydrate metrics from aggregated metrics for accurate report
            try:
                metrics.images_with_decoding_errors = aggregated_metrics.get("images_with_decoding_errors", 0)
                metrics.images_with_no_faces = aggregated_metrics.get("images_with_no_faces", 0)
                metrics.faces_above_size_threshold = aggregated_metrics.get("faces_above_size_threshold", 0)
                metrics.faces_classified = aggregated_metrics.get("faces_classified", 0)
                metrics.faces_with_eyeglasses = aggregated_metrics.get("faces_with_eyeglasses", 0)
                metrics.faces_rejected_as_sunglasses = aggregated_metrics.get("faces_rejected_as_sunglasses", 0)
                face_diag = aggregated_metrics.get("face_count_diagnostics", {})
                metrics.faces_per_image_distribution = face_diag.get("faces_per_image_distribution", {})
                metrics.max_faces_in_single_image = face_diag.get("max_faces_in_single_image", 0)
                metrics.images_with_multiple_faces = face_diag.get("images_with_multiple_faces", 0)
                if not df.empty and "face_score" in df.columns:
                    metrics.face_confidence_scores = df["face_score"].tolist()
            except Exception as e:
                logger.warning(f"Failed to hydrate metrics from aggregated metrics: {e}")
            metrics.finalize()

            # Report
            report_path = output_dir / "report.md"
            generate_report(metrics, report_path)
            logger.info("Inline artifact generation complete.")
        else:
            logger.warning("Skipping inline artifact generation: no annotated faces available.")
    except Exception as e:
        logger.error(f"Artifact generation failed: {e}", exc_info=True)
    
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
        # Post-process: ensure resource utilization plot exists and refresh report with hydrated metrics
        try:
            run_dir = Path(config.paths.output_dir)
            viz_dir = run_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            resource_data_path = run_dir / "resource_utilization.json"
            if resource_data_path.exists():
                with open(resource_data_path, 'r') as f:
                    resource_data = json.load(f)
                plot_resource_utilization(resource_data, viz_dir / "resource_utilization.png")
            # Rehydrate metrics from saved metadata before regenerating report
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    md = json.load(f)
                agg = md.get("aggregated_metrics", {})
                metrics.total_images_processed = agg.get("total_images_processed", metrics.total_images_processed)
                metrics.images_with_decoding_errors = agg.get("images_with_decoding_errors", 0)
                metrics.images_with_no_faces = agg.get("images_with_no_faces", 0)
                metrics.total_faces_detected = agg.get("total_faces_detected", metrics.total_faces_detected)
                metrics.faces_above_size_threshold = agg.get("faces_above_size_threshold", 0)
                metrics.faces_classified = agg.get("faces_classified", 0)
                metrics.faces_with_eyeglasses = agg.get("faces_with_eyeglasses", 0)
                metrics.faces_rejected_as_sunglasses = agg.get("faces_rejected_as_sunglasses", 0)
                metrics.final_target_count = agg.get("final_target_count", metrics.final_target_count)
                face_diag = agg.get("face_count_diagnostics", {})
                metrics.faces_per_image_distribution = face_diag.get("faces_per_image_distribution", {})
                metrics.max_faces_in_single_image = face_diag.get("max_faces_in_single_image", 0)
                metrics.images_with_multiple_faces = face_diag.get("images_with_multiple_faces", 0)
                # Also hydrate worker performance
                worker_perf = md.get("worker_performance", {})
                metrics.worker_processing_times = worker_perf.get("worker_processing_times", metrics.worker_processing_times)
                metrics.worker_time_avg = worker_perf.get("avg_chunk_time_seconds", metrics.worker_time_avg)
                metrics.worker_time_std = worker_perf.get("std_chunk_time_seconds", metrics.worker_time_std)
                metrics.worker_time_min = worker_perf.get("min_chunk_time_seconds", metrics.worker_time_min)
                metrics.worker_time_max = worker_perf.get("max_chunk_time_seconds", metrics.worker_time_max)
                metrics.finalize()
                # Regenerate report with hydrated values
                report_path = run_dir / "report.md"
                generate_report(metrics, report_path)
        except Exception as e:
            logger.error(f"Post-processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
