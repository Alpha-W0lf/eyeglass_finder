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
from src.utils.memory_manager import MemoryPoolManager
from src.utils.production_logger import ProductionLogger


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
    # Memory manager to proactively throttle prefetch under pressure
    mem_thresh = int(getattr(config.observability.performance_alerts, 'memory_threshold', 85) or 85)
    memory_manager = MemoryPoolManager(pressure_percent_threshold=mem_thresh, min_available_gb=4.0)
    
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
        run_start_ts = time.perf_counter()
        prod_logger = ProductionLogger(Path(config.paths.output_dir) / "logs" / "production.jsonl")
        prod_logger.log_processing_milestone("run_start", images_processed=0, elapsed_seconds=0.0, config_snapshot=metrics.config_snapshot)
        futures = {}
        submitted = 0
        completed = 0
        stagger_ms = int(getattr(config.performance, 'rampup_stagger_worker_submissions_ms', 0) or 0)
        prefetch_steady = int(getattr(config.performance, 'prefetch_chunks', 4) or 1)
        prefetch_warm = int(getattr(config.performance, 'rampup_initial_prefetch_chunks', prefetch_steady) or prefetch_steady)

        chunks_iter = iter(data_generator)

        def current_window() -> int:
            if ramp_en and warmup_k > 0 and submitted < warmup_k:
                desired = max(1, prefetch_warm)
            else:
                desired = max(1, prefetch_steady)
            return memory_manager.recommend_prefetch_window(desired)

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

        with tqdm(total=total_images_processed, desc="Processing Chunks", unit="img") as pbar:
            while futures:
                # Wait for any future to complete
                for future in as_completed(list(futures.keys()), timeout=None):
                    fut_start_ts = futures.pop(future, None)
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
                        if fut_start_ts is not None:
                            duration = time.perf_counter() - fut_start_ts
                            metrics.worker_processing_times.append(duration)
                    except Exception as e:
                        logger.error(f"A worker process failed: {e}", exc_info=True)
                        total_failed_chunks += 1
                        num_processed = 0
                    finally:
                        # Progress by number of images processed in this future
                        pbar.update(max(0, int(num_processed)))
                        completed += 1
                        # Emit lightweight progress snapshot
                        prod_logger.log_event(
                            "progress_snapshot",
                            level="INFO",
                            submitted=submitted,
                            completed=completed,
                            total_images_processed=total_worker_input_images,
                        )

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

                    # Under memory pressure, opportunistically yield
                    if memory_manager.is_memory_pressure():
                        import gc
                        gc.collect()
                        time.sleep(0.05)
                
    finally:
        logger.info("Main process is shutting down. Waiting for workers to terminate...")
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("All worker processes have been terminated.")
        try:
            # Finalize production log
            elapsed = time.perf_counter() - run_start_ts
            ips = (total_worker_input_images / elapsed) if elapsed > 0 else 0.0
            prod_logger.log_processing_milestone("run_end", images_processed=total_worker_input_images, elapsed_seconds=elapsed, images_per_second=ips)
        except Exception:
            pass

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
        reason_counts: dict[str, int] = {}
        per_image_face_counts: dict[int, int] = {}
        for d in diagnostics:
            r = d.get("reason")
            if r:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            if r == "faces_detected":
                img_id = int(d.get("image_id")) if d.get("image_id") is not None else None
                cnt = int(d.get("count", 0))
                if img_id is not None:
                    per_image_face_counts[img_id] = cnt

        images_with_decoding_errors = reason_counts.get("image_decoding_error", 0)
        images_with_no_faces = reason_counts.get("no_faces_detected", 0)

        # Faces per image distribution based on raw detections (pre size-threshold)
        faces_per_image = {}
        if per_image_face_counts:
            # Build a Series for all images we have counts for (no_faces are implied by diagnostics)
            # Fill zeros explicitly for images with no faces
            all_ids = set(range(total_images_processed))
            with_counts = set(per_image_face_counts.keys())
            zero_ids = sorted(list(all_ids - with_counts))
            counts_series = pd.Series(per_image_face_counts)
            if images_with_no_faces > 0:
                # Restrict zeros to the expected number of no-face images if any drift exists
                zero_ids = zero_ids[:images_with_no_faces]
            zeros_series = pd.Series(0, index=zero_ids)
            combined = pd.concat([counts_series, zeros_series])
            hist = combined.value_counts().sort_index()
            faces_per_image = {str(int(num_faces)): int(num_images) for num_faces, num_images in hist.items()}

        faces_above_size_threshold = len(valid_faces)
        faces_classified = len(valid_faces)
        faces_with_eyeglasses = sum(1 for f in valid_faces if f.get("is_target"))
        faces_rejected_as_sunglasses = sum(1 for f in valid_faces if f.get("sunglasses_prediction"))
        total_raw_faces_detected = int(sum(per_image_face_counts.values()))

        # Cross-source max faces: from histogram and from high_face_images list (if present)
        max_faces_hist = max([int(k) for k in faces_per_image.keys()], default=0)
        try:
            max_faces_high = max((int(img.get("num_faces", 0)) for img in high_face_images_all), default=0)
        except Exception:
            max_faces_high = 0
        max_faces_final = max(max_faces_hist, max_faces_high)

        # Build top-N images by face count for auditability
        try:
            sorted_high = sorted(high_face_images_all, key=lambda x: int(x.get("num_faces", 0)), reverse=True)
            top_k_records = [
                {
                    "rank": i + 1,
                    "image_id": int(rec.get("image_id")) if rec.get("image_id") is not None else None,
                    "num_faces": int(rec.get("num_faces", 0)),
                    "image_url": rec.get("image_url"),
                }
                for i, rec in enumerate(sorted_high[:50])
            ]
        except Exception:
            top_k_records = []

        face_count_diagnostics = {
            "faces_per_image_distribution": faces_per_image,  # key: num_faces, value: number_of_images
            "max_faces_in_single_image": max_faces_final,
            "images_with_multiple_faces": sum(int(v) for k, v in faces_per_image.items() if int(k) > 1),
            "high_face_count_images": top_k_records,
        }

        aggregated_metrics = {
            "total_images_processed": int(total_images_processed),
            "images_with_decoding_errors": int(images_with_decoding_errors),
            "images_with_no_faces": int(images_with_no_faces),
            # Use raw detection counts for total faces detected (pre size-threshold)
            "total_faces_detected": int(total_raw_faces_detected),
            "faces_above_size_threshold": int(faces_above_size_threshold),
            "faces_classified": int(faces_classified),
            "faces_with_eyeglasses": int(faces_with_eyeglasses),
            "faces_rejected_as_sunglasses": int(faces_rejected_as_sunglasses),
            "final_target_count": int(metrics.final_target_count),
            "face_count_diagnostics": face_count_diagnostics,
            # Persist global maximum directly for robust reporting regardless of diagnostic sampling
            "global_max_faces_in_single_image": int(max_faces_final),
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

    # Set observability flags for downstream modules
    try:
        os.environ["ENABLE_MEMORY_PROFILING"] = "1" if bool(getattr(config.observability, "memory_profiling", False)) else "0"
    except Exception:
        os.environ["ENABLE_MEMORY_PROFILING"] = "0"
    try:
        detailed = bool(getattr(config.observability, "detailed_metrics", False))
        os.environ["ENABLE_VERBOSE_LOADER"] = "1" if detailed else "0"
        os.environ["ENABLE_VERBOSE_WORKER"] = "1" if detailed else "0"
    except Exception:
        os.environ["ENABLE_VERBOSE_LOADER"] = "0"
        os.environ["ENABLE_VERBOSE_WORKER"] = "0"

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
                # Also hydrate global max if present for robust reporting
                metrics.global_max_faces_in_single_image = agg.get("global_max_faces_in_single_image", metrics.max_faces_in_single_image)
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
