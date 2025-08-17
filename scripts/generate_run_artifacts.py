import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
import sys
import json

# Ensure src is on path for standalone execution
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config, AppConfig
from src.utils.logging_setup import setup_logging
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


def find_latest_run_dir(root_output_dir: str) -> Path:
    output_dir = Path(root_output_dir)
    run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {root_output_dir}")
    latest_run = run_dirs[-1]
    logger.info(f"Found latest run directory: {latest_run}")
    return latest_run


def create_qualitative_samples(df: pd.DataFrame, output_dir: Path, config: AppConfig):
    logger.info("Generating qualitative samples...")
    sample_size = config.report_generation.qualitative_analysis_sample_size

    samples_root = output_dir / "qualitative_analysis"
    final_targets_dir = samples_root / "final_targets"
    rejected_sunglasses_dir = samples_root / "rejected_as_sunglasses"
    final_targets_dir.mkdir(parents=True, exist_ok=True)
    rejected_sunglasses_dir.mkdir(parents=True, exist_ok=True)

    target_df = df[df["is_target"] == True]
    if not target_df.empty:
        target_sample = target_df.sample(n=min(sample_size, len(target_df)))
        for idx, row in target_sample.iterrows():
            img_bytes = row["cropped_face_jpeg"]
            with open(final_targets_dir / f"target_{idx}.jpg", "wb") as f:
                f.write(img_bytes)
        logger.info(f"Saved {len(target_sample)} final target samples.")

    rejected_df = df[df["sunglasses_prediction"] == True]
    if not rejected_df.empty:
        rejected_sample = rejected_df.sample(n=min(sample_size, len(rejected_df)))
        for idx, row in rejected_sample.iterrows():
            img_bytes = row["cropped_face_jpeg"]
            with open(rejected_sunglasses_dir / f"rejected_{idx}.jpg", "wb") as f:
                f.write(img_bytes)
        logger.info(f"Saved {len(rejected_sample)} rejected sunglasses samples.")


def create_high_face_count_samples(metadata: dict, output_dir: Path):
    logger.info("Generating high face count diagnostic samples...")
    high_face_images_path = output_dir / "high_face_count_images.pkl"

    if not high_face_images_path.exists():
        logger.info("No high face count images file found. Skipping diagnostic samples.")
        return

    try:
        import pickle
        with open(high_face_images_path, 'rb') as f:
            high_face_count_images = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load high face count images: {e}")
        return

    if not high_face_count_images:
        logger.info("No high face count images found. Skipping diagnostic samples.")
        return

    samples_root = output_dir / "qualitative_analysis"
    high_face_dir = samples_root / "high_face_count_images"
    high_face_dir.mkdir(parents=True, exist_ok=True)

    sorted_images = sorted(high_face_count_images, key=lambda x: x['num_faces'], reverse=True)
    max_samples = min(20, len(sorted_images))

    for i, image_data in enumerate(sorted_images[:max_samples]):
        try:
            image_bytes = image_data['image_bytes']
            num_faces = image_data['num_faces']
            image_url = image_data.get('image_url', 'unknown')
            filename = f"high_faces_{i+1:02d}_count_{num_faces}_faces.jpg"
            with open(high_face_dir / filename, "wb") as f:
                f.write(image_bytes)
            logger.debug(f"Saved high face count image: {filename} ({num_faces} faces)")
        except Exception as e:
            logger.warning(f"Failed to save high face count image {i+1}: {e}")

    logger.info(f"Saved {max_samples} high face count diagnostic samples to {high_face_dir}")

    summary_file = high_face_dir / "README.txt"
    with open(summary_file, "w") as f:
        f.write("High Face Count Images - Diagnostic Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"This folder contains {max_samples} images that had the highest number of detected faces.\n")
        f.write("These images are saved for manual inspection to investigate the unexpectedly high\n")
        f.write("face detection count in the pipeline run.\n\n")
        f.write("Possible explanations:\n")
        f.write("- Group photos (weddings, team photos, conferences)\n")
        f.write("- Crowd scenes (protests, concerts, gatherings)\n")
        f.write("- False positives (artwork, posters, reflections)\n")
        f.write("- Dataset content different from expected pre-filtering\n\n")
        f.write("Image Details:\n")
        f.write("-" * 30 + "\n")
        for i, image_data in enumerate(sorted_images[:max_samples]):
            f.write(f"{i+1:2d}. {image_data['num_faces']:2d} faces - {image_data.get('image_url', 'unknown')}\n")


def generate_final_dataset(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating final filtered dataset...")
    target_df = df[df["is_target"] == True]
    final_columns = {
        "image_url": "image_url",
        "source_file": "source_file",
        "face_bbox": "face_bbox",
        "face_score": "face_confidence",
        "cropped_face_jpeg": "face_jpeg",
    }
    final_df = target_df[final_columns.keys()].rename(columns=final_columns)
    output_path = output_dir / "filtered_dataset.parquet"
    if not final_df.empty:
        final_df.to_parquet(output_path, index=False)
        logger.info(f"Saved final dataset with {len(final_df)} rows to {output_path}")
    else:
        logger.warning("No target faces found. Final dataset is empty.")


def main():
    parser = argparse.ArgumentParser(description="Generate run artifacts from processed data.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--run_dir", default=None, help="Optional: Path to a specific run directory.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run_dir(config.paths.output_dir)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    setup_logging(log_level="INFO", log_file=logs_dir / "artifact_generation.log")

    logger.info(f"Starting artifact generation for run: {run_dir.name}")

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        logger.error(f"FATAL: Metadata file not found at {metadata_path}")
        return
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    annotated_faces_path = run_dir / "annotated_faces.parquet"
    if not annotated_faces_path.exists():
        logger.error(f"FATAL: Intermediate artifact not found at {annotated_faces_path}")
        return

    logger.info(f"Loading intermediate data from {annotated_faces_path}...")
    df = pd.read_parquet(annotated_faces_path)

    generate_final_dataset(df, run_dir)
    create_qualitative_samples(df, run_dir, config)
    create_high_face_count_samples(metadata, run_dir)

    logger.info("Generating visualizations...")
    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)

    plot_confidence_histogram(df['face_score'].tolist(), visualizations_dir / "face_confidence_histogram.png")
    plot_face_size_distribution(df, visualizations_dir / "face_size_distribution.png")
    plot_confidence_vs_face_size(df, visualizations_dir / "confidence_vs_face_size.png")
    plot_image_mode_distribution(df, visualizations_dir / "image_mode_distribution.png")

    face_diagnostics = metadata.get("aggregated_metrics", {}).get("face_count_diagnostics", {})
    face_count_dist = face_diagnostics.get("faces_per_image_distribution", {})
    if face_count_dist:
        plot_face_count_distribution(face_count_dist, visualizations_dir / "face_count_distribution.png")

    resource_data_path = run_dir / "resource_utilization.json"
    if resource_data_path.exists():
        with open(resource_data_path, 'r') as f:
            resource_data = json.load(f)
        plot_resource_utilization(resource_data, visualizations_dir / "resource_utilization.png")
    else:
        logger.warning(f"Resource utilization file not found at {resource_data_path}. Skipping plot.")

    worker_perf = metadata.get("worker_performance", {})
    worker_times = worker_perf.get("worker_processing_times", [])
    if worker_times:
        plot_worker_performance_histogram(worker_times, visualizations_dir / "worker_performance.png")
    else:
        logger.warning("No worker timing data found in metadata. Skipping performance histogram.")

    logger.info("Calculating final metrics and generating report...")
    run_summary = metadata["run_summary"]
    agg_metrics = metadata["aggregated_metrics"]

    metrics = MetricsManager(
        run_id=run_summary["run_id"],
        run_command=run_summary["run_command"],
        git_commit_hash=run_summary["git_commit_hash"],
        environment=run_summary["environment"],
        config_snapshot=metadata["config"],
        start_time=metadata["start_time"],
    )
    metrics.total_images_processed = agg_metrics.get("total_images_processed", 0)
    metrics.images_with_decoding_errors = agg_metrics.get("images_with_decoding_errors", 0)
    metrics.images_with_no_faces = agg_metrics.get("images_with_no_faces", 0)
    metrics.total_faces_detected = agg_metrics.get("total_faces_detected", 0)
    metrics.faces_above_size_threshold = agg_metrics.get("faces_above_size_threshold", 0)
    metrics.faces_classified = agg_metrics.get("faces_classified", 0)
    metrics.faces_with_eyeglasses = agg_metrics.get("faces_with_eyeglasses", 0)
    metrics.faces_rejected_as_sunglasses = agg_metrics.get("faces_rejected_as_sunglasses", 0)
    metrics.final_target_count = agg_metrics.get("final_target_count", 0)

    metrics.failed_inference_batches = agg_metrics.get("failed_inference_batches", 0)
    metrics.corrupted_batches = agg_metrics.get("corrupted_batches", 0)
    metrics.corrupted_batch_images = agg_metrics.get("corrupted_batch_images", 0)

    face_diagnostics = agg_metrics.get("face_count_diagnostics", {})
    metrics.faces_per_image_distribution = face_diagnostics.get("faces_per_image_distribution", {})
    metrics.max_faces_in_single_image = face_diagnostics.get("max_faces_in_single_image", 0)
    metrics.images_with_multiple_faces = face_diagnostics.get("images_with_multiple_faces", 0)
    metrics.high_face_count_images = face_diagnostics.get("high_face_count_images", [])

    worker_perf = metadata.get("worker_performance", {})
    if worker_perf:
        metrics.worker_processing_times = worker_perf.get("worker_processing_times", [])
        metrics.worker_time_avg = worker_perf.get("avg_chunk_time_seconds", 0)
        metrics.worker_time_std = worker_perf.get("std_chunk_time_seconds", 0)
        metrics.worker_time_min = worker_perf.get("min_chunk_time_seconds", 0)
        metrics.worker_time_max = worker_perf.get("max_chunk_time_seconds", 0)

    if not df.empty:
        metrics.face_confidence_scores = df['face_score'].tolist()
        metrics.total_detection_time_seconds = df['detection_time_seconds'].sum()
        metrics.total_classification_time_seconds = df['classification_time_seconds'].sum()

    metrics.finalize()

    report_path = run_dir / "report.md"
    generate_report(metrics, report_path)
    logger.info(f"Report saved to {report_path}")

    logger.info("Artifact generation complete.")


if __name__ == "__main__":
    main()
