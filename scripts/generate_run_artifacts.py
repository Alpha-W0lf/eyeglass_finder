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


REPO_URL = "https://github.com/Alpha-W0lf/eyeglass_finder"
CREDITS_URL = f"{REPO_URL}/blob/main/CREDITS.md"
GALLERY_STYLE = (
    "<style>body{font-family:sans-serif;margin:0 16px;background:#fff;color:#111} header{max-width:820px;margin:16px 0} "
    "header p{line-height:1.5} .card{display:inline-block;margin:8px;border:1px solid #ddd;padding:6px;width:260px;"
    "max-width:100%;box-sizing:border-box;vertical-align:top;word-wrap:break-word} .img{width:100%;height:auto} "
    ".meta{font-size:14px;color:#333} .meta a{word-break:break-all}</style>"
)


def load_visual_review(path: str | Path) -> dict:
    """Read a visual review CSV (image_id, face_index, visual_review, plus any other columns) keyed by (image_id, face_index)."""
    table = pd.read_csv(path)
    return {(int(r.image_id), int(r.face_index)): r.visual_review for r in table.itertuples()}


def _face_key(record: dict) -> tuple:
    face_index = record.get("face_index")
    return int(record["image_id"]), (int(face_index) if face_index is not None and face_index == face_index else -1)


def select_gallery_records(records: list[dict], min_proba: float, visual_review: dict | None = None) -> list[dict]:
    """Records for the public page: score >= min_proba and, when a visual review is given, confirmed eyeglasses."""
    shown = [r for r in records if r["eyeglasses_proba"] >= min_proba]
    if visual_review is not None:
        return [r for r in shown if visual_review.get(_face_key(r)) == "eyeglasses"]
    # Without a visual review, show every record rather than an empty page
    return shown or records


def final_targets_header(run_id: str, total: int, shown: int, threshold: float, gallery_min: float,
                         confirmed: int | None = None) -> tuple[str, str]:
    """Title and header block for the public finds gallery."""
    if confirmed is not None:
        title = "Eyeglass Finder: finds confirmed in a blind visual review"
        summary = (f"The pipeline flagged {total} faces at or above the {threshold:.2f} eyeglasses cutoff in run "
                   f"<code>{run_id}</code>, a blind visual review confirmed {confirmed}, and this gallery shows the "
                   f"{shown} confirmed finds scored {gallery_min:.2f} or above, sorted by score.")
    elif shown < total:
        scope = (f"This page shows the {shown} clearest, scored {gallery_min:.2f} or higher, sorted by score. "
                 f"All {total} are in the run's metadata.csv and filtered dataset.")
        title = f"Eyeglass Finder: the {shown} clearest finds"
    else:
        scope = f"This page shows all {total}, sorted by score."
        title = f"Eyeglass Finder: the {total} finds"
    if confirmed is None:
        summary = (f"Run <code>{run_id}</code> kept {total} faces that the classifier scored at or above the "
                   f"{threshold:.2f} eyeglasses cutoff. {scope}")
    header = (
        "<header>\n"
        f"<h1>{title}</h1>\n"
        f"<p>{summary}</p>\n"
        f'<p><a href="{REPO_URL}">Code, method and full run report on GitHub</a> · '
        '<a href="../false_negative_candidates/index.html">Near misses below the cutoff</a></p>\n'
        "<p>Each card shows the eyeglasses score (eyeglasses_proba), the face-detector confidence (detector_conf), "
        f'the face size in pixels and a link to the source file page. Image credits: <a href="{CREDITS_URL}">CREDITS.md</a>.</p>\n'
        "</header>"
    )
    return title, header


def near_miss_header(count: int, low: float, high: float, threshold: float, left_out: int = 0) -> tuple[str, str]:
    """Title and header block for the near-miss gallery (left_out: faces dropped from a public copy)."""
    title = "Eyeglass Finder: near misses"
    note = (f" {left_out} of them {'is' if left_out == 1 else 'are'} left out of this page because the source image "
            "is not on Wikimedia Commons.") if left_out else ""
    header = (
        "<header>\n"
        f"<h1>{title}</h1>\n"
        f"<p>These are the {count} highest-scoring faces below the eyeglasses cutoff: "
        f"scores {low:.2f} to {high:.2f}, cutoff {threshold:.2f}. They are kept for manual review, "
        f"to spot eyeglasses the classifier missed.{note}</p>\n"
        f'<p><a href="{REPO_URL}">Code and full run report on GitHub</a> · '
        '<a href="../final_targets/index.html">The finds</a> · '
        f'Image credits: <a href="{CREDITS_URL}">CREDITS.md</a></p>\n'
        "</header>"
    )
    return title, header


def commons_file_page(image_url: str) -> str:
    """Map an upload.wikimedia.org Commons file URL to its file page (author and licence live there)."""
    marker = "upload.wikimedia.org/wikipedia/commons/"
    if marker not in image_url:
        return image_url
    return "https://commons.wikimedia.org/wiki/File:" + image_url.rsplit("/", 1)[-1]


def render_gallery_html(records: list[dict], title: str, header: str) -> str:
    """Render one gallery page: a header block plus one card per record."""
    html_lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        GALLERY_STYLE,
        "</head><body>",
        header,
    ]
    for rec in records:
        html_lines += [
            "<div class='card'>",
            f"<img class='img' src='{rec['filename']}' alt='Face crop, image_id {rec['image_id']}' loading='lazy'/>",
            f"<div class='meta'><b>image_id:</b> {rec['image_id']}</div>",
            f"<div class='meta'><b>eyeglasses_proba:</b> {rec['eyeglasses_proba']:.2f}</div>",
            f"<div class='meta'><b>detector_conf:</b> {rec['detector_conf']:.2f}</div>",
            f"<div class='meta'><b>is_target:</b> {rec['is_target']}</div>",
            f"<div class='meta'><b>sunglasses_prediction:</b> {rec['sunglasses_prediction']}</div>",
            f"<div class='meta'><b>face_size:</b> {rec['face_size']}</div>",
            f"<div class='meta'><b>face_bbox:</b> {rec['face_bbox']}</div>",
            (f"<div class='meta'><b>source:</b> <a href='{commons_file_page(rec['image_url'])}' target='_blank' rel='noopener noreferrer'>file page</a></div>" if rec['image_url'] else "<div class='meta'><b>source:</b> -</div>"),
            "</div>",
        ]
    html_lines.append("</body></html>")
    return "\n".join(html_lines) + "\n"


def _as_int_list(value):
    """Face sizes and boxes come back from parquet as numpy arrays; show them as plain lists."""
    if value is None:
        return None
    try:
        return [int(v) for v in value]
    except TypeError:
        return value


def rebuild_metrics(metadata: dict, df: pd.DataFrame) -> MetricsManager:
    """Rebuild the run's metrics from run_metadata.json so Stage 2 can re-run on its own."""
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
    for key in (
        "total_images_processed", "images_with_decoding_errors", "images_with_no_faces",
        "total_faces_detected", "faces_above_size_threshold", "faces_classified",
        "faces_with_eyeglasses", "faces_rejected_as_sunglasses", "final_target_count",
        "images_failed",
    ):
        setattr(metrics, key, agg_metrics.get(key, 0))

    face_diagnostics = agg_metrics.get("face_count_diagnostics", {})
    metrics.faces_per_image_distribution = face_diagnostics.get("faces_per_image_distribution", {})
    metrics.max_faces_in_single_image = face_diagnostics.get("max_faces_in_single_image", 0)
    metrics.images_with_multiple_faces = face_diagnostics.get("images_with_multiple_faces", 0)
    metrics.high_face_count_images = face_diagnostics.get("high_face_count_images", [])
    metrics.global_max_faces_in_single_image = agg_metrics.get("global_max_faces_in_single_image", 0)

    if not df.empty:
        metrics.face_confidence_scores = df["face_score"].tolist()

    worker_perf = metadata.get("worker_performance", {})
    metrics.finalize()  # sets worker stats; the run's own timings below take precedence
    if worker_perf:
        metrics.worker_processing_times = worker_perf.get("worker_processing_times", [])
        metrics.worker_time_avg = worker_perf.get("avg_chunk_time_seconds", 0)
        metrics.worker_time_std = worker_perf.get("std_chunk_time_seconds", 0)
        metrics.worker_time_min = worker_perf.get("min_chunk_time_seconds", 0)
        metrics.worker_time_max = worker_perf.get("max_chunk_time_seconds", 0)
    performance = metadata.get("performance")
    if performance:
        for key in ("total_runtime_seconds", "images_per_second",
                    "total_detection_time_seconds", "total_classification_time_seconds"):
            setattr(metrics, key, performance.get(key, 0.0))
    return metrics


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
    false_negative_candidates_dir = samples_root / "false_negative_candidates"
    final_targets_dir.mkdir(parents=True, exist_ok=True)
    rejected_sunglasses_dir.mkdir(parents=True, exist_ok=True)
    false_negative_candidates_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(config.model_params.classification.eyewear_prob_threshold)
    gallery_min = float(config.report_generation.gallery_min_eyeglasses_proba)

    visual_review = None
    visual_review_csv = config.report_generation.gallery_visual_review_csv
    if visual_review_csv:
        if Path(visual_review_csv).exists():
            visual_review = load_visual_review(visual_review_csv)
        else:
            logger.warning(f"Visual review file {visual_review_csv} not found; the finds page uses the score filter only.")

    def _targets_header(records, shown):
        confirmed = None
        if visual_review is not None:
            confirmed = sum(visual_review.get(_face_key(r)) == "eyeglasses" for r in records)
        return final_targets_header(output_dir.name, len(records), len(shown), threshold, gallery_min, confirmed)

    def _near_miss_header(records, shown):
        probs = [r["eyeglasses_proba"] for r in records]
        return near_miss_header(len(records), min(probs), max(probs), threshold)

    def _write_metadata_and_index(sample_df: pd.DataFrame, folder: Path, kind_label: str, header_fn=None,
                                  min_proba: float = 0.0, visual_review: dict | None = None):
        if sample_df.empty:
            return
        # Build metadata
        records = []
        for idx, row in sample_df.iterrows():
            records.append({
                "filename": None,  # filled below
                "face_id": idx,
                "image_id": int(row["image_id"]) if "image_id" in row and pd.notna(row["image_id"]) else idx,
                "face_index": row.get("face_index", None),
                "eyeglasses_proba": float(row.get("has_eyewear_confidence", row.get("glasses_confidence", 0.0)) or 0.0),
                "detector_conf": float(row.get("face_score", 0.0) or 0.0),
                "is_target": bool(row.get("is_target", False)),
                "sunglasses_prediction": bool(row.get("sunglasses_prediction", False)),
                "face_size": _as_int_list(row.get("face_size", None)),
                "face_bbox": _as_int_list(row.get("face_bbox", None)),
                "image_url": row.get("image_url", None),
                "source_file": row.get("source_file", None),
            })
        # Sort by eyeglasses probability descending for consistent display order
        records.sort(key=lambda r: r["eyeglasses_proba"], reverse=True)
        # Assign filenames consistently (must match actual saved files)
        for rec in records:
            rec["filename"] = f"{kind_label}_{rec['face_id']}_clf{rec['eyeglasses_proba']:.2f}_det{rec['detector_conf']:.2f}.jpg"
        # Write CSV metadata
        meta_df = pd.DataFrame.from_records(records)
        meta_df.to_csv(folder / "metadata.csv", index=False)
        # Write the gallery page (optionally only the clearest, confirmed records)
        shown = select_gallery_records(records, min_proba, visual_review)
        if header_fn is None:
            title = f"Eyeglass Finder: {kind_label} samples"
            header = f"<header>\n<h1>{title}</h1>\n</header>"
        else:
            title, header = header_fn(records, shown)
        (folder / "index.html").write_text(render_gallery_html(shown, title, header), encoding="utf-8")

    target_df = df[df["is_target"].eq(True)]
    if not target_df.empty:
        # Choose probability column for ranking
        proba_col = "has_eyewear_confidence" if "has_eyewear_confidence" in target_df.columns else ("glasses_confidence" if "glasses_confidence" in target_df.columns else None)
        if proba_col is not None:
            ranked_targets = target_df.sort_values(by=proba_col, ascending=False, na_position='last')
        else:
            ranked_targets = target_df
        # Save ALL final targets, sorted by eyewear probability desc
        saved = 0
        for idx, row in ranked_targets.iterrows():
            img_bytes = row.get("cropped_face_jpeg")
            if not img_bytes:
                continue
            fname = f"target_{idx}_clf{float(row.get('has_eyewear_confidence', row.get('glasses_confidence', 0.0)) or 0.0):.2f}_det{float(row.get('face_score', 0.0) or 0.0):.2f}.jpg"
            with open(final_targets_dir / fname, "wb") as f:
                f.write(img_bytes)
            saved += 1
        logger.info(f"Saved {saved} final target images.")
        _write_metadata_and_index(ranked_targets, final_targets_dir, "target", _targets_header, gallery_min, visual_review)

    # Optionally generate rejected sunglasses samples only if enabled and column exists
    try:
        if getattr(config.model_params.classification, "enable_sunglasses_rejection", False) and "sunglasses_prediction" in df.columns:
            rejected_df = df[df["sunglasses_prediction"].eq(True)]
            if not rejected_df.empty:
                proba_col = "has_eyewear_confidence" if "has_eyewear_confidence" in rejected_df.columns else ("glasses_confidence" if "glasses_confidence" in rejected_df.columns else None)
                if proba_col is not None:
                    rejected_df = rejected_df.sort_values(by=proba_col, ascending=False, na_position='last')
                rejected_sample = rejected_df.head(min(sample_size, len(rejected_df)))
                for idx, row in rejected_sample.iterrows():
                    img_bytes = row["cropped_face_jpeg"]
                    fname = f"rejected_{idx}_clf{float(row.get('has_eyewear_confidence', row.get('glasses_confidence', 0.0)) or 0.0):.2f}_det{float(row.get('face_score', 0.0) or 0.0):.2f}.jpg"
                    with open(rejected_sunglasses_dir / fname, "wb") as f:
                        f.write(img_bytes)
                logger.info(f"Saved {len(rejected_sample)} rejected sunglasses samples.")
                _write_metadata_and_index(rejected_sample, rejected_sunglasses_dir, "rejected")
            else:
                logger.info("No sunglasses rejections to sample.")
        else:
            logger.info("Sunglasses rejection disabled; skipping rejected_as_sunglasses samples.")
    except Exception as e:
        logger.warning(f"Failed generating rejected_sunglasses samples: {e}")

    # Updated: False-negative candidates — top-N highest eyeglasses probability below threshold (most borderline rejects)
    try:
        try:
            thr = float(config.model_params.classification.eyewear_prob_threshold)
        except Exception:
            thr = 0.5

        # Choose probability column
        proba_col = None
        if "has_eyewear_confidence" in df.columns:
            proba_col = "has_eyewear_confidence"
        elif "glasses_confidence" in df.columns:
            proba_col = "glasses_confidence"

        if proba_col is not None and not df.empty:
            candidates = df[df["is_target"].eq(False) & (df[proba_col].notnull()) & (df[proba_col] < thr)]
            ranked = candidates.sort_values(by=proba_col, ascending=False)
            sample = ranked.head(sample_size)
            if not sample.empty:
                for idx, row in sample.iterrows():
                    img_bytes = row.get("cropped_face_jpeg")
                    if not img_bytes:
                        continue
                    fname = f"candidate_{idx}_clf{float(row.get(proba_col, 0.0) or 0.0):.2f}_det{float(row.get('face_score', 0.0) or 0.0):.2f}.jpg"
                    with open(false_negative_candidates_dir / fname, "wb") as f:
                        f.write(img_bytes)
                logger.info(f"Saved {len(sample)} false-negative candidate samples (top-{min(sample_size, len(ranked))} below threshold {thr}).")
                _write_metadata_and_index(sample, false_negative_candidates_dir, "candidate", _near_miss_header)
            else:
                logger.info("No non-target faces found below threshold to sample for false-negative candidates.")
        else:
            logger.info("No probability column found for false-negative ranking; skipping.")
    except Exception as e:
        logger.warning(f"Failed generating false-negative candidates: {e}")


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

    sorted_images = sorted(high_face_count_images, key=lambda x: x.get('num_faces', 0), reverse=True)
    max_samples = min(20, len(sorted_images))

    for i, image_data in enumerate(sorted_images[:max_samples]):
        try:
            # Metadata may not include bytes if saved from metadata-only top_k; fall back to annotated_faces parquet lookup
            num_faces = int(image_data.get('num_faces', 0))
            filename = f"high_faces_{i+1:02d}_count_{num_faces}_faces.jpg"
            image_bytes = image_data.get('image_bytes')
            if image_bytes is None:
                # Try to locate cropped face bytes from annotated_faces parquet as fallback is not feasible for whole image.
                # We gracefully skip if not available.
                logger.warning(f"No raw image bytes for high-face image (rank {i+1}). Skipping save.")
            else:
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
        f.write("They are saved for manual review: mostly group photos and crowd scenes.\n\n")
        f.write("Image Details:\n")
        f.write("-" * 30 + "\n")
        for i, image_data in enumerate(sorted_images[:max_samples]):
            f.write(f"{i+1:2d}. {image_data['num_faces']:2d} faces - {image_data.get('image_url', 'unknown')}\n")


def generate_final_dataset(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating final filtered dataset...")
    target_df = df[df["is_target"].eq(True)]
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
    metrics = rebuild_metrics(metadata, df)

    report_path = run_dir / "report.md"
    generate_report(metrics, report_path)
    logger.info(f"Report saved to {report_path}")

    logger.info("Artifact generation complete.")


if __name__ == "__main__":
    main()
