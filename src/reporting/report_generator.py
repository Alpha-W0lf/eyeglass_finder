"""Generates the text-based Markdown report for a pipeline run.

This module is responsible for creating the final, human-readable summary
report in Markdown format. It takes a fully populated `MetricsManager` object
and uses it to render a template that includes run metadata, performance
metrics, a data funnel, and links to visual artifacts.
"""
import textwrap
import yaml
from pathlib import Path

from src.utils.metrics import MetricsManager


def generate_report(metrics: MetricsManager, output_path: Path):
    """
    Renders and saves a comprehensive Markdown report from pipeline metrics.
    """
    viz_dir_name = "visualizations"
    qual_dir_name = "qualitative_analysis"

    # Prefer a globally persisted max if the caller hydrated it into metrics via run_metadata.json
    global_max = getattr(metrics, "global_max_faces_in_single_image", None)
    max_faces_effective = global_max if isinstance(global_max, int) and global_max > 0 else metrics.max_faces_in_single_image

    # Conditionally show sunglasses rows/links
    cfg = metrics.config_snapshot
    cls_cfg = cfg.get("model_params", {}).get("classification", {}) if isinstance(cfg, dict) else {}
    sunglasses_enabled = bool(cls_cfg.get("enable_sunglasses_rejection", False))

    report_content = f"""
# Pipeline Run Report: {metrics.run_id}

## 1. Run Summary
- **Git Commit Hash:** `{metrics.git_commit_hash}`
- **Run Command:** `{metrics.run_command}`
- **Environment:**
```json
{yaml.dump(metrics.environment, indent=2)}
```

## 2. Performance Dashboard
| Metric                    | Value      |
| ------------------------- | ---------- |
| Total Runtime (seconds)   | {metrics.total_runtime_seconds} |
| Images per Second         | {metrics.images_per_second}     |
| Total Detection Time (s)  | {metrics.total_detection_time_seconds:.2f} |
| Total Classification Time (s)| {metrics.total_classification_time_seconds:.2f} |

#### 2.1. Worker Performance
| Metric (per chunk)            | Value      |
| ----------------------------- | ---------- |
| Avg. Processing Time (s)      | {metrics.worker_time_avg} |
| Std. Dev. Processing Time (s) | {metrics.worker_time_std} |
| Min. Processing Time (s)      | {metrics.worker_time_min} |
| Max. Processing Time (s)      | {metrics.worker_time_max} |

![Worker Performance]({viz_dir_name}/worker_performance.png)

### 2.2. System Resource Utilization
![System Resource Utilization]({viz_dir_name}/resource_utilization.png)

## 3. Data Funnel
| Stage                               | Count      |
| ----------------------------------- | ---------- |
| Total Images Processed              | {metrics.total_images_processed} |
| Images with Decoding Errors         | {metrics.images_with_decoding_errors} |
| Images with No Faces                | {metrics.images_with_no_faces} |
| Images with Faces                   | {max(0, metrics.total_images_processed - metrics.images_with_no_faces)} |
| Total Faces Detected                | {metrics.total_faces_detected} |
| Faces > Size Threshold              | {metrics.faces_above_size_threshold} |
| Faces Classified                    | {metrics.faces_classified} |
| Faces with Eyeglasses (Predicted)   | {metrics.faces_with_eyeglasses} |
| Faces Rejected as Sunglasses        | {metrics.faces_rejected_as_sunglasses if sunglasses_enabled else 'N/A'} |
| **Final Target Faces**              | **{metrics.final_target_count}** |

### 3.1. Face Detection Diagnostics

**Key Findings:**
- **Maximum faces in single image:** {max_faces_effective}
- **Images with multiple faces:** {metrics.images_with_multiple_faces}
- **Average faces per image (images with faces):** {((metrics.total_faces_detected / max(1, (metrics.total_images_processed - metrics.images_with_no_faces)))):.2f}

**Face Count Distribution:**
{_generate_face_count_summary(metrics.faces_per_image_distribution, metrics.total_images_processed)}

**Investigation Note:** The high face detection count ({metrics.total_faces_detected} total faces) warrants investigation. See the [High Face Count Images](./{qual_dir_name}/high_face_count_images) for manual inspection of images with the most detected faces.

### 3.2. Pipeline Robustness

**Failure Tracking:**
- **Failed Inference Batches:** {metrics.failed_inference_batches}
- **Corrupted Image Batches:** {metrics.corrupted_batches} 
- **Images in Corrupted Batches:** {metrics.corrupted_batch_images}
- **Individual Decoding Errors:** {metrics.images_with_decoding_errors}

*These metrics help diagnose pipeline robustness and identify potential issues with batch processing, memory pressure, or data quality.*

## 4. Model Quality Analysis

### 4.1. Distributions

| Face Detection Confidence | Original Image Modes |
| :---: | :---: |
| ![{viz_dir_name}/face_confidence_histogram.png]({viz_dir_name}/face_confidence_histogram.png) | ![{viz_dir_name}/image_mode_distribution.png]({viz_dir_name}/image_mode_distribution.png) |

| Face Size Distribution | Confidence vs. Face Size |
| :---: | :---: |
| ![{viz_dir_name}/face_size_distribution.png]({viz_dir_name}/face_size_distribution.png) | ![{viz_dir_name}/confidence_vs_face_size.png]({viz_dir_name}/confidence_vs_face_size.png) |

### 4.3. Face Count Distribution (Diagnostic)

![Face Count Distribution]({viz_dir_name}/face_count_distribution.png)

This plot shows the distribution of how many faces were detected per image. The high total face count ({metrics.total_faces_detected} faces) can be analyzed by examining this distribution to identify if the pipeline is processing many group photos, crowd scenes, or experiencing false positives.

### 4.4. Qualitative Samples

**Final Targets (Sample)**: A sample of faces that were correctly identified and included in the final dataset.

[Open Index](./{qual_dir_name}/final_targets/index.html)

{('**Rejected Sunglasses (Sample)**: A sample of faces that were rejected because the model predicted they were wearing sunglasses. This is a key area for failure analysis.\n\n[Open Index](./' + qual_dir_name + '/rejected_as_sunglasses/index.html)') if sunglasses_enabled else '*Sunglasses rejection disabled in this run.*'}

**False-Negative Candidates (Sample)**: Large faces above the size threshold that were classified as non-target. Manually inspect to spot missed eyeglasses.

[Open Index](./{qual_dir_name}/false_negative_candidates/index.html)

**High Face Count Images (Diagnostic)**: Images with the highest number of detected faces (>5 faces per image). These are saved for manual inspection to investigate the unexpectedly high face detection count.

[View Diagnostic Samples](./{qual_dir_name}/high_face_count_images)

## 5. Full Configuration (`config.yaml`)
```yaml
{yaml.dump(metrics.config_snapshot, indent=2, sort_keys=False)}
```
"""
    with open(output_path, "w") as f:
        f.write(textwrap.dedent(report_content))


def _generate_face_count_summary(face_count_distribution: dict, total_images: int) -> str:
    if not face_count_distribution or total_images == 0:
        return "**Face Count Distribution Summary:** No data available"
    face_count_int_dict = {}
    for key, value in face_count_distribution.items():
        try:
            face_count_int_dict[int(key)] = value
        except (ValueError, TypeError):
            continue
    if not face_count_int_dict:
        return "**Face Count Distribution Summary:** No valid data available"
    no_faces = face_count_int_dict.get(0, 0)
    one_face = face_count_int_dict.get(1, 0)
    small_groups = sum(face_count_int_dict.get(i, 0) for i in range(2, 6))
    large_groups = sum(count for face_count, count in face_count_int_dict.items() if face_count >= 6)
    max_faces = max(face_count_int_dict.keys()) if face_count_int_dict else 0
    no_faces_pct = (no_faces / total_images) * 100 if total_images > 0 else 0
    one_face_pct = (one_face / total_images) * 100 if total_images > 0 else 0
    small_groups_pct = (small_groups / total_images) * 100 if total_images > 0 else 0
    large_groups_pct = (large_groups / total_images) * 100 if total_images > 0 else 0
    summary = f"""**Face Count Distribution Summary:**
- **Images with no faces:** {no_faces:,} ({no_faces_pct:.1f}%)
- **Images with 1 face:** {one_face:,} ({one_face_pct:.1f}%)
- **Images with 2-5 faces:** {small_groups:,} ({small_groups_pct:.1f}%) - *Small groups*
- **Images with 6+ faces:** {large_groups:,} ({large_groups_pct:.1f}%) - *Group photos/crowd scenes*
- **Highest concentration:** {max_faces} faces in a single image

*See visualization below for complete distribution pattern.*"""
    return summary
