
# Pipeline Run Report: run_2026-10-19_01-20-39

## 1. Run Summary
- **Git Commit Hash:** `e20624b` (pre-release history; the public repository starts from a later single commit)
- **Run Command:** `scripts/process_data.py --config config/production.yaml`
- **Environment:**
```json
python_version: 3.12.8

```

## 2. Performance Dashboard
| Metric                    | Value      |
| ------------------------- | ---------- |
| Total Runtime (seconds)   | 174.44 |
| Images per Second         | 225.05     |
| Total Detection Time (s)  | 498.20 |
| Total Classification Time (s)| 11.90 |

#### 2.1. Worker Performance
| Metric (per chunk)            | Value      |
| ----------------------------- | ---------- |
| Avg. Processing Time (s)      | 3.09 |
| Std. Dev. Processing Time (s) | 0.63 |
| Min. Processing Time (s)      | 1.02 |
| Max. Processing Time (s)      | 7.8 |

![Worker Performance](visualizations/worker_performance.png)

### 2.2. System Resource Utilization
![System Resource Utilization](visualizations/resource_utilization.png)

## 3. Data Funnel
| Stage                               | Count      |
| ----------------------------------- | ---------- |
| Total Images Processed              | 39258 |
| Images with Decoding Errors         | 0 |
| Images with No Faces                | 31990 |
| Images with Faces                   | 7268 |
| Total Faces Detected                | 15465 |
| Faces > Size Threshold              | 414 |
| Faces Classified                    | 414 |
| Faces with Eyeglasses (Predicted)   | 44 |
| Faces Rejected as Sunglasses        | N/A |
| **Final Target Faces**              | **44** |

### 3.1. Face Detection Diagnostics

**Key Findings:**
- **Maximum faces in single image:** 73
- **Images with multiple faces:** 2343
- **Average faces per image (images with faces):** 2.13

**Face Count Distribution:**
**Face Count Distribution Summary:**
- **Images with no faces:** 31,990 (81.5%)
- **Images with 1 face:** 4,925 (12.5%)
- **Images with 2-5 faces:** 1,854 (4.7%) - *Small groups*
- **Images with 6+ faces:** 489 (1.2%) - *Group photos/crowd scenes*
- **Highest concentration:** 73 faces in a single image

*See visualization below for complete distribution pattern.*

**Note:** Many of the 15465 detected faces come from group photos and crowd scenes. The run saved the images with the most detected faces for manual review; those full photos are left out of this public copy ([note](./qualitative_analysis/high_face_count_images/README.txt)).

### 3.2. Pipeline Robustness

**Failure Tracking:**
- **Individual Decoding Errors:** 0

## 4. Model Quality Analysis

### 4.1. Distributions

| Face Detection Confidence | Original Image Modes |
| :---: | :---: |
| ![visualizations/face_confidence_histogram.png](visualizations/face_confidence_histogram.png) | ![visualizations/image_mode_distribution.png](visualizations/image_mode_distribution.png) |

| Face Size Distribution | Confidence vs. Face Size |
| :---: | :---: |
| ![visualizations/face_size_distribution.png](visualizations/face_size_distribution.png) | ![visualizations/confidence_vs_face_size.png](visualizations/confidence_vs_face_size.png) |

### 4.3. Face Count Distribution (Diagnostic)

![Face Count Distribution](visualizations/face_count_distribution.png)

This plot shows how many faces were detected per image (15465 faces in total).

### 4.4. Qualitative Samples

*This public copy leaves out the full high-face-count photos and one near-miss image that is not on Wikimedia Commons; the counts above include them.*

**Final Targets**: Faces kept in the final dataset (eyeglasses score at or above the cutoff). The page header states which scores it shows.

[Open Index](./qualitative_analysis/final_targets/index.html)

*Sunglasses rejection disabled in this run.*

**False-Negative Candidates (Sample)**: Large faces above the size threshold that were classified as non-target. Manually inspect to spot missed eyeglasses.

[Open Index](./qualitative_analysis/false_negative_candidates/index.html)

**High Face Count Images (Diagnostic)**: The images with the most detected faces (more than 5 per image), saved for manual review. The full photos are left out of this public copy ([note](./qualitative_analysis/high_face_count_images/README.txt)).

## 5. Full Configuration (`config.yaml`)
```yaml
paths:
  input_dir: data/raw
  output_dir: outputs/run_2026-10-19_01-20-39
  output_filename: filtered_dataset.parquet
  logs_dir: outputs/run_2026-10-19_01-20-39/logs
data_processing:
  file_pattern: train-*.parquet
  chunk_size: 512
model_params:
  face_detection:
    model_path: models/yolov8n-face-lindevs.pt
    min_face_size: 100
    min_confidence: 0.75
    keep_all: false
    target_size:
    - 224
    - 224
  classification:
    present_label: present
    output_image_format: JPEG
    eyewear_prob_threshold: 0.9
    kind: eyeglasses
    enable_sunglasses_rejection: false
    sunglasses_prob_threshold: 0.65
execution:
  diagnostic_serial_mode: false
  inference_batch_size: 64
logging:
  level: INFO
  format: '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level>
    | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
report_generation:
  qualitative_analysis_sample_size: 20
  gallery_min_eyeglasses_proba: 0.97
hardware:
  max_workers: 10
  worker_memory_limit_mb: 1500
  use_mps_acceleration: true
  device_override: null
performance:
  prefetch_chunks: 4
  gc_frequency: 10
  thread_pool_workers: 4
  batch_optimization: true
  inference_batch_size: 64
  face_classification_batch_size: 16
  max_batch_accumulation_time: 100
  rampup_enabled: true
  rampup_warmup_chunks: 4
  rampup_initial_prefetch_chunks: 1
  rampup_initial_chunk_size_override: 192
  rampup_stagger_worker_submissions_ms: 100
observability:
  detailed_metrics: false
  memory_profiling: false
  benchmark_mode: true
  performance_alerts:
    cpu_threshold: 90
    memory_threshold: 85
    min_throughput: 90.0
sampling:
  square_crop: true
  crop_margin: 0.2
  clamp_to_image: true
  pad_mode: null
  apply_to_classifier: false
run_id: run_2026-10-19_01-20-39

```
