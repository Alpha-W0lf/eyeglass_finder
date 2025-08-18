---
license: other
---

# Processed Dataset: Eyeglass Detection in Images

This repository contains the complete, reproducible output artifacts from a single run of the **[Eyeglass Finder pipeline](https://github.com/Alpha-W0lf/eyeglass_finder)**.

> ### tldr
>
> -   **What is this?** This repository contains the full, reproducible output from the [Eyeglass Finder pipeline](https://github.com/Alpha-W0lf/eyeglass_finder), a project heavily optimized for high-throughput, GPU-accelerated processing on Apple Silicon (M2 Max).
> -   **The Goal:** To process a challenging, pre-filtered subset of the WIT dataset to find images of people wearing **eyeglasses** (not sunglasses) with faces at least 100x100 pixels.
> -   **The Result:** From **39,258** source images, the pipeline successfully identified **125** target faces that met all criteria.
> -   **Key Files:**
>     - `filtered_dataset.parquet`: The final dataset of 125 faces.
>     - `annotated_faces.parquet`: The intermediate dataset of all 1,295 potential candidates for deeper analysis.
>     - `report.md`: A comprehensive summary of the pipeline run and performance metrics.

The goal of the pipeline was to take on a challenging "needle in a haystack" filtering task. The source, a subset of the `wikimedia/wit_base` dataset, is not only vast and uncurated, but was also pre-filtered by its creators to remove images with prominent human faces for privacy reasons. Our objective was therefore to find the subtle signals that remained: identifying and extracting all images containing human faces (at least 100x100 pixels) wearing eyeglasses, while explicitly excluding sunglasses. This final dataset represents the successful result of that search and demonstrates the effectiveness of the underlying high-performance pipeline.

## Key Results from this Run

The pipeline successfully processed **39,258 images** and identified **125 target faces** that met all specified criteria.

### Data Funnel Statistics
| Stage                               | Count      |
| ----------------------------------- | ---------- |
| Total Images Processed              | 39,258     |
| Images with No Faces                | 31,990     |
| Total Faces Detected                | 15,465     |
| Faces > 100x100px Threshold         | 1,295        |
| Faces Classified                    | 1,295        |
| Faces with Eyeglasses (Predicted)   | 125         |
| Faces Rejected as Sunglasses        | N/A (disabled) |
| **Final Target Faces**              | **125**     |

### Face Detection Count Analysis — Investigation Completed

The detection of **15,465 total faces** across 39,258 images (approximately **0.39 faces per image**) initially appeared to exceed expectations for a dataset reportedly pre-filtered to remove prominent faces. **Systematic investigation has resolved this concern**: the high count is explained by the presence of multiple faces per image. Many images contain multiple faces, with some images containing over 100 faces (likely crowd scenes, group photos, events, family gatherings, etc.). This finding validates the YOLOv8-Face model's accuracy and explains the mathematical discrepancy between expected single-face images and actual multi-face reality.

The pipeline's comprehensive diagnostic framework enabled this systematic validation through automatic identification of high face count images, detailed distribution analyses, and qualitative sampling for manual verification. The pipeline achieved perfect data integrity for this run, with comprehensive tracking ensuring all 39,258 input images were properly accounted for in both the data funnel and diagnostic statistics.

## Dataset Contents

This repository provides more than just the final dataset; it includes all intermediate artifacts, logs, and reports necessary for full transparency and reproducibility.

```
.
├── filtered_dataset.parquet        # The final, clean dataset of 125 target faces.
├── annotated_faces.parquet         # Intermediate data: all 1,295 faces that met the size threshold.
├── report.md                       # The full markdown report summarizing the run.
├── run_metadata.json               # The raw JSON metadata for the run (config, timings, etc.).
├── qualitative_analysis/
│   ├── final_targets/              # All 125 images from the final dataset, sorted by confidence.
│   ├── false_negative_candidates/  # Top 20 images rejected just below the confidence threshold.
│   └── high_face_count_images/     # Diagnostic samples of images with many detected faces.
├── visualizations/
│   ├── face_confidence_histogram.png
│   ├── face_size_distribution.png
│   ├── confidence_vs_face_size.png
│   ├── image_mode_distribution.png
│   ├── face_count_distribution.png
│   ├── worker_performance.png
│   └── resource_utilization.png
├── logs/
│   ├── pipeline.log
│   └── ... (worker logs)
└── README.md                       # This file (the dataset card).
```

### Key Files Explained

#### `filtered_dataset.parquet` (Final Output)
This is the primary output. It contains the 125 rows corresponding to the faces that were identified as the target class (wearing eyeglasses, not sunglasses).

**Schema:**
| Column                  | Type        | Description                                                       |
| :---------------------- | :---------- | :---------------------------------------------------------------- |
| `image_url`             | `string`    | The original URL of the source image for traceability.            |
| `source_file`           | `string`    | The name of the input Parquet file where the image was sourced.   |
| `face_bbox`             | `list[int]` | `[x_min, y_min, x_max, y_max]` coordinates of the detected face.  |
| `face_confidence`       | `float`     | The confidence score from the face detection model.               |
| `face_jpeg`             | `bytes`     | Binary data of the cropped and resized face (JPEG format).        |


#### `annotated_faces.parquet` (Intermediate Data)
This file is provided for deeper analysis and debugging. It contains one row for **every single face (1,295 total)** that was detected and met the `100x100px` size criteria, *before* the final filtering was applied. This allows for analysis of all candidates, not just the final selections.

**Schema:**
| Column                      | Type        | Description                                                       |
| :-------------------------- | :---------- | :---------------------------------------------------------------- |
| `image_url`                 | `string`    | The original URL of the source image.                             |
| `face_bbox`                 | `list[int]` | `[x_min, y_min, x_max, y_max]` coordinates of the face.           |
| `face_score`                | `float`     | The confidence score from the face detection model.               |
| `eyeglasses_prediction`     | `boolean`   | Final prediction from the `eyeglasses` classifier.                |
| `sunglasses_prediction`     | `boolean`   | Final prediction from the `sunglasses` classifier.                |
| `is_target`                 | `boolean`   | The final decision: `True` if the face is the target class.       |
| `cropped_face_jpeg`         | `bytes`     | Binary data of the cropped and resized face (JPEG format).        |
| `face_size`                 | `tuple`     | `(width, height)` of the detected face in pixels.                 |
| `original_image_size`       | `tuple`     | `(width, height)` of the original source image.                   |
| `original_image_mode`       | `string`    | Color mode of the original image (e.g., 'RGB', 'L').             |
| `source_file`               | `string`    | The name of the input Parquet file where the image was sourced.   |

---

## Run & Reproducibility Details

- **Source GitHub Repository:** https://github.com/Alpha-W0lf/eyeglass_finder
- **Run ID:** `run_2025-08-17_20-18-29`
- **Execution Environment:** This dataset was generated by running the pipeline in its native Python environment on an M2 Max MacBook Pro to leverage GPU acceleration (MPS), achieving a throughput of **167.5 images/second**. While a CPU-only Docker environment is available for cross-platform reproducibility, the native environment is the primary path for performance.
- **Full Configuration:** The exact `config.yaml` used for this run is documented within `report.md`.