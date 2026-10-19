---
license: other
---

# Dataset Card: Eyeglasses Faces from WIT

This card describes the dataset one run of the **[Eyeglass Finder pipeline](https://github.com/Alpha-W0lf/eyeglass_finder)** produces: faces of people wearing eyeglasses, found in Wikimedia's WIT images.

> ### tldr
>
> -   **Run:** `run_2026-10-19_01-20-39` on an Apple M2 Max (native, GPU via MPS).
> -   **Result:** from **39,258** source images, **44** faces at least 100 px that the classifier scored as wearing eyeglasses (score ≥ 0.90). A blind visual review confirmed 43 of them.
> -   **In this repository:** the run's report, plots and galleries, plus [`confirmed_visual_review.csv`](docs/latest_run_showcase/confirmed_visual_review.csv) and [`image_credits.csv`](docs/latest_run_showcase/image_credits.csv), all in [`docs/latest_run_showcase/`](docs/latest_run_showcase/). The parquet files described below are written to `outputs/run_<timestamp>/` when you run the pipeline; they are not committed.

## Source data

The source is two shards of `wikimedia/wit_base`. For privacy, its publishers left out images where a person is the primary subject (a face covering more than 10% of the picture, by their RetinaFace detector) ([dataset card](https://huggingface.co/datasets/wikimedia/wit_base)), so the faces that remain are small and rare. The pipeline keeps faces at least 100 px on each side and looks for eyeglasses (sunglasses rejection is off in this run). Image credits: [`CREDITS.md`](CREDITS.md).

## Data funnel (this run)

| Stage                               | Count      |
| ----------------------------------- | ---------- |
| Total Images Processed              | 39,258     |
| Images with No Faces                | 31,990     |
| Total Faces Detected                | 15,465     |
| Faces > 100x100px Threshold         | 414        |
| Faces Classified                    | 414        |
| Faces with Eyeglasses (Predicted)   | 44         |
| Faces Rejected as Sunglasses        | N/A (disabled) |
| **Final Target Faces**              | **44**     |
| Confirmed in a blind visual review  | 43 of 44   |

The detector finds **15,465 faces** across 39,258 images (about **0.39 faces per image**) because many images hold several faces, up to 73 in one image (crowd scenes, group photos, events).

## Files a run writes

```
outputs/run_<timestamp>/
├── filtered_dataset.parquet        # The final dataset (44 rows in this run).
├── annotated_faces.parquet         # Every face that met the size threshold (414 in this run).
├── report.md                       # The run report (committed for this run).
├── run_metadata.json               # Config, timings and funnel counts.
├── qualitative_analysis/
│   ├── final_targets/              # The finds; the public page shows the 38 confirmed in a blind visual review and scored 0.97+.
│   ├── false_negative_candidates/  # The 20 highest-scoring faces below the cutoff.
│   └── high_face_count_images/     # The images with the most faces (left out of the public copy).
├── visualizations/                 # Plots (committed for this run).
└── logs/                           # Run and production logs.
```

### `filtered_dataset.parquet`

One row per target face.

| Column            | Type        | Description |
| :---------------- | :---------- | :---------- |
| `image_url`       | `string`    | Source image URL. |
| `source_file`     | `string`    | Input parquet file the image came from. |
| `face_bbox`       | `list[int]` | `[x_min, y_min, x_max, y_max]` of the detected face. |
| `face_confidence` | `float`     | Face-detector confidence. |
| `face_jpeg`       | `bytes`     | The face as a JPEG: a square crop with a 20% margin, resized to 224 x 224. |

### `annotated_faces.parquet`

One row per face that met the size threshold, before filtering.

| Column                   | Type        | Description |
| :----------------------- | :---------- | :---------- |
| `image_id`               | `int`       | Row id of the source image across the input files. |
| `image_url`              | `string`    | Source image URL. |
| `source_file`            | `string`    | Input parquet file the image came from. |
| `original_image_mode`    | `string`    | Colour mode of the source image (e.g. `RGB`, `L`). |
| `face_index`             | `int`       | Index of the face within its image. |
| `face_bbox`              | `list[int]` | `[x_min, y_min, x_max, y_max]` of the face. |
| `face_size`              | `list[int]` | `[width, height]` of the face in pixels. |
| `face_score`             | `float`     | Face-detector confidence. |
| `has_eyewear_confidence` | `float`     | Eyeglasses classifier score. |
| `glasses_confidence`     | `float`     | Same score, kept for older readers. |
| `sunglasses_prediction`  | `bool`      | Always `False` while sunglasses rejection is off. |
| `is_target`              | `bool`      | `True` if the score is at or above the cutoff (0.90). |
| `cropped_face_jpeg`      | `bytes`     | The face crop as a JPEG (as in `face_jpeg`). |

## Run details

- **Source repository:** https://github.com/Alpha-W0lf/eyeglass_finder
- **Run ID:** `run_2026-10-19_01-20-39` (225.05 images/sec)
- **Environment:** native Python on an M2 Max MacBook Pro, GPU via MPS. Docker is the portable, CPU-only option on macOS.
- **Configuration:** `config/production.yaml`, echoed in full in [`report.md`](docs/latest_run_showcase/report.md).
