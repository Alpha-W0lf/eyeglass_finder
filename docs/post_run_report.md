# Post-Project Analysis & Optimization Summary

---

## 1. Executive Summary

-   **Objective:** To architect and systematically optimize a data pipeline for a challenging "needle-in-a-haystack" computer vision task: identifying faces with eyeglasses in a large, noisy, pre-filtered dataset.
-   **Outcome:** Tuning and moving to the M2 Max GPU took throughput from **10.6 images/second** (pre-tuning code, Docker CPU; [baseline report](./baseline_run/report.md)) to **225 images/second** ([showcase report](./latest_run_showcase/report.md)): **~21× faster** (225.05 / 10.58 = 21.27). An earlier check run of the same code measured 179 images/sec (its report is not committed); speed varies between runs on the same machine.
-   **Final Results (from `run_2026-10-19_01-20-39`):** The optimized pipeline processed **39,258 images** and kept **44 target faces** (eyeglasses score ≥ 0.90). A blind AI visual review confirmed 43 of them ([`visual_review.csv`](./latest_run_showcase/visual_review.csv)).
-   **Showcase artifacts:** The run report, plots and galleries are in [`docs/latest_run_showcase/`](./latest_run_showcase/report.md). Full parquet outputs are written to `outputs/` locally and are not committed.

---

## 2. System Design

### 2.1. Two-Stage Architecture

1.  **Stage 1 (`process_data.py`):** Runs face detection and classification in parallel worker processes on the best available hardware and saves every candidate face to an intermediate dataset.
2.  **Stage 2 (`generate_run_artifacts.py`):** Filters that data and writes the final dataset, the Markdown report, the plots and the galleries. It can be re-run on its own, so filtering and reporting change without a new model pass.

More detail: [High-Level Architecture](./TECHNICAL.md#high-level-architecture).

### 2.2. Technology Stack

| Component | Technology | Why |
| :-- | :-- | :-- |
| Environments | Poetry (native) and Docker | Native Poetry is the fast path on Apple Silicon (GPU via MPS). Docker is the portable path; on macOS it is CPU-only and gives the 10.6 images/sec baseline. |
| Dependencies | Poetry lock file | Exact, repeatable installs. |
| ML framework | PyTorch | Runs both models on CUDA, MPS or CPU. |
| Data format | Apache Parquet | Columnar and typed; streams in chunks. |

### 2.3. Model Selection

Detection uses a `YOLOv8-Face` model trained by **[Lindevs](https://github.com/lindevs/yolov8-face)** on WIDER FACE; classification uses [`glasses-detector`](https://github.com/mantasu/glasses-detector). Two specialised models keep each stage swappable and easy to debug. Alternatives considered (RetinaFace, YuNet) and why YOLOv8-Face won: see [Design Rationale](./TECHNICAL.md#design-rationale--alternatives-considered).

**The decision behind the eyeglasses-only scope:** in the earlier dual-classifier design, a `sunglasses` classifier removed faces the `eyeglasses` model had kept. The pipeline therefore runs with sunglasses rejection **off** (`enable_sunglasses_rejection: false`); the showcase run's **44** finds come from the eyeglasses classifier alone. The switch stays in the config for a future calibrated sunglasses model.

### 2.4. Accounting for Every Image

Every input image lands in the run's data funnel: no face, faces detected, decode error, or failed (an image in a failed mini-batch or a crashed worker chunk). An earlier version lost images because worker-level and pipeline-level counts collided; fixing that made the funnel add up. The showcase run reports 0 decode errors.

### 2.5. Performance Choices

GPU acceleration was the path to throughput, so CPU-specific work such as custom `ONNX Runtime` builds was left out to keep the Docker build simple. A built-in monitor records CPU, memory and disk use every second, plus per-chunk worker times, and every run plots them.

---

## 3. The Optimization Journey: From Baseline to High-Throughput

**Starting baseline:** the pre-tuning code (commit `1a6c8aa`, August 2025) in Docker on the CPU of the same M2 Max, 8 workers, a 4,000-image test set. The original 2025 run (`2025-08-16_17-26-06`) measured 9.8 images/second in 408 s; re-measured on 2026-10-19 with the same code and images it ran at **10.58 images/second** in 378 s ([baseline report](./baseline_run/report.md)).

### 3.1. Phase 1: Native Execution and MPS
-   Moved from CPU-only Docker to native Python on Apple Silicon, so the models run on the GPU through Metal Performance Shaders (MPS).
-   Added automatic device selection: CUDA, then MPS, then CPU (`src/utils/device.py`).

### 3.2. Phase 2: Tuning
-   **Batched face detection:** detection runs on mini-batches of images.
-   **Worker scaling:** measured 8 to 12 workers on the M2 Max and settled on 10.
-   **Memory management:** a `MemoryManager` watches system memory and throttles data prefetching under pressure.

### 3.3. Phase 3: Operations
-   **Production log:** milestones and progress snapshots go to a JSONL log in each run folder.
-   **Galleries:** browsable HTML pages of the finds and near misses, with scores and source links.
-   **Production config:** a tuned `config/production.yaml`.

---

## 4. Dataset Findings

### 4.1. A Needle in a Haystack

The `wikimedia/wit_base` creators removed many images with prominent faces for privacy, so the remaining faces are small and rare. The showcase run (`run_2026-10-19_01-20-39`) processed **39,258** images and kept **44** target faces (~0.11% of source images); see [`report.md`](./latest_run_showcase/report.md) and [`dataset_card.md`](../dataset_card.md). A source with more portraits would yield more finds per image scanned.

### 4.2. Why 15,465 Faces

The run detects **15,465 faces** across 39,258 images (about **0.39 faces per image**), more than a privacy-filtered dataset suggests. The diagnostics explain it: many images hold several faces, up to 73 in one image, mostly crowd scenes, group photos and events. The saved high-face-count images and the face-count plot make that easy to check, and those diagnostics now run on every run.

### 4.3. Signals for Downstream Users

-   **Confidence scores:** the final dataset keeps the detector and classifier scores, so users can set their own operating point.
-   **Galleries:** each run writes pages of kept faces and near misses for fast manual review.

---

## 5. Scaling Approach

The pipeline runs on one machine. To scale out, each model becomes its own auto-scaling service behind a message queue (for example Apache Kafka), run on Kubernetes with images in object storage such as Amazon S3. New filters then plug in as new services without touching the existing ones. Deployment notes: [Production Deployment](./production_deployment.md).

---

## 6. Next Steps

1.  **Higher-density source data:** a portrait-rich source to build a large eyeglasses dataset faster.
2.  **Calibrated sunglasses rejection:** turn the existing switch back on once a calibrated sunglasses model is checked against a labelled set.
3.  **Fine-tuning for the WIT domain:** start with the last layer and move to full fine-tuning as a labelled set grows.
4.  **Multimodal filtering:** pre-filter by image quality and content type, and use WIT's captions (keyword rules or CLIP).
5.  **Run reviews:** turn each run's report and galleries into evidence-backed tasks for the next iteration.

---

## 7. References
- Srinivasan, K., Raman, K., Chen, J., Bendersky, M., & Najork, M. (2021). *WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning*. arXiv preprint arXiv:2103.01913.
