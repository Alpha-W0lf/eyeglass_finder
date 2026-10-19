# Eyeglass Finder – Image Filtering Pipeline

This document covers the design, setup, configuration and outputs of a data pipeline that finds faces and classifies eyewear.

> ### tldr
>
> -   **The Goal:** Build a data pipeline to find a "needle-in-a-haystack": images of people wearing **eyeglasses** in a vast, noisy, pre-filtered dataset where faces are not the primary subject.
> -   **The Result:** A high-throughput, hardware-accelerated pipeline that runs **~21× faster** than the pre-tuning Docker CPU baseline: **10.6 → 225 images/second** on an M2 Max.
> -   **Run on Apple Silicon (Recommended for Performance):**
>     ```bash
>     # Install dependencies once
>     # Python 3.12 required (3.13+ lacks wheels for pinned pyarrow/pydantic-core).
>     poetry install
>     # Run the two-stage pipeline
>     poetry run python scripts/process_data.py --config config/production.yaml
>     ```
> -   **Run with Docker (for Cross-Platform Reproducibility):**
>     ```bash
>     GIT_COMMIT_HASH=$(git rev-parse HEAD) docker compose build && docker compose run --rm app
>     ```
> -   **Key Highlights:**
>     -   **Extreme Performance Optimization:** Transitioned from a containerized CPU baseline to a native, GPU-accelerated pipeline on Apple Silicon, using MPS, batched detection, and memory tuning for an order-of-magnitude speedup.
>     -   **Modular Architecture:** A two-stage design, structured logging, memory-aware prefetching and a typed configuration.
>     -   **Deep Observability:** Every run generates a rich set of artifacts, including detailed performance reports, resource utilization plots, and extensive qualitative samples for data validation and model behavior analysis.
>     -   **Strategic Problem Solving:** Systematically diagnosed and resolved model performance issues, replacing a dual-classifier approach with a config-gated "eyeglasses-only" model after review showed the sunglasses model removing valid targets, which also simplified the pipeline.
>
> This project is a case study in a real-world ML problem, emphasizing not just a functional outcome but the methodical process of optimization, diagnostics, and building a maintainable system.

---

## Table of Contents

- [tldr](#tldr)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [High-Level Architecture](#high-level-architecture)
- [Directory Structure](#directory-structure)
- [Design Rationale & Alternatives Considered](#design-rationale--alternatives-considered)
  - [Technology Stack Rationale](#technology-stack-rationale)
  - [Architectural Rationale](#architectural-rationale)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Native Python Environment](#1-native-python-environment-high-performance)
  - [Running the Test Suite](#running-the-test-suite)
  - [2. Docker Environment](#2-docker-environment-reproducibility)
  - [Hardware Acceleration Notes](#hardware-acceleration-notes)
  - [Configuration and Tuning](#configuration-and-tuning)
- [Output Dataset Schema](#output-dataset-schema)
  - [1. Intermediate Artifact: `annotated_faces.parquet`](#1-intermediate-artifact-annotated_facesparquet)
  - [2. Final Output: `filtered_dataset.parquet`](#2-final-output-filtered_datasetparquet)
  - [3. Diagnostic and Analysis Artifacts](#3-diagnostic-and-analysis-artifacts)
- [Scaling and Extensibility](#scaling-and-extensibility)
  - [Scaling Out: A Microservices Approach](#scaling-out-a-microservices-approach)
  - [Designed for Extensibility](#designed-for-extensibility)
- [Next Steps](#next-steps)
- [Project Links](#project-links)

---

---

## Project Overview
The primary goal of this project is to showcase the engineering of a robust, scalable, and high-performance data pipeline to solve a challenging "needle-in-a-haystack" problem. It processes the Wikipedia-based Image Text (WIT) dataset—whose publishers left out images where a person is the primary subject (a face covering more than 10% of the picture, by their RetinaFace detector)—to identify the rare instances of individuals wearing eyeglasses.

The pipeline is engineered as a multi-stage ML system, separating heavy processing from final artifact generation:

1.  **Stage 1: Process & Enrich:**
    -   **Ingest** images from the source dataset using a memory-efficient streaming process.
    -   **Detect** all human faces using a hardware-accelerated YOLOv8-Face model.
    -   **Classify** each face with a pre-trained eyeglasses classifier (`glasses-detector`).
    -   **Persist** this rich, unfiltered data to an intermediate Parquet file, capturing every detected face for full traceability.
2.  **Stage 2: Filter & Report:**
    -   **Load** the intermediate data from Stage 1.
    -   **Filter** the faces based on project criteria (e.g., face size, final classification).
    -   **Generate** a clean final dataset and a comprehensive suite of run artifacts, including a detailed Markdown report, performance visualizations, and qualitative image samples for validation.

> **Note on Classification Strategy:** Initial analysis revealed that a dual-classifier system (for `eyeglasses` and `sunglasses`) was counterproductive, with the sunglasses model aggressively removing valid targets. The architecture was refactored to use a single classifier focused only on eyeglasses. The legacy dual-classifier logic remains available and can be re-enabled via the configuration file for future experimentation.

## Key Features
-   **Extreme Performance Engineering:** Methodically optimized the pipeline from a 10.6 images/sec Docker CPU baseline to **225 images/sec** on Apple Silicon, **~21× faster** ([baseline report](./baseline_run/report.md), [showcase report](./latest_run_showcase/report.md)). This was achieved through a multi-faceted approach including native execution, MPS GPU acceleration, worker scaling, memory architecture tuning, and batch processing.
-   **Hardware-Aware Execution:** The pipeline automatically detects and utilizes the best available hardware—NVIDIA (CUDA), Apple (MPS), or CPU—ensuring optimal performance on any machine with zero configuration changes.
-   **Two-Stage Architecture:** Employs a modular two-stage pipeline that decouples expensive model inference from lightweight reporting. This enhances flexibility, simplifies debugging, and allows for rapid iteration on filtering logic and analytics without re-running the entire process.
-   **Deep Observability & Rich Reporting:** Every run is treated as a reproducible experiment, generating a unique, timestamped output directory. Artifacts include a detailed Markdown report with performance analytics, system resource utilization plots (CPU, Memory, Disk I/O), worker efficiency histograms, and structured JSON logs for production monitoring.
-   **Advanced Qualitative Analysis:** Generates extensive qualitative samples to validate model behavior, including final targets, false negative candidates, and images with high face counts. Samples are presented in browsable HTML galleries with embedded metadata like confidence scores. See the latest results here:
    -   **[Final Targets Gallery](https://alpha-w0lf.github.io/eyeglass_finder/docs/latest_run_showcase/qualitative_analysis/final_targets/index.html)**
    -   **[False Negative Candidates Gallery](https://alpha-w0lf.github.io/eyeglass_finder/docs/latest_run_showcase/qualitative_analysis/false_negative_candidates/index.html)**
-   **Intelligent Memory Management:** Includes a dynamic `MemoryManager` that monitors memory pressure in real-time, automatically throttling data prefetching to keep runs stable during high-load processing, even with 10+ parallel workers.
-   **Two Environments:** Native Poetry for speed (GPU via MPS on Apple Silicon) and Docker for portability (CPU-only on macOS).
-   **Designed to Scale Out:** The single-machine design maps onto a distributed, queue-based microservices architecture (see [Scaling and Extensibility](#scaling-and-extensibility)).

## High-Level Architecture

The pipeline follows a decoupled, two-stage process designed for enhanced observability, flexibility, and easier debugging.

```mermaid
graph LR
    subgraph "Stage 1: Process & Enrich"
        direction TD
        A[Input Data] --> B{Data Loader};
        B --> C{Parallel Workers};
        
        subgraph "Worker Logic"
            D[Batch Images] --> E[Face Detect] --> F[Crop Faces] --> G[Classify];
        end

        C ==> D;
    end

    subgraph "Stage 2: Filter & Report"
        direction TD
        I[Intermediate Data] --> J{Artifact Generator};
        J --> K[Final Dataset];
        J --> L[Report & Visuals];
    end

    G --> I;
```

1.  **Stage 1 (`process_data.py`):** This script is responsible for the heavy lifting. It processes all input images, runs the expensive face detection and classification models, and saves all detected faces—regardless of whether they meet the target criteria—into a single, rich intermediate Parquet file. It also captures detailed metadata and performance metrics for the run.
2.  **Stage 2 (`generate_run_artifacts.py`):** This lightweight script consumes the intermediate data from Stage 1. It is responsible for all post-processing: filtering the data to produce the final dataset, generating the summary report, creating performance visualizations, and saving qualitative image samples.

This decoupled design means that if you want to change a plot or adjust a filter threshold, you only need to re-run the fast, lightweight second stage, not the time-consuming model inference stage.

## Directory Structure
```
.
├── .gitignore            # Specifies intentionally untracked files to ignore.
├── config/
│   ├── production.yaml # Tuned config used for the showcase run.
│   └── config.yaml     # Development config (1,000-row samples in data/test/).
├── data/                 # Directory for all datasets.
│   ├── raw/            # Input location for the raw Parquet files from WIT.
│   └── ...             # Other subdirectories for test or intermediate datasets.
├── docs/                 # All project documentation and planning files.
├── models/               # Face-detector weights, downloaded at first use (not committed).
├── outputs/              # Run outputs, written locally and not committed.
├── scripts/
│   ├── compare_runs.py           # Checks that two runs found the same faces.
│   ├── create_test_dataset.py    # Utility to create a smaller test set from raw data.
│   ├── generate_run_artifacts.py # Stage 2: Generates reports from intermediate data.
│   ├── precache_models.py        # Downloads the pinned face-detector weights.
│   └── process_data.py   # Stage 1: Processes images and runs models.
├── src/
│   ├── data_processing/  # Modules for data loading, chunking, and schema management.
│   ├── modeling/         # Modules for wrapping the ML models (face detector, classifier).
│   ├── processing/       # Core pipeline logic for orchestrating workers and processing.
│   ├── reporting/        # Module for generating the final Markdown report.
│   └── utils/            # Shared utilities (config, logging, metrics, visualizations).
├── tests/                # Unit tests (CPU only, no data needed), run in CI.
├── docker-compose.yml    # Orchestrates running the application and test containers.
├── Dockerfile            # Recipe for building the reproducible application container.
├── poetry.lock           # Exact dependency versions for reproducible builds.
└── pyproject.toml        # Project metadata and dependencies for Poetry.
```

## Design Rationale & Alternatives Considered

This section details the engineering rationale behind the implementation, focusing on the "why" behind key choices and documenting the alternatives that were considered and rejected. This demonstrates a deliberate and informed design process.

> **Performance Optimization Note:** The performance gains documented in this project came from systematic analysis of hardware utilization, dependency optimization, and algorithmic improvements across multiple optimization phases — detailed in [Post-Project Analysis](./post_run_report.md).

### Technology Stack Rationale

| Component                 | Technology                                       | Rationale & Alternatives Considered                                                                                                                                                                                                                            |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language                  | Python 3.12+                                     | A hard requirement from the `glasses-detector` library. Using a modern version of Python is best practice.                                                                                                                                                     |
| Containerization          | Docker & Docker Compose                          | **Decision:** Both. Docker is the portable path (CPU-only on macOS). Native Poetry is the fast path on Apple Silicon, because Docker on macOS cannot reach the GPU.          |
| Dependency Management     | Poetry                                           | **Alternative:** `pip` and `requirements.txt`. <br> **Decision:** Poetry provides superior, deterministic dependency resolution via its lock file, preventing subtle version conflicts that can break production systems. It is the modern standard for robust Python projects. |
| Core ML Framework         | PyTorch                                          | The entire pipeline runs on PyTorch. The `glasses-detector` uses it directly, and the `YOLOv8-Face` model is loaded via the `ultralytics` library, which is built on PyTorch. This provides broad hardware support (CUDA/MPS/CPU) and optimal inference speed.       |
| Data Format               | Apache Parquet                                   | **Alternative:** CSV. <br> **Decision:** Parquet is a columnar storage format offering significantly better performance and compression than CSV for large datasets. It enforces a schema, reducing the risk of data corruption.                                 |

### Architectural Rationale

#### Two-Stage Pipeline vs. Single-Stage Object Detector

A key architectural decision was to use a two-stage pipeline (detection then classification) instead of a single, end-to-end object detection model that might find "faces with glasses."

-   **Why We Chose a Two-Stage Pipeline:**
    -   **Modularity & Flexibility:** Each model can be updated, fine-tuned, or replaced independently. If a better face detector becomes available, we can swap it in without touching the classifier.
    -   **Uses Best-of-Breed Models:** It allows us to use models that are highly specialized for their specific task. `YOLOv8-Face` is a fast, accurate face detector, while `glasses-detector` is purpose-built for its classification task.
    -   **Improved Debuggability:** If an image is misclassified, it's easy to isolate the point of failure: did the detector fail to find the face, or did the classifier make an incorrect prediction? This is much harder with a single monolithic model.
    -   **Efficiency Funnel & Batch Processing:** The lightweight face detection stage acts as a rapid filter. This design is highly efficient because it allows for optimized batch processing at both stages: large batches of images are fed to the detection model, and the smaller set of resulting face crops can then be processed as a single, optimized batch by the classification models.

#### YOLOv8-Face vs. Other Face Detectors (e.g., RetinaFace, YuNet)

-   **Why We Chose `YOLOv8-Face`:**
    -   **Speed:** the nano model is small and fast, which matters when most of a 39,258-image run is spent in face detection.
    -   `YOLOv8-Face` is delivered via the `ultralytics` library, which is modern, robust, and exceptionally easy to use. This removes the significant implementation and portability issues we encountered with other libraries, making it a superior choice from an engineering perspective.
    -   The default configuration uses the **`YOLOv8n-Face` ("nano")** variant. It is smaller and faster than the larger variants. The weights are downloaded from the [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face/releases/tag/1.0.1) release on first use and checked against a pinned SHA-256 (`src/modeling/weights.py`); sources and licences are in [CREDITS.md](../CREDITS.md). To try another variant, add its URL and SHA-256 there and point `model_path` at it.
    -   The specific `YOLOv8-Face` models used in this project were developed by **[Lindevs](https://github.com/lindevs/yolov8-face)** and trained on the well-known **WIDERFace dataset**, providing a strong foundation of performance on a diverse range of faces.

#### Scalable Single-Machine Processing: A "Stream, Chunk, and Batch" Strategy
The pipeline is engineered to handle datasets that are far too large to fit into memory. Instead of loading an entire Parquet file at once, it uses a more robust and scalable approach.

-   **Why It Matters:** This design ensures that the pipeline's memory usage remains low and constant, regardless of whether the input file is 1 GB or 100 GB. It is a critical feature for processing large-scale datasets efficiently on a single machine.
-   **How It Works:**
    1.  **Stream from Disk:** The application streams the input Parquet file from disk in large, memory-efficient chunks.
    2.  **Batch for Inference:** From each chunk, smaller batches of images are created and sent to the models for inference. This is highly efficient as it maximizes the parallel processing capabilities of modern hardware (CPUs and GPUs), even in a serial execution flow.

#### A Note on Performance Monitoring
The pipeline includes a built-in monitoring suite that provides deep insight into its performance characteristics. During execution, it automatically records system-level metrics (CPU, memory, I/O) and application-level metrics (per-worker processing time). This data provides empirical evidence of the system's performance, helps identify potential bottlenecks, and validates the efficiency of the parallel processing architecture.

For example, the following plot from a production run on an M2 Max shows stable resource utilization and the initial ramp-up period, demonstrating a well-behaved system under load:

![Resource Utilization](./latest_run_showcase/visualizations/resource_utilization.png)

#### A Note on CPU-Specific Optimization
While the primary path for performance scaling is through GPU acceleration (NVIDIA CUDA or Apple MPS), we also considered advanced CPU optimization strategies. For instance, frameworks like `ONNX Runtime` can offer enhanced CPU performance through parallel execution providers. However, enabling this often requires building the runtime from source with specific flags, a process that adds significant time and complexity to the Docker build and can compromise portability.

Given that the ultimate goal for a production system is GPU-based scaling, we made a deliberate decision to prioritize a simple, clean, and highly portable Docker environment over pursuing complex, CPU-specific optimizations that offered diminishing returns for this project's scope.

#### Pre-trained `glasses-detector` vs. Training a Custom Model

-   **Why We Chose a Pre-trained Model:**
    -   The primary reason was **pragmatism and efficiency**. The open-source [`glasses-detector`](https://github.com/mantasu/glasses-detector) package directly solves the project's most difficult constraint: explicitly distinguishing between `eyeglasses` and `sunglasses`.
    -   Collecting, labeling, and cleaning a high-quality dataset to train a custom classification model from scratch is a massive undertaking, often taking weeks or months. Using an effective pre-trained model is a huge project accelerator and a common practice when available.

## Setup and Installation
This project offers two distinct, well-supported environments to balance the needs of performance and reproducibility.

### Environment Options

1.  **Native Python Environment (Recommended for Apple Silicon & Performance)**
    -   **Use Case:** The primary path for development and achieving maximum performance, especially on Apple Silicon (M1/M2/M3/M4) where it enables direct GPU access.
    -   **Performance:** With the tuning, **~21× faster** than the Docker CPU baseline (225 vs. 10.6 images/sec), using Metal Performance Shaders (MPS).
    -   **Setup:** Managed by [Poetry](https://python-poetry.org/) for robust dependency management.

2.  **Docker Environment (Recommended for Cross-Platform Reproducibility)**
    -   **Use Case:** When the same environment on every machine (Linux, Windows, Intel Macs) matters more than speed, or when sharing with collaborators.
    -   **Performance:** CPU-only on macOS due to virtualization limits. Can use NVIDIA GPUs on Linux/WSL2 with the NVIDIA Container Toolkit.
    -   **Setup:** Managed by Docker and Docker Compose (build, then run).

### Prerequisites
-   [Git](https://git-scm.com/downloads)
-   **For Native Environment:**
    -   [Python 3.12+](https://www.python.org/downloads/)
    -   [Poetry](https://python-poetry.org/docs/#installation)
-   **For Docker Environment:**
    -   [Docker](https://docs.docker.com/get-docker/)
    -   [Docker Compose](https://docs.docker.com/compose/install/) (Typically included with Docker Desktop)

> **Note:** For the Docker environment, ensure the Docker daemon is running. On macOS and Windows, this means starting Docker Desktop.

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Alpha-W0lf/eyeglass_finder.git
    cd eyeglass_finder
    ```

2.  **Download the data:**
    Download the required Parquet files from the [WIT dataset on Hugging Face](https://huggingface.co/datasets/wikimedia/wit_base/tree/main/data). The pipeline is configured to process any files matching the pattern `train-*.parquet`. For the documented runs, the following files were used:
    -   `train-00000-of-00330.parquet`
    -   `train-00001-of-00330.parquet`
    
    Place the downloaded files into the `data/raw/` directory.

3.  **Set up your chosen environment:**
    -   **For Native Environment:**
        ```bash
        poetry install
        ```
    -   **For Docker Environment:**
        No separate installation step is needed. Dependencies will be installed when the image is built.

## Running the Pipeline
The pipeline is executed differently depending on your chosen environment.

### 1. Native Python Environment (High Performance)
This is the recommended method for development and for users on Apple Silicon seeking maximum performance. Stage 1 also writes every Stage 2 artifact, so one command gives a complete run.

1.  **Run Stage 1 (Processing):**
    This script performs the heavy-lifting of model inference and creates a new, timestamped output directory (e.g., `outputs/run_2025-08-18_15-30-00/`). We recommend using the production-tuned [config/production.yaml](../config/production.yaml) configuration.
    ```bash
    poetry run python scripts/process_data.py --config config/production.yaml
    ```

2.  **Re-run Stage 2 (optional):**
    To change a threshold or a plot without a new model pass, re-run artifact generation for a run. Without `--run_dir` it uses the most recent run in the config's output folder.
    ```bash
    poetry run python scripts/generate_run_artifacts.py --config config/production.yaml --run_dir outputs/<run-id>/
    ```

### 2. Docker Environment (Reproducibility)
This method gives a consistent environment using Docker Compose (CPU only on macOS).

```bash
GIT_COMMIT_HASH=$(git rev-parse HEAD) docker compose build && docker compose run --rm app
```

> **Note on Reproducibility:** The `GIT_COMMIT_HASH` variable is a critical MLOps practice. It embeds the current Git commit hash into the Docker image, linking the output artifacts directly to the exact version of the source code that created them.

-   `build`: Builds the Docker image. Run this the first time or when dependencies in `pyproject.toml` change. To force a rebuild without cache, add the `--no-cache` flag.
-   `run --rm app`: Runs the main pipeline service inside a container. The `--rm` flag ensures the container is removed after execution to keep your system clean.

### Running the Test Suite
Unit tests live in `tests/` and run on CPU with no data or network. CI runs them with `ruff` on every pull request and every push to `main`.
```bash
poetry run pytest -q
```

### Hardware Acceleration Notes
The application code is fully hardware-aware and engineered for optimal performance on different platforms.

#### **Apple Silicon (M1/M2/M3/M4) - PERFORMANCE OPTIMIZED**
-   **Native Execution (Recommended):** Runs **~21× faster** than the Docker CPU baseline (with the tuning) by using the full capabilities of Apple Silicon, including GPU acceleration via Metal Performance Shaders (MPS) and the unified memory architecture.
-   **Docker (CPU-Only):** Provides reproducibility but is limited to CPU-only processing due to Docker's virtualization constraints on macOS.

#### **NVIDIA GPU Systems**
-   **Docker + NVIDIA Container Toolkit:** Full GPU acceleration is available on Linux/WSL2.
    -   **Requirement:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) must be installed.
    -   **Configuration:** Uncomment the `deploy` section in `docker-compose.yml` to enable GPU access for the container.
-   **Native Execution:** Also a viable path for maximum GPU utilization if the environment is configured manually.

#### **CPU-Only Systems**
-   **Docker (Recommended):** Guarantees reproducibility across all platforms (e.g., Intel Macs, Windows without WSL2, Linux without NVIDIA GPUs).
-   **Native Execution:** An alternative for development convenience.

#### **Automatic Device Selection**
The pipeline intelligently assigns models to the best available hardware in the following priority order: **NVIDIA CUDA → Apple MPS → CPU**. This ensures that you get the best possible performance automatically, with no need for manual configuration.

### Configuration and Tuning
The pipeline is controlled by a YAML config: `config/production.yaml` (tuned, used for the showcase run) or `config/config.yaml` (development, runs on the 1,000-row samples that `scripts/create_test_dataset.py` writes to `data/test/`). This approach separates the core logic from the operational parameters, allowing for easy experimentation and tuning without modifying the source code.

> **Thresholds:** both configs keep faces at least 100 px with face-detector confidence ≥ 0.75. The eyeglasses cutoff is 0.90 in `production.yaml` and 0.65 in `config.yaml`, which keeps more borderline faces for exploration.

You can easily modify this file to suit your needs. For example, to find more potential faces at the cost of including more false positives, you could lower the `min_confidence` value in the `face_detection` section.

The configuration is split into several logical sections:
-   `paths`: Defines all input and output locations.
-   `data_processing`: Controls how data is chunked and batched.
-   `model_params`: Contains all parameters for model inference, including confidence thresholds and filtering logic.
-   `hardware`: Number of worker processes (`max_workers`), Apple GPU use and an optional device override.
-   `logging`: Configures the logging level for the application.
-   `report_generation`: Controls parameters for the final report: the number of near-miss samples (`qualitative_analysis_sample_size`), the lowest eyeglasses score shown on the public finds page (`gallery_min_eyeglasses_proba`; 0 shows every find), and an optional blind visual review CSV (`gallery_visual_review_csv`) that limits that page to finds confirmed in a blind visual review.

#### Typed Config Usage
Configuration is parsed into a typed dataclass tree (`AppConfig`) for clarity and maintainability. All application code uses attribute access, not dict indexing.

Example:

```python
from src.utils.config import load_config, AppConfig

config: AppConfig = load_config("config/config.yaml")

# Attribute access throughout the codebase
input_dir = config.paths.input_dir
pattern = config.data_processing.file_pattern
num_workers = config.hardware.max_workers
fd_model_path = config.model_params.face_detection.model_path

# For serializing into metadata or logs:
from dataclasses import asdict
config_snapshot = asdict(config)
```

To increase logging verbosity at runtime, set `logging.level: "DEBUG"` in `config/config.yaml`.

## Output Dataset Schema
The pipeline's two-stage design produces two key data outputs, located in `outputs/<run-id>/`, where `<run-id>` is a unique ID for each pipeline execution. The schemas for these files are different, as they serve different purposes.

### 1. Intermediate Artifact: `annotated_faces.parquet`
This is the primary data output from the first stage of the pipeline. It is a rich, detailed dataset containing one row for **every single face** that was detected and met the minimum size criteria. This "catch-all" approach is invaluable for debugging and data analysis, as it provides a complete picture of what the model processed before final filtering. Its schema is implicitly defined by the data processing script.

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | `int` | Row id of the source image across the input files. |
| `image_url` | `string` | The original URL of the source image for traceability. |
| `source_file` | `string` | The name of the input Parquet file where the image was sourced. |
| `original_image_mode` | `string` | The Pillow image mode (e.g., "RGB", "L") of the source image. |
| `face_index` | `int` | Index of the face within its image. |
| `face_bbox` | `list[int]` | `[x_min, y_min, x_max, y_max]` coordinates of the detected face. |
| `face_size` | `list[int]` | `[width, height]` of the face in pixels. |
| `face_score` | `float` | The confidence score from the face detection model. |
| `has_eyewear_confidence` | `float` | Eyeglasses classifier score. |
| `glasses_confidence` | `float` | The same score, kept for older readers. |
| `sunglasses_prediction` | `boolean` | Always `False` while sunglasses rejection is off. |
| `is_target` | `boolean` | `True` if the score is at or above the eyeglasses cutoff. |
| `cropped_face_jpeg` | `bytes` | The face as a JPEG: a square crop with a 20% margin, resized to 224 x 224. |

Per-run detection and classification times are in `run_metadata.json` (`performance`), not per face.

### 2. Final Output: `filtered_dataset.parquet`
This is the final, user-facing dataset. It is a subset of the intermediate artifact. The artifact generation script first filters the intermediate data for rows where `is_target` is `True`, then selects and renames a subset of the columns to create a more concise final output.

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_url` | `string` | The original URL of the source image for traceability. |
| `source_file` | `string` | The name of the input Parquet file where the image was sourced. |
| `face_bbox` | `list[int]` | `[x_min, y_min, x_max, y_max]` coordinates of the detected face. |
| `face_confidence` | `float` | The confidence score from the face detection model. |
| `face_jpeg` | `bytes` | Binary data of the cropped and resized face (JPEG format). Renamed from `cropped_face_jpeg`. |

### 3. Diagnostic and Analysis Artifacts
Beyond the core datasets, each pipeline run generates comprehensive diagnostic artifacts designed to provide deep insights into face detection patterns and model behavior. These artifacts support the iterative improvement process and help validate pipeline performance.

#### Comprehensive Reporting (`report.md`)
Every run produces a detailed Markdown report featuring:
- **Executive Summary**: Human-readable face count distribution with percentages and categorization (no faces, single faces, small groups, large groups/crowd scenes)
- **Data Funnel**: Every image counted once: no face, faces, decode error, or failed (a failed mini-batch or a crashed worker chunk)
- **Performance Analytics**: Worker efficiency, system resource utilization, and processing time distributions
- **Quality Analysis**: Face detection confidence distributions, size analysis, and model performance metrics

#### Face Detection Diagnostics
The pipeline automatically identifies and analyzes face detection patterns:
- **Face Count Distribution Visualization**: Bar chart showing the distribution of faces per image with statistical overlays
- **High Face Count Image Sampling**: Automatic identification and preservation of images with >5 detected faces for manual inspection
- **Diagnostic Image Collection** (`qualitative_analysis/high_face_count_images/`): Up to 20 images with the highest face counts, saved as JPEG files with descriptive filenames indicating face count (kept locally; left out of the public showcase)
- **Diagnostic Summary** (`README.txt`): Detailed analysis document listing all high face count images with URLs and face counts for investigation

#### Qualitative Analysis Samples
- **Final Target Samples**: Representative examples of faces meeting all filtering criteria
- **Rejected Sunglasses Samples**: Faces the sunglasses classifier rejected (only when sunglasses rejection is on)
- **Visualization Suite**: Complete set of diagnostic plots including confidence histograms, face size distributions, and system performance metrics

#### Technical Artifacts
- **Intermediate Data Preservation** (`high_face_count_images.pkl`): Binary storage of high face count images with metadata for offline analysis
- **Resource Monitoring** (`resource_utilization.json`): Detailed system performance data captured throughout the run
- **Structured Metadata** (`run_metadata.json`): Complete run configuration and metrics in machine-readable format

Together these make each run easy to review visually and to compare with the next one (`scripts/compare_runs.py`).

## Scaling and Extensibility
The pipeline runs on one machine. Growing past that adds:

-   **Advanced Observability:** Monitoring for data drift and implementing model explainability.
-   **Model & Data Governance:** Using model and schema registries to ensure reproducibility and prevent pipeline failures.
-   **Cost Optimization:** Employing strategies like Spot Instances and intelligent autoscaling.

The immediate path to scale, however, begins with a shift to a distributed microservices architecture.

### Scaling Out: A Microservices Approach
To scale out, the system becomes a distributed, asynchronous set of microservices:

1.  **Model Containerization:** Each model (face detector, eyeglass classifier) would be packaged into its own Docker container, exposing a simple API endpoint (e.g., `POST /detect-faces`).
2.  **Asynchronous Task Orchestration:** A message queue (like **Apache Kafka** or **RabbitMQ**) would replace the linear `for` loop. A producer would populate a queue with image locations.
3.  **Dedicated Worker Groups:** Fleets of servers (or Kubernetes pods), each running a specific model service, would pull tasks from the queue, perform their function, and push resulting tasks onto downstream queues (e.g., a `face-classification-queue`).
4.  **Scalable Storage:** Raw data would live in a cloud object store like **Amazon S3**, and results would be written to a scalable, searchable database like **Elasticsearch** or a properly indexed **PostgreSQL**.
5.  **Deployment & Management:** The entire ecosystem would be managed by an orchestrator like **Kubernetes**, which can automatically scale the number of worker pods based on queue length.

### Designed for Extensibility
This microservices architecture is inherently extensible. Adding a new filter—for example, a "beard detector"—would require **zero modification to the existing pipeline**. The process would be:
1.  Train or acquire a beard detection model.
2.  Package it into a new `Beard Classifier` microservice.
3.  Deploy a new group of workers running this service.
4.  Subscribe these workers to the exact same `face-classification-queue`.

Both the eyeglass workers and beard workers would then process face crops in parallel, writing their respective results to the database. This design enables the rapid and independent addition of new filtering capabilities.

## Next Steps
The rich output artifacts from each run feed a structured improvement loop:

-   **Fine-tune for the WIT domain:** The pre-trained models were trained on cleaner portrait data than WIT's in-the-wild images. A phased plan starts with last-layer tuning and moves to full fine-tuning as a labelled set grows.
-   **Human-in-the-loop review:** Route low-confidence predictions and near misses to a reviewer. Each reviewed example becomes new training data for the fine-tuning above.
-   **Better face processing:** Add a landmark model for face alignment and head-pose estimation to skip extreme angles.
-   **Use the text metadata:** WIT's captions are not used yet. Text heuristics (for example a caption that says "sunglasses") or a multimodal model such as CLIP can filter images by meaning as well as pixels.

## Project Links
*(Primary repository link.)*

- **GitHub Repository:** https://github.com/Alpha-W0lf/eyeglass_finder
