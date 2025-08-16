# Eyeglass Finder – Image Filtering Pipeline

This document provides a comprehensive overview of a production‑grade, fully reproducible data pipeline designed to identify faces and classify eyewear. It emphasizes reproducibility, scalability, and maintainability.

> ### tldr
>
> -   **The Task:** Process the WIT dataset to find all images of people wearing **eyeglasses** (not sunglasses) with faces at least 100x100 pixels.
> -   **The Solution:** A fully containerized, two-stage pipeline that detects faces, classifies them, and generates a clean, final dataset.
> -   **Run with a Single Command:** The entire system is orchestrated with Docker Compose for perfect reproducibility.
>     ```bash
>     GIT_COMMIT_HASH=$(git rev-parse HEAD) docker compose build && docker compose run --rm app
>     ```
> -   **Key Highlights:**
>     -   **Fully Reproducible & Portable:** Runs identically on any machine with Docker.
>     -   **Hardware-Optimized Performance:** Native execution on Apple Silicon provides 3-5x performance improvement (30-50 vs 9.8 images/second) through MPS GPU acceleration. Automatically uses NVIDIA (CUDA) or Apple (MPS) GPUs if available, falling back to CPU.
>     -   **Comprehensive Reporting:** Every run produces a unique output directory with a detailed report, performance metrics, visualizations, and qualitative samples.
>     -   **Built for Scale:** The architecture is designed with a clear roadmap to a distributed microservices system.
>
> This project emphasizes not just a functional solution, but the engineering discipline required for a scalable and maintainable research environment.

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
  - [Running the Application](#running-the-application)
  - [Running the Test Suite](#running-the-test-suite)
  - [Alternative Method: Native Python Environment](#alternative-method-native-python-environment)
  - [Hardware Acceleration Notes](#hardware-acceleration-notes)
  - [Configuration and Tuning](#configuration-and-tuning)
- [Output Dataset Schema](#output-dataset-schema)
  - [1. Intermediate Artifact: `annotated_faces.parquet`](#1-intermediate-artifact-annotated_facesparquet)
  - [2. Final Output: `filtered_dataset.parquet`](#2-final-output-filtered_datasetparquet)
  - [3. Diagnostic and Analysis Artifacts](#3-diagnostic-and-analysis-artifacts)
- [Scaling and Extensibility](#scaling-and-extensibility)
  - [The Path to Billions of Images: A Microservices Approach](#the-path-to-billions-of-images-a-microservices-approach)
  - [Designed for Extensibility](#designed-for-extensibility)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Project Links](#project-links)
- [Known Security Issues](#known-security-issues)

---

---

## Project Overview
The primary goal of this project is to build a robust and scalable data processing pipeline that filters a large image dataset for highly specific content. It processes image data from the Wikipedia-based Image Text (WIT) dataset, identifies human faces, and classifies them based on whether they are wearing eyeglasses.

The pipeline is engineered to solve a multi-stage ML problem, separating heavy processing from final artifact generation:

1.  **Stage 1: Process & Enrich:**
    -   **Ingest** images from the source dataset.
    -   **Detect** all human faces and their corresponding facial landmarks, regardless of size.
    -   **Crop and resize** every detected face to create a standardized input for classification.
    -   **Classify** each cropped face to distinguish between **eyeglasses** and **sunglasses**.
    -   **Persist** this rich, unfiltered data to an intermediate file.
2.  **Stage 2: Filter & Report:**
    -   **Load** the intermediate data from Stage 1.
    -   **Filter** the faces based on project criteria (e.g., minimum size, classification results).
    -   **Output** a clean, well-structured, and searchable final dataset for researchers, alongside reports and visualizations.

> **Note on Classification Model Performance:** Analysis has revealed that the sunglasses classifier may be overly aggressive, potentially removing valid eyeglasses results at a high rate. Additionally, the primary eyeglasses classifier may already be filtering out sunglasses, making the dedicated sunglasses classifier redundant. Further investigation is recommended (see detailed analysis in the Limitations section).

This project emphasizes not only the functional outcome but also the engineering discipline required for a research environment. It is designed to be reproducible, well-documented, and built upon a foundation that is both scalable and extensible.

## Key Features
-   **Reproducible Environment:** Fully containerized with Docker and orchestrated by Docker Compose, ensuring a consistent, one-command setup that runs identically across any machine.
-   **Hardware-Agnostic Execution:** The application automatically detects and leverages available hardware accelerators (NVIDIA CUDA, Apple MPS) for significant performance gains, while seamlessly falling back to CPU-only execution.
-   **Modular Two-Stage Pipeline:** Decouples the computationally expensive model processing (Stage 1) from the lightweight filtering and artifact generation (Stage 2). This design enhances flexibility, simplifies debugging, and allows reporting logic to be changed without re-running the entire pipeline.
-   **Robust Face Processing:** Detects faces and extracts their corresponding 5-point facial landmarks. This allows for potential downstream tasks like alignment, while the current implementation uses a robust crop-and-resize strategy that is resilient to faces near image borders.
-   **Dual-Classifier Filtering:** Employs two separate classification models in an attempt to explicitly distinguish between `eyeglasses` and `sunglasses`. This design allows for granular control over the final filtering logic.
-   **Comprehensive Observability & Reporting:** Every pipeline run is treated as a reproducible experiment. The system automatically generates a unique, timestamped output directory containing a detailed Markdown report with executive summaries, enhanced data funnels, and comprehensive diagnostic capabilities. This includes deep performance analysis, visualizations for system resource utilization (CPU, Memory, Disk I/O), parallel worker efficiency, face detection pattern analysis, high face count image sampling for quality investigation, qualitative image samples, and structured logs. This rich set of artifacts provides unparalleled insight into the system's behavior and supports iterative improvement.
-   **Enterprise-Grade Data Integrity:** The pipeline includes comprehensive data tracking and verification systems that ensure perfect accountability for every processed image. Enhanced error handling with retry logic, detailed failure tracking across multiple dimensions, and systematic verification checkpoints guarantee that no images are lost or unaccounted for during processing, providing the level of data integrity required for production MLOps systems.
-   **Designed for Scale & Extensibility:** The current single-machine pipeline is built with a clear, documented path toward a distributed microservices architecture capable of processing billions of images and videos.
-   **Comprehensive Documentation:** Meticulously documented to support asynchronous collaboration, ensuring that any developer can understand the project's architecture, decisions, and operations with ease.

## High-Level Architecture

The pipeline follows a decoupled, two-stage process designed for enhanced observability, flexibility, and easier debugging.

1.  **Stage 1 (`process_data.py`):** This script is responsible for the heavy lifting. It processes all input images, runs the expensive face detection and classification models, and saves all detected faces—regardless of whether they meet the target criteria—into a single, rich intermediate Parquet file. It also captures detailed metadata and performance metrics for the run.
2.  **Stage 2 (`generate_run_artifacts.py`):** This lightweight script consumes the intermediate data from Stage 1. It is responsible for all post-processing: filtering the data to produce the final dataset, generating the summary report, creating performance visualizations, and saving qualitative image samples.

This decoupled design means that if you want to change a plot or adjust a filter threshold, you only need to re-run the fast, lightweight second stage, not the time-consuming model inference stage.

## Directory Structure
```
.
├── .dockerignore         # Specifies files to exclude from the Docker build context.
├── .gitignore            # Specifies intentionally untracked files to ignore.
├── config/
│   └── config.yaml     # Central configuration file for the entire pipeline.
├── data/                 # Directory for all datasets.
│   ├── raw/            # Input location for the raw Parquet files from WIT.
│   └── ...             # Other subdirectories for test or intermediate datasets.
├── docs/                 # All project documentation and planning files.
├── models/               # Pre-trained model weights.
│   ├── yolov8n-face-lindevs.pt # The "nano" variant of the YOLOv8-Face model.
│   └── yolov8s-face.pt # The "small" variant of the YOLOv8-Face model.
├── notebooks/
│   └── .gitkeep        # For exploratory data analysis (EDA) and model prototyping.
├── outputs/
│   └── .gitkeep        # Default output directory for all pipeline runs.
├── scripts/
│   ├── create_test_dataset.py    # Utility to create a smaller test set from raw data.
│   ├── generate_run_artifacts.py # Stage 2: Generates reports from intermediate data.
│   └── process_data.py   # Stage 1: Processes images and runs models.
├── src/
│   ├── data_processing/  # Modules for data loading, chunking, and schema management.
│   ├── modeling/         # Modules for wrapping the ML models (face detector, classifier).
│   ├── processing/       # Core pipeline logic for orchestrating workers and processing.
│   ├── reporting/        # Module for generating the final Markdown report.
│   └── utils/            # Shared utilities (config, logging, metrics, visualizations).
├── tests/                # Automated test suite (currently obsolete).
│   └── ...
├── docker-compose.yml    # Orchestrates running the application and test containers.
├── Dockerfile            # Recipe for building the reproducible application container.
├── poetry.lock           # Exact dependency versions for reproducible builds.
├── pyproject.toml        # Project metadata and dependencies for Poetry.
└── pytest.ini            # Configuration for the Pytest testing framework.
```

## Design Rationale & Alternatives Considered

This section details the engineering rationale behind the implementation, focusing on the "why" behind key choices and documenting the alternatives that were considered and rejected. This demonstrates a deliberate and informed design process.

> **Performance Optimization Note:** For detailed analysis of performance optimization opportunities and the strategic shift to native execution on Apple Silicon, see **[optimization_notes.md](./docs/optimization_notes.md)**. This document provides comprehensive analysis of hardware utilization, dependency optimization, and algorithmic improvements that can achieve 3-5x performance improvements.

### Technology Stack Rationale

| Component                 | Technology                                       | Rationale & Alternatives Considered                                                                                                                                                                                                                            |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language                  | Python 3.12+                                     | A hard requirement from the `glasses-detector` library. Using a modern version of Python is best practice.                                                                                                                                                     |
| Containerization          | Docker & Docker Compose                          | **Alternative:** Native local environment. <br> **Decision:** Rejected for being non-reproducible. Containerization guarantees a perfectly consistent environment across any machine, solving the "works on my machine" problem, which is critical for MLOps.          |
| Dependency Management     | Poetry                                           | **Alternative:** `pip` and `requirements.txt`. <br> **Decision:** Poetry provides superior, deterministic dependency resolution via its lock file, preventing subtle version conflicts that can break production systems. It is the modern standard for robust Python projects. |
| Core ML Framework         | PyTorch                                          | The entire pipeline leverages PyTorch. The `glasses-detector` uses it directly, and the `YOLOv8-Face` model is loaded via the `ultralytics` library, which is built on PyTorch. This provides broad hardware support (CUDA/MPS/CPU) and optimal inference speed.       |
| Data Format               | Apache Parquet                                   | **Alternative:** CSV. <br> **Decision:** Parquet is a columnar storage format offering significantly better performance and compression than CSV for large datasets. It enforces a schema, reducing the risk of data corruption.                                 |

> For a detailed justification of our model architecture and specific model choices, please see **[model_selection.md](./model_selection.md)**.

### Architectural Rationale

#### Two-Stage Pipeline vs. Single-Stage Object Detector

A key architectural decision was to use a two-stage pipeline (detection then classification) instead of a single, end-to-end object detection model that might find "faces with glasses."

-   **Why We Chose a Two-Stage Pipeline:**
    -   **Modularity & Flexibility:** Each model can be updated, fine-tuned, or replaced independently. If a better face detector becomes available, we can swap it in without touching the classifier.
    -   **Leverages Best-of-Breed Models:** It allows us to use models that are highly specialized for their specific task. `YOLOv8-Face` is excellent at finding faces and landmarks, while `glasses-detector` is purpose-built for its classification task.
    -   **Improved Debuggability:** If an image is misclassified, it's easy to isolate the point of failure: did the detector fail to find the face, or did the classifier make an incorrect prediction? This is much harder with a single monolithic model.
    -   **Efficiency Funnel & Batch Processing:** The lightweight face detection stage acts as a rapid filter. This design is highly efficient because it allows for optimized batch processing at both stages: large batches of images are fed to the detection model, and the smaller set of resulting face crops can then be processed as a single, optimized batch by the classification models.

#### YOLOv8-Face vs. Other Face Detectors (e.g., RetinaFace, YuNet)

-   **Why We Chose `YOLOv8-Face`:**
    -   The single most important reason was its ability to predict **5 facial landmarks** (eye centers, nose tip, mouth corners) in the same forward pass as detection. This provides a rich set of data for each face and was a key factor in its selection, offering capabilities on par with other advanced detectors like `RetinaFace`.
    -   While these landmarks can be used for advanced applications like face alignment, the current pipeline uses a more robust **crop-and-resize** strategy. An earlier implementation that used an affine transformation for alignment was found to be brittle, especially for faces near image borders. The availability of landmarks, however, provides a clear path for future enhancements.
    -   `YOLOv8-Face` is delivered via the `ultralytics` library, which is modern, robust, and exceptionally easy to use. This removes the significant implementation and portability issues we encountered with other libraries, making it a superior choice from an engineering perspective.
    -   The default configuration uses the **`YOLOv8n-Face` ("nano")** variant. It is significantly smaller and faster than the "small" variant, and the minor trade-off in accuracy is acceptable for this project's requirements. That said, to maximize portability and flexibility, both the "nano" and "small" models are included in the repository and can be easily switched via `config.yaml`.
    -   The specific `YOLOv8-Face` models used in this project were developed by **[Lindevs](https://github.com/lindevs/yolov8-face)** and trained on the well-known **WIDERFace dataset**, providing a strong foundation of performance on a diverse range of faces.

#### Scalable Single-Machine Processing: A "Stream, Chunk, and Batch" Strategy
The pipeline is engineered to handle datasets that are far too large to fit into memory. Instead of loading an entire Parquet file at once, it uses a more robust and scalable approach.

-   **Why It Matters:** This design ensures that the pipeline's memory usage remains low and constant, regardless of whether the input file is 1 GB or 100 GB. It is a critical feature for processing large-scale datasets efficiently on a single machine.
-   **How It Works:**
    1.  **Stream from Disk:** The application streams the input Parquet file from disk in large, memory-efficient chunks.
    2.  **Batch for Inference:** From each chunk, smaller batches of images are created and sent to the models for inference. This is highly efficient as it maximizes the parallel processing capabilities of modern hardware (CPUs and GPUs), even in a serial execution flow.

#### A Note on Performance Monitoring
The pipeline includes a built-in monitoring suite that provides deep insight into its performance characteristics. During execution, it automatically records:
- **System-level metrics:** CPU utilization, memory consumption, and disk I/O rates are sampled every second.
- **Application-level metrics:** The processing time for every data chunk is recorded for each parallel worker.

This data is saved to the output directory and used to generate detailed visualizations in the final report, including time-series plots of resource usage and histograms of worker performance. This provides empirical evidence of the system's performance, helps identify potential bottlenecks (e.g., I/O vs. CPU-bound), and validates the efficiency of the parallel processing architecture.

#### A Note on CPU-Specific Optimization
While the primary path for performance scaling is through GPU acceleration (NVIDIA CUDA or Apple MPS), we also considered advanced CPU optimization strategies. For instance, frameworks like `ONNX Runtime` can offer enhanced CPU performance through parallel execution providers. However, enabling this often requires building the runtime from source with specific flags, a process that adds significant time and complexity to the Docker build and can compromise portability.

Given that the ultimate goal for a production system is GPU-based scaling, we made a deliberate decision to prioritize a simple, clean, and highly portable Docker environment over pursuing complex, CPU-specific optimizations that offered diminishing returns for this project's scope.

#### Pre-trained `glasses-detector` vs. Training a Custom Model

-   **Why We Chose a Pre-trained Model:**
    -   The primary reason was **pragmatism and efficiency**. The open-source [`glasses-detector`](https://github.com/mantasu/glasses-detector) package directly solves the project's most difficult constraint: explicitly distinguishing between `eyeglasses` and `sunglasses`.
    -   Collecting, labeling, and cleaning a high-quality dataset to train a custom classification model from scratch is a massive undertaking, often taking weeks or months. Using an effective pre-trained model is a huge project accelerator and a common practice when available.

## Setup and Installation
This project is fully containerized with Docker, so setup is straightforward on any operating system.

### Prerequisites
-   [Git](https://git-scm.com/downloads)
-   **Docker Environment:**
       - [Docker](https://docs.docker.com/get-docker/)
       - [Docker Compose](https://docs.docker.com/compose/install/) (Typically included with Docker Desktop)
       - **Note:** Provides reproducibility but CPU-only processing on macOS
   **Native Environment (Recommended for Apple Silicon):**
       - [Python 3.12+](https://www.python.org/downloads/)
       - [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)
       - **Performance:** 3-5x faster on M2 Max via GPU acceleration (30-50 vs 9.8 images/second)

> **Note:** Before proceeding, ensure the Docker daemon is running (if you are using the Docker method). On macOS and Windows, this typically means starting the Docker Desktop application. On most Linux distributions, the Docker service runs automatically after installation.

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

## Running the Pipeline
The entire data processing pipeline is executed through Docker Compose.

### Running the Application
This command will build the Docker image, create a container, and run the main processing script. This is the recommended method as it guarantees a perfectly reproducible environment.

```bash
GIT_COMMIT_HASH=$(git rev-parse HEAD) docker compose build && docker compose run --rm app
```

> **Note on Reproducibility:** The `GIT_COMMIT_HASH` variable in this command is essential. It embeds the current Git commit hash into the Docker image at build time. This is a critical MLOps practice that makes every run fully reproducible by linking the output artifacts directly back to the exact version of the source code that created them.

-   `build`: Builds the Docker image, installing dependencies and embedding the Git hash. Run this the first time or when dependencies change.
-   `run --rm app`: Runs the main pipeline service. The `--rm` flag removes the container after execution.

> **Note on Re-building the Image:** If you change the `Dockerfile` or dependencies in `pyproject.toml`, you must rebuild the image. To force a rebuild without using Docker's cache, include the `--no-cache` flag:
>
>```bash
>GIT_COMMIT_HASH=$(git rev-parse HEAD) docker compose build --no-cache && docker compose run --rm app
>```

The script will process the raw data from `data/raw/`. All artifacts from the run—including logs, reports, and the final filtered dataset—will be saved into a unique, timestamped subdirectory within `outputs/`.

### Running the Test Suite
> **Warning: Obsolete Test Suite**
> The test suite for this project has not been updated to reflect the latest significant refactors of the pipeline. As a result, the tests are currently failing and should not be considered a reliable measure of the application's correctness. The test files remain in the `tests/` directory as a foundation for a future refactoring effort, but they should not be run until they are brought up to date with the current codebase.

To run the test suite (which is currently expected to fail), you can use the following command:
```bash
docker compose run --rm test
```

### Native Python Environment (Recommended for Apple Silicon)
> **Performance Note:** For **Apple Silicon (M1/M2/M3/M4) users**, this native execution method is **strongly recommended** as it provides **3-5x performance improvement** through GPU acceleration via Metal Performance Shaders (MPS). Docker on macOS cannot access the GPU due to virtualization limitations.

This method leverages the full capabilities of Apple Silicon hardware, including GPU acceleration, unified memory architecture, and optimized inference performance.

1.  **Install Dependencies:**
    First, ensure you have Poetry installed. Then, from the project's root directory, run the following command to create a virtual environment and install all required dependencies from the `pyproject.toml` file.
    ```bash
    poetry install
    ```

2.  **Run the Pipeline:**
    Once the dependencies are installed, you can run the processing script using `poetry run`. This command ensures the script executes within the correct virtual environment. The two stages must be run in sequence.

    First, run the processing stage. This script will create a new, timestamped output directory (e.g., `outputs/run_2024-05-21_15-30-00/`) for all the results.
    ```bash
    poetry run python scripts/process_data.py
    ```
    Then, run the artifact generation stage. By default, this script automatically finds and processes the **most recent** run directory inside `outputs/`.
    ```bash
    poetry run python scripts/generate_run_artifacts.py
    ```

    If you need to re-run the artifact generation for a **specific** previous run, you must pass its path directly using the `--run_dir` argument.
    ```bash
    # Example for re-generating artifacts for a specific run
    poetry run python scripts/generate_run_artifacts.py --run_dir outputs/run_2024-05-21_15-30-00/
    ```

### Hardware Acceleration Notes
The application code is fully hardware-aware with optimized execution paths for maximum performance on different platforms.

#### **Apple Silicon (M1/M2/M3/M4) - PERFORMANCE OPTIMIZED**
**For M2 Max and similar hardware, native execution is strongly recommended for optimal performance:**

-   **Native Execution (Recommended):** Provides **3-5x performance improvement** through full GPU acceleration via Metal Performance Shaders (MPS)
    -   **GPU Utilization:** YOLOv8-Face and glasses-detector models automatically leverage Apple GPU
    -   **Unified Memory:** Optimized data flow with Apple Silicon's unified memory architecture
    -   **Expected Throughput:** 30-50 images/second (vs 9.8 images/second CPU-only)
    -   **Setup:** Use Poetry environment as described in Alternative Method section

-   **Docker (Available):** Provides reproducibility but **CPU-only processing**
    -   **Limitation:** Docker on macOS cannot access GPU due to virtualization constraints
    -   **Use Case:** Development, testing, or when reproducibility is more important than performance

#### **NVIDIA GPU Systems**
-   **Docker + NVIDIA Container Toolkit:** GPU acceleration available on Linux/WSL2
    -   **Requirement:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) must be installed
    -   **Configuration:** Uncomment the `deploy` section in docker-compose.yml
-   **Native Execution:** Alternative path for maximum GPU utilization

#### **CPU-Only Systems**
-   **Docker (Recommended):** Guaranteed reproducibility across all platforms
-   **Native Execution:** Alternative for development convenience

#### **Automatic Device Selection**
The pipeline intelligently assigns models to the best available hardware:
-   **Priority Order:** NVIDIA CUDA → Apple MPS → CPU
-   **YOLOv8-Face:** Managed by `ultralytics` library with automatic device detection
-   **Glasses-detector:** Uses custom utility (`src/utils/device.py`) for optimal device selection

> **A Note on the Future of Containers on macOS:** We are aware that at WWDC 2025, Apple announced a native Containerization framework for future versions of macOS. Once this technology matures, it is expected to provide direct GPU access to containers, which would eliminate the need for the native Python workaround for Apple Silicon users. This project is well-positioned to adopt this new framework, which would further simplify cross-platform development and deployment.

### Configuration and Tuning
The entire behavior of the pipeline is controlled by a single, well-documented configuration file located at `config/config.yaml`. This approach separates the core logic from the operational parameters, allowing for easy experimentation and tuning without modifying the source code.

> **A Note on the Default Settings**
>
> The default values in `config.yaml` have been carefully chosen to ensure a high-quality output and a smooth, successful first-run experience on a wide variety of hardware. They generally prioritize precision over recall (e.g., higher confidence thresholds) to build a clean, reliable dataset.

You can easily modify this file to suit your needs. For example, to find more potential faces at the cost of including more false positives, you could lower the `min_confidence` value in the `face_detection` section.

The configuration is split into several logical sections:
-   `paths`: Defines all input and output locations.
-   `data_processing`: Controls how data is chunked and batched.
-   `model_params`: Contains all parameters for model inference, including confidence thresholds and filtering logic.
-   `execution`: Manages parallel processing settings.
-   `logging`: Configures the logging level for the application.
-   `report_generation`: Controls parameters for the final report, like the number of qualitative samples.

#### Typed Config Usage (Senior-friendly API)
Configuration is parsed into a typed dataclass tree (`AppConfig`) for clarity and maintainability. All application code uses attribute access, not dict indexing.

Example:

```python
from src.utils.config import load_config, AppConfig

config: AppConfig = load_config("config/config.yaml")

# Attribute access throughout the codebase
input_dir = config.paths.input_dir
pattern = config.data_processing.file_pattern
num_workers = config.execution.num_workers
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
| `image_url` | `string` | The original URL of the source image for traceability. |
| `source_file` | `string` | The name of the input Parquet file where the image was sourced. |
| `image_width` | `int` | The width of the original source image in pixels. |
| `image_height` | `int` | The height of the original source image in pixels. |
| `original_image_mode` | `string` | The Pillow image mode (e.g., "RGB", "L") of the source image. |
| `face_bbox` | `list[int]` | `[x_min, y_min, x_max, y_max]` coordinates of the detected face. |
| `face_width` | `float` | The width of the face bounding box in pixels. |
| `face_height` | `float` | The height of the face bounding box in pixels. |
| `face_score` | `float` | The confidence score from the face detection model. |
| `eyeglasses_prediction` | `boolean` | Final prediction from the `eyeglasses` classifier. |
| `sunglasses_prediction` | `boolean` | Final prediction from the `sunglasses` classifier. |
| `is_target` | `boolean` | The final decision: `True` if the face is the target class. |
| `cropped_face_jpeg` | `bytes` | Binary data of the cropped and resized face (JPEG format). |
| `detection_time_seconds`| `float` | The per-face time spent in the detection stage. |
| `classification_time_seconds`| `float` | The per-face time spent in the classification stage. |


### 2. Final Output: `filtered_dataset.parquet`
This is the final, user-facing dataset. It is a cleaned and validated version of the intermediate artifact. The artifact generation script first filters the intermediate data for rows where `is_target` is `True`, then selects and renames a subset of the columns to create a more concise final output.

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_url` | `string` | The original URL of the source image for traceability. |
| `source_file` | `string` | The name of the input Parquet file where the image was sourced. |
| `face_bbox` | `object` | `[x_min, y_min, x_max, y_max]` coordinates of the detected face. |
| `face_confidence` | `float` | The confidence score from the face detection model. |
| `face_jpeg` | `bytes` | Binary data of the cropped and resized face (JPEG format). Renamed from `cropped_face_jpeg`. |

### 3. Diagnostic and Analysis Artifacts
Beyond the core datasets, each pipeline run generates comprehensive diagnostic artifacts designed to provide deep insights into face detection patterns and model behavior. These artifacts support the iterative improvement process and help validate pipeline performance.

#### Comprehensive Reporting (`report.md`)
Every run produces a detailed Markdown report featuring:
- **Executive Summary**: Human-readable face count distribution with percentages and categorization (no faces, single faces, small groups, large groups/crowd scenes)
- **Enhanced Data Funnel**: Complete processing pipeline metrics including the new "Images with Faces" metric for improved clarity
- **Performance Analytics**: Worker efficiency, system resource utilization, and processing time distributions
- **Quality Analysis**: Face detection confidence distributions, size analysis, and model performance metrics

#### Face Detection Diagnostics
The pipeline automatically identifies and analyzes face detection patterns:
- **Face Count Distribution Visualization**: Bar chart showing the distribution of faces per image with statistical overlays
- **High Face Count Image Sampling**: Automatic identification and preservation of images with >5 detected faces for manual inspection
- **Diagnostic Image Collection** (`qualitative_analysis/high_face_count_images/`): Up to 20 images with the highest face counts, saved as JPEG files with descriptive filenames indicating face count
- **Diagnostic Summary** (`README.txt`): Detailed analysis document listing all high face count images with URLs and face counts for investigation

#### Qualitative Analysis Samples
- **Final Target Samples**: Representative examples of faces meeting all filtering criteria
- **Rejected Sunglasses Samples**: Examples of faces rejected by the sunglasses classifier for failure analysis
- **Visualization Suite**: Complete set of diagnostic plots including confidence histograms, face size distributions, and system performance metrics

#### Technical Artifacts
- **Intermediate Data Preservation** (`high_face_count_images.pkl`): Binary storage of high face count images with metadata for offline analysis
- **Resource Monitoring** (`resource_utilization.json`): Detailed system performance data captured throughout the run
- **Structured Metadata** (`run_metadata.json`): Complete run configuration and metrics in machine-readable format

This comprehensive diagnostic framework transforms each pipeline run into a rich source of insights, supporting both immediate quality evaluation and long-term iterative improvement of the face detection and classification models.

## Scaling and Extensibility
This project was intentionally designed as a robust foundation that can be evolved into a production-grade, web-scale system. A detailed strategy for this evolution is documented in the **[Production Roadmap](./production_roadmap.md)**. The roadmap outlines a multi-phase plan that goes beyond simply scaling the architecture and addresses the key pillars of enterprise-ready MLOps, including:

-   **Advanced Observability:** Monitoring for data drift and implementing model explainability.
-   **Model & Data Governance:** Using model and schema registries to ensure reproducibility and prevent pipeline failures.
-   **Cost Optimization:** Employing strategies like Spot Instances and intelligent autoscaling.

The immediate path to scale, however, begins with a shift to a distributed microservices architecture.

### The Path to Billions of Images: A Microservices Approach
The current single-script pipeline is not suitable for processing billions of files. To handle that scale, we would evolve this system into a distributed, asynchronous architecture based on microservices. This is the industry standard for resilient MLOps systems.

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

## Limitations and Future Improvements
A robust engineering project acknowledges its limitations and outlines a clear path for future work. This section discusses the outcomes of the current pipeline and proposes strategic next steps.

-   **Input Dataset Characteristics — A Challenging Real-World Problem:** The pipeline was successfully executed on a sample of the `wikimedia/wit_base` dataset, a collection known for its vast and uncurated nature. This task was made significantly more challenging by the dataset's own curation process: for privacy reasons, the original creators had already removed images where human faces were a primary subject. This left our pipeline to solve a true "needle in a haystack" problem, searching for the subtle or less prominent faces that remained. Processing **39,258 images** to find **67 target faces** (~0.17% success rate) not only confirms the pipeline is functional but also demonstrates its robustness in extracting a rare signal from a noisy, pre-filtered source.

-   **Face Detection Count Investigation — Systematic Analysis Completed:** Pipeline runs initially detected **24,499 total faces** across 39,258 processed images, which appeared to contradict expectations for a dataset pre-filtered to remove prominent faces. **Systematic investigation revealed the explanation**: many images contain multiple faces, with some images containing over 100 faces (likely crowd scenes, group photos, events, etc.). This finding validates the face detection model's accuracy and explains the mathematical discrepancy between expected single-face images and actual multi-face reality. The pipeline includes **comprehensive diagnostic capabilities** that automatically identify and preserve high face count images (>5 faces) for manual inspection, generate detailed face count distribution analyses with executive summaries, and provide qualitative samples for investigation. These diagnostic tools enabled the systematic validation that confirmed detection accuracy and resolved the initial concern. **The latest implementation achieved perfect data integrity with all 39,258 input images properly tracked and accounted for in both the data funnel and diagnostic statistics.**

-   **Model Accuracy and Bias:** The pre-trained models, while effective, are not perfect. There will inevitably be false positives (e.g., reflections misidentified as glasses) and false negatives. The confidence scores are provided in the output data specifically to allow researchers to set their own precision/recall thresholds. Furthermore, because the WIT dataset is sourced from Wikipedia, it has known demographic and geographic biases, which will be reflected in the models' performance and the final filtered dataset.

-   **Questionable Efficacy of the Sunglasses Classifier:** Post-run analysis indicates that the secondary `sunglasses` classifier may be overly aggressive, rejecting a significant number of valid eyeglass images. This also raises the possibility that the primary `eyeglasses` classifier is already trained to implicitly exclude sunglasses, making the second filter redundant and potentially harmful to recall. A key future investigation would be to analyze the outputs of each classifier independently to determine if a single-model approach is sufficient and more effective.

-   **Strategic Next Steps & A Vision for Iterative Improvement:** The current system provides a strong foundation designed for continuous evolution. The rich output artifacts (reports, visualizations, qualitative samples) are not just for a one-time analysis; they are the fuel for a **structured, iterative improvement loop**. This process of using evidence from each run to inform the next cycle is critical in a research environment. A detailed roadmap for this is available in our **[Enhancements Plan](./enhancements_plan.md)**, but key initiatives would include:
    -   **Address the Model Domain Gap via Fine-Tuning:** The accuracy of the pre-trained models is likely constrained by a "domain gap"—a mismatch between the data they were originally trained on (likely clean, well-lit portraits) and the "in-the-wild" nature of the WIT dataset. To mitigate this, a formal fine-tuning strategy is a high-priority next step. This could be approached in phases:
        -   **Simple Last-Layer Tuning:** A fast and efficient starting point would be to freeze the majority of the model's layers and retrain only the final classification layer on a curated set of examples from our dataset. This often yields significant gains with minimal risk of damaging the model's core learned features.
        -   **Full Model Fine-Tuning:** For greater accuracy, a more advanced approach involves unfreezing the entire model and retraining it on our data using a very low learning rate. This allows the model to adapt all of its parameters to the new domain but requires more data and careful tuning to be successful.
    -   **Implement a Human-in-the-Loop (HITL) Workflow:** A more advanced MLOps system would include a human review process for low-confidence predictions. These reviewed examples would create a continuous stream of new, high-quality training data, providing the ideal fuel for the fine-tuning efforts described above.
    -   **Activate the Full Test Suite:** The existing test suite is currently obsolete due to a major architectural refactor. A high-priority next step would be to update the tests to align with the current two-stage pipeline, ensuring long-term reliability and maintainability.
    -   **Enhanced Preprocessing and Multimodal Filtering:** Beyond model-centric improvements, significant gains could be achieved by enhancing the data preprocessing pipeline itself. Future iterations could incorporate:
        -   **Pre-computation Filtering:** Implementing lightweight checks for image quality (e.g., blur and contrast detection) and content type (e.g., filtering out diagrams or logos) to avoid processing irrelevant images.
        -   **Advanced Face Processing:** Developing a more robust face alignment strategy using the available landmarks and adding head pose estimation to filter out faces at extreme angles, which are difficult to classify reliably.
        -   **Leveraging Textual Metadata:** The WIT dataset's rich textual metadata is currently unused. A powerful enhancement would be to implement text-based heuristics (e.g., flagging images with the caption "sunglasses") or evolve the system toward a true multimodal approach using models like CLIP to filter images based on the semantic similarity between the image and text descriptions.
        -   **Formalize Support for Native and GPU Environments:** While the pipeline can be run in a native Python environment to leverage GPU acceleration, a future workstream should be dedicated to creating a comprehensive test suite and formally validating the stability and performance of the native and GPU execution paths.

## Project Links
*(Primary repository link.)*

- **GitHub Repository:** https://github.com/Alpha-W0lf/eyeglass_finder

---

## Known Security Issues

- **`torch==2.3.1` Vulnerability**: The project currently uses `torch==2.3.1`, which has a known security vulnerability. We are locked into this specific version because the `torchvision==0.18.1` package, a critical dependency for the models used in this pipeline, has a strict requirement for it. Upgrading `torch` independently would break the environment and could compromise the validity of the model's predictions. We will continue to monitor the upstream packages and will upgrade to a secure version as soon as a compatible and stable release of `torchvision` becomes available.
