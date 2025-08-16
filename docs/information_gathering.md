# Information Gathering

## Tooling and Dependencies

- Python: 3.12+
- Dependency management: Poetry
- Containerization: Docker, Docker Compose (with build arg for git hash)
- Core libs: torch, torchvision, ultralytics, pandas, pyarrow, pandera, opencv-python, Pillow, numpy, tqdm, loguru, seaborn/matplotlib, psutil

## Environments

### Apple Silicon (M1/M2/M3/M4) - PERFORMANCE OPTIMIZED
- **Native (recommended):** Poetry virtualenv with MPS GPU acceleration (3-5x performance improvement)
- **Docker (available):** Reproducible but CPU-only due to macOS virtualization limitations

### Other Platforms
- **Docker (recommended):** Reproducible, hardware-aware with GPU fallback to CPU if unavailable
- **Native (alternative):** Poetry virtualenv for development or maximum GPU utilization

## Configuration

- `config/config.yaml` drives paths, batching, thresholds, devices
- Model weights must be set explicitly:
  - Repo models: e.g., `models/yolov8n-face-lindevs.pt`
  - Local absolute paths: e.g., `/Users/tom/Models/yolov8n-face-lindevs.pt`

## Performance Assumptions

- Stream and chunk to maintain stable memory usage
- Batch inference for throughput; adjust per hardware  
- Control threading to avoid oversubscription; prefer simple, portable defaults

## Performance Optimization (Updated)

### Current Baseline
- **Throughput:** 9.8 images/second (8 workers, CPU-only Docker on M2 Max)
- **Hardware Utilization:** 65-70% CPU, 3GB memory (11% of 28GB Docker allocation)

### Optimization Opportunities  
- **Native + MPS:** 30-50 images/second target (3-5x improvement via GPU acceleration)
- **Worker Scaling:** Test 10-12 workers with Apple Silicon's 12 CPU cores
- **Dependency Optimization:** Replace OpenCV with Pillow+NumPy for lighter processing
- **Algorithmic Improvements:** Batch classification, vectorized operations, optimized index mapping

## Observability

- Progress bars; structured logs; per-worker timings; run-level metadata
- Resource utilization capture and plots in the final report

## Security & Hygiene

- `.env` and secrets not committed; assistant will supply templates when needed
- `.gitignore` includes env, caches, editors; allows `outputs/` and `data/` during rebuild
