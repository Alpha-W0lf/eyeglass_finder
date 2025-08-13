# Information Gathering

## Tooling and Dependencies

- Python: 3.12+
- Dependency management: Poetry
- Containerization: Docker, Docker Compose (with build arg for git hash)
- Core libs: torch, torchvision, ultralytics, pandas, pyarrow, pandera, opencv-python, Pillow, numpy, tqdm, loguru, seaborn/matplotlib, psutil

## Environments

- Docker-first: reproducible, hardware-aware fallback to CPU if GPU unavailable
- Native (optional): Poetry virtualenv; leverage Apple Silicon GPU (MPS) when applicable

## Configuration

- `config/config.yaml` drives paths, batching, thresholds, devices
- Model weights must be set explicitly:
  - Repo models: e.g., `models/yolov8n-face-lindevs.pt`
  - Local absolute paths: e.g., `/Users/tom/Models/yolov8n-face-lindevs.pt`

## Performance Assumptions

- Stream and chunk to maintain stable memory usage
- Batch inference for throughput; adjust per hardware
- Control threading to avoid oversubscription; prefer simple, portable defaults

## Observability

- Progress bars; structured logs; per-worker timings; run-level metadata
- Resource utilization capture and plots in the final report

## Security & Hygiene

- `.env` and secrets not committed; assistant will supply templates when needed
- `.gitignore` includes env, caches, editors; allows `outputs/` and `data/` during rebuild
