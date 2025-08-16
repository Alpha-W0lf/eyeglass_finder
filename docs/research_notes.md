# Research Notes

## Detectors

- YOLOv8-Face (ultralytics)
  - Pros: landmarks + simple API, active ecosystem, good portability
  - Cons: model weight management, version pinning
- Alternatives considered
  - RetinaFace: strong accuracy, but extra setup; portability trade-offs
  - YuNet: very fast, but limited accuracy/landmarks vs YOLOv8-Face

Decision: Use YOLOv8-Face for detection and landmarks. Keep batch size/config tunables in `config.yaml`.

## Eyewear Classification

- glasses-detector (vendored fork)
  - Pros: solves domain task quickly; PyTorch-based; simple API
  - Notes: keep LICENSE; path dependency via pyproject; ensure torch/torchvision pins

Open tuning items:
- Evaluate necessity of a secondary “sunglasses” classifier vs single-model approach
- Batch sizing, thread settings (env vars), and device selection

## Datasets / Inputs

- Parquet source (URLs/bytes) streamed in chunks
- Tiny sample set for smoke tests/dry runs is recommended

## Reporting

- Markdown report: executive summary, performance plots, distributions, qualitative samples
- Diagnostics for face counts, size, and confidences

## Tooling & Environment

### Execution Strategy (Updated)
- **Apple Silicon:** Native (Poetry) recommended for 3-5x MPS GPU acceleration
- **Other Platforms:** Docker + Docker Compose (primary), Poetry (alternative)
- Python 3.12+, PyTorch, ultralytics, pandas, pyarrow, pandera, loguru, tqdm, seaborn/matplotlib
- Config-driven; avoid hardcoding paths or thresholds

### Performance Optimization Considerations
- **Dependency lightening:** OpenCV → Pillow+NumPy for reduced overhead
- **Algorithmic efficiency:** Batch processing, vectorized operations, index pre-computation
- **Hardware utilization:** Scale to 10-12 workers on M2 Max, leverage unified memory architecture
- **Memory optimization:** Target <20GB usage while maximizing throughput
