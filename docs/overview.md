# Eyeglass Finder – Project Overview

## Objectives

- Build a two-stage pipeline to identify faces and classify eyewear presence.
- Produce a clean, searchable dataset and a per-run report with diagnostics.
- Ensure easy reproducibility (Docker available) and performance-optimized native execution (Poetry recommended for Apple Silicon).

## Scope (MVP)

- Face detection: YOLOv8-Face via ultralytics
- Eyewear classification: glasses-detector (vendored library)
- Input: Parquet with image URLs/bytes; robust streaming and chunking
- Outputs per run: intermediate parquet, final filtered parquet, report.md, plots

## Constraints

- Portable, reproducible environment; minimal host setup
- Include historical `data/` and `outputs/` initially; clean later after fresh run
- No references to prior assessment or company branding
- Keep code modular, readable, and well-documented

## Architecture (high level)

- Stage 1 (process): stream parquet → detect faces + landmarks → crop → classify → write intermediate
- Stage 2 (report): load intermediate → filter by rules → write final dataset → generate report/plots

## Execution Paths

### Apple Silicon (M1/M2/M3/M4) - PERFORMANCE OPTIMIZED
- **Native (Poetry, Python 3.12+):** Recommended for maximum performance with 3-5x GPU acceleration via MPS
- **Docker Compose:** Available for reproducibility, but CPU-only processing due to macOS virtualization limitations

### Other Platforms  
- **Docker Compose:** Recommended for portability and reproducibility
- **Native (Poetry, Python 3.12+):** Alternative for development convenience or NVIDIA GPU access

## Configuration

- Single source: `config/config.yaml` controls paths, batching, thresholds, and device use
- Model weights paths must point to either repo-copied models or valid local absolute paths

## Quality & Observability

- Structured logging, progress bars, timing metrics, and resource utilization
- Verification gates at key milestones (build, tiny dry run, report generation)

## Performance Optimization

For comprehensive optimization analysis and implementation strategy, see **[optimization_notes.md](./optimization_notes.md)**.

Key optimization opportunities:
- Native execution on Apple Silicon for 3-5x MPS GPU acceleration  
- Worker scaling from 8 to 10-12 workers on M2 Max hardware
- Dependency optimization (OpenCV → Pillow+NumPy)
- Algorithmic improvements (batch processing, vectorized operations)
