# Eyeglass Finder – Project Overview

## Objectives

- Build a two-stage pipeline to identify faces and classify eyewear presence.
- Produce a clean, searchable dataset and a per-run report with diagnostics.
- Ensure easy reproducibility (Docker-first) and optional native execution (Poetry).

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

- Docker Compose (recommended for portability)
- Native (Poetry, Python 3.12+) for GPU on Apple Silicon or local runs

## Configuration

- Single source: `config/config.yaml` controls paths, batching, thresholds, and device use
- Model weights paths must point to either repo-copied models or valid local absolute paths

## Quality & Observability

- Structured logging, progress bars, timing metrics, and resource utilization
- Verification gates at key milestones (build, tiny dry run, report generation)
