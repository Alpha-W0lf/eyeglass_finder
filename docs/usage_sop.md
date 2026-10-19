# Usage SOP

## Docker (portable; CPU-only on macOS)

Build and run:

```bash
GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "dev") docker compose build
GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "dev") docker compose run --rm app
```

Generate artifacts (most recent run by default):

```bash
docker compose run --rm app python scripts/generate_run_artifacts.py
```

## Native (Poetry; fastest on Apple Silicon)

```bash
poetry install
poetry run python scripts/process_data.py --config config/production.yaml
poetry run python scripts/generate_run_artifacts.py --config config/production.yaml
```

## Config notes

- Config sections: `paths`, `data_processing`, `model_params.face_detection`, `model_params.classification`, `hardware`, `performance`, `logging`, `report_generation`, `observability`, `sampling`.
- `model_params.face_detection.model_path` is `models/yolov8n-face-lindevs.pt`. The file is downloaded on first use and checked against a pinned SHA-256 (`src/modeling/weights.py`); `python scripts/precache_models.py --config config/production.yaml` fetches it ahead of time.
- Adjust `data_processing.chunk_size`, `hardware.max_workers` and `performance.inference_batch_size` per hardware.
