# Usage SOP

## Docker (recommended)

Build and run:

```bash
GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "dev") docker compose build
GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "dev") docker compose run --rm app
```

Generate artifacts (most recent run by default):

```bash
docker compose run --rm app python scripts/generate_run_artifacts.py
```

## Native (optional, Poetry)

```bash
poetry install
poetry run python scripts/process_data.py
poetry run python scripts/generate_run_artifacts.py
```

## Config notes

- Full config mirrors the original project structure: `paths`, `data_processing`, `model_params.face_detection`, `model_params.classification`, `execution`, `logging`, `report_generation`.
- Set model paths in `model_params.face_detection.model_path` to either:
  - repo-copied weights: `models/yolov8n-face-lindevs.pt`
  - local absolute paths: `/Users/tom/Models/yolov8n-face-lindevs.pt`
- Adjust `data_processing.chunk_size`, `execution.num_workers`, and `execution.inference_batch_size` per hardware.
