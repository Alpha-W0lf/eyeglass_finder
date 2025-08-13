Eyeglass Finder
================

A two-stage pipeline to detect faces and classify eyewear, producing reproducible runs and a concise final dataset with reporting.

Quickstart (Docker)
-------------------

```bash
GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo dev) docker compose build && docker compose run --rm app
```

Native (Poetry)
---------------

```bash
poetry install
poetry run python scripts/process_data.py
poetry run python scripts/generate_run_artifacts.py
```

Configuration
-------------
- Edit `config/config.yaml` to set input paths and model weights. Weights can be under `models/` or absolute local paths.

Artifacts
---------
- Each run writes to `outputs/run_*` with `report.md`, plots, logs, and datasets.


