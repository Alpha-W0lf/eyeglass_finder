# AGENTS.md

Eyeglass Finder is a two-stage computer-vision pipeline. Stage 1 (`scripts/process_data.py`) finds faces with YOLOv8-Face, scores each large face for eyeglasses, and then runs Stage 2. Stage 2 (`scripts/generate_run_artifacts.py`) filters the results and writes the report and galleries; it can be re-run on its own.

## Environment

- Python 3.12, managed by Poetry (torch 2.6.0 / torchvision 0.21.0, pinned as a matched pair).
- Face-detector weights are not committed: `src/modeling/weights.py` downloads them on first use and checks a pinned SHA-256. A new model file needs its URL and SHA-256 added there and a line in `CREDITS.md`.
- Input data: WIT parquet shards in `data/raw/` matching `train-*.parquet`. Data and `outputs/` stay out of git.
- Device order is CUDA, then Apple MPS, then CPU (`src/utils/device.py`). Set `FORCE_CPU_ONLY=true` to force CPU.

## Commands

```bash
poetry install
python -m compileall -q src scripts libs          # the CI gate
poetry run pytest -q                              # unit tests (CPU only, no data, no network)
poetry run ruff check src scripts tests           # lint, as CI runs it
poetry run python scripts/process_data.py --config config/production.yaml    # full run (GPU, minutes)
poetry run python scripts/generate_run_artifacts.py --config config/production.yaml --run_dir outputs/<run_id>/  # Stage 2 only
```

## Done means

- Lint, tests and the compile gate pass locally, and CI is green.
- New or changed logic has a CPU-only pytest test in `tests/` that needs no GPU and no data. A bug fix ships with a test that fails on the old code.
- If a change can move pipeline output or speed, re-run both showcase shards. Compare the data funnel and images/sec with `docs/latest_run_showcase/report.md`, check the finds with `scripts/compare_runs.py`, and paste the results into the PR.
- Every public number (README, docs, GitHub description) points to a committed run report. The headline is ~21× (10.6 → 225 images/sec), from `docs/baseline_run/report.md` and `docs/latest_run_showcase/report.md`; it changes only with a new committed report.

## Conventions

- Config lives in `config/*.yaml` and is read through the typed `AppConfig` (`src/utils/config.py`). Add new settings there.
- Update a doc in place when its facts change. Link to the run report instead of copying numbers into new docs.
- `libs/glasses-detector/` is a vendored MIT fork. Change it only for compatibility fixes.
- Claude-only or personal settings go in `CLAUDE.local.md`, which git ignores.

## Read when

- Changing setup, config keys or output schemas: [`docs/TECHNICAL.md`](docs/TECHNICAL.md)
- Working on throughput or memory: [`docs/post_run_report.md`](docs/post_run_report.md)
- Touching images, galleries or licensing: [`CREDITS.md`](CREDITS.md) and [`NOTICE`](NOTICE)
