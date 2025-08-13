# Roadmap

## Phase 1: Rebuild (this repo)
- Milestone 0: Planning docs (overview, research, info gathering, architecture, roadmap)
- Milestone A: Scaffolding (license, gitignore, editorconfig, pre-commit, usage SOP)
- Milestone B: Runtime/config (pyproject, Docker, config loader, config.yaml)
- Milestone C–G: Utilities, data layer, modeling, processing, reporting
- Milestone H–J: Entry scripts, historical data/outputs, classifier integration, polish

## Phase 2: Simplification & hardening
- Evaluate removing OpenCV if PIL/NumPy path is sufficient (benchmark)
- Restrict Pandera to artifact-boundary checks or dev-only
- Make SVG conversion optional via config
- Prune unused dependencies (e.g., datasets) and dead code

## Phase 3: Testing & CI
- Smoke tests for loader and tiny pipeline run
- Optional CI workflows for lint and smoke run

## Phase 4: Publication & demo
- Fresh run; regenerate outputs and report
- (Optional) Publish a small demo dataset/run samples
- Update README and dataset card
