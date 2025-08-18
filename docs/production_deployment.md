# Production Deployment Guide

## Hardware
- Apple Silicon (M2 Max), 32 GB unified memory recommended
- Fast SSD for input parquet reading

## Environment
- Poetry-managed virtualenv, Python 3.12
- MPS enabled (PyTorch with mps backend)

## Configuration
- Use `config/production.yaml` template and adjust worker count and thresholds per host.

## Monitoring & Logging
- Structured JSONL logs written to `outputs/<run>/logs/production.jsonl`
- Resource utilization stored in `resource_utilization.json` + PNG plot

## Troubleshooting
- Reduce `hardware.max_workers` if peak memory approaches swap
- Disable rampup overrides if startup spike smoothing not desired
- Set `FORCE_CPU_ONLY=1` to diagnose GPU fallback paths


