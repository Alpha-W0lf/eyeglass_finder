"""Simple performance benchmarking utility for pipeline runs.

Provides a minimal API to load the latest run's metrics and validate
optimizations against a configurable regression threshold.

Designed to be lightweight and dependency-free beyond the standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class BenchmarkResult:
    run_id: str
    images_per_second: float
    total_runtime_seconds: float
    total_images_processed: int
    total_detection_time_seconds: float
    total_classification_time_seconds: float


class PerformanceBenchmark:
    def __init__(self, outputs_root: str = "outputs", regression_threshold: float = 0.10):
        self.outputs_root = Path(outputs_root)
        self.regression_threshold = float(regression_threshold)

    def _load_latest_run_dir(self) -> Optional[Path]:
        if not self.outputs_root.exists():
            return None
        run_dirs = sorted(
            [p for p in self.outputs_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return run_dirs[0] if run_dirs else None

    def _load_run_metadata(self, run_dir: Path) -> Dict[str, Any]:
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}

    def _load_report_metrics(self, run_dir: Path) -> Dict[str, Any]:
        # Prefer report markdown table values if present
        report_path = run_dir / "report.md"
        if not report_path.exists():
            return {}
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        def _extract(label: str) -> Optional[float]:
            for line in text.splitlines():
                if label in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        try:
                            return float(parts[2].strip().split()[0])
                        except Exception:
                            return None
            return None
        return {
            "total_runtime_seconds": _extract("Total Runtime (seconds)"),
            "images_per_second": _extract("Images per Second"),
            "total_detection_time_seconds": _extract("Total Detection Time (s)"),
            "total_classification_time_seconds": _extract("Total Classification Time (s)"),
        }

    def load_latest(self) -> Optional[BenchmarkResult]:
        run_dir = self._load_latest_run_dir()
        if run_dir is None:
            return None
        meta = self._load_run_metadata(run_dir)
        rpt = self._load_report_metrics(run_dir)

        run_id = meta.get("run_summary", {}).get("run_id", run_dir.name)
        perf = meta.get("aggregated_metrics", {})
        total_images = perf.get("total_images_processed")

        ips = rpt.get("images_per_second")
        runtime = rpt.get("total_runtime_seconds")
        det_s = rpt.get("total_detection_time_seconds")
        cls_s = rpt.get("total_classification_time_seconds")

        # Fallbacks if report did not parse
        if ips is None and runtime and total_images:
            ips = round(float(total_images) / float(runtime), 2)
        if runtime is None:
            runtime = 0.0
        if det_s is None:
            det_s = 0.0
        if cls_s is None:
            cls_s = 0.0
        if total_images is None:
            total_images = 0

        return BenchmarkResult(
            run_id=run_id,
            images_per_second=float(ips or 0.0),
            total_runtime_seconds=float(runtime or 0.0),
            total_images_processed=int(total_images or 0),
            total_detection_time_seconds=float(det_s or 0.0),
            total_classification_time_seconds=float(cls_s or 0.0),
        )

    def validate_optimization(self, optimization_name: str, baseline_ips: Optional[float] = None) -> Dict[str, Any]:
        latest = self.load_latest()
        if latest is None:
            return {"status": "no_runs"}

        current_ips = latest.images_per_second
        if baseline_ips is None:
            # If no baseline provided, treat current as baseline (pass by default)
            return {
                "status": "baseline_recorded",
                "optimization": optimization_name,
                "baseline_images_per_second": current_ips,
                "run_id": latest.run_id,
            }

        change = (current_ips - baseline_ips) / max(baseline_ips, 1e-9)
        status = "pass" if change >= -self.regression_threshold else "regression"
        return {
            "status": status,
            "optimization": optimization_name,
            "baseline_images_per_second": round(baseline_ips, 2),
            "current_images_per_second": round(current_ips, 2),
            "relative_change": round(change * 100.0, 2),
            "run_id": latest.run_id,
        }


