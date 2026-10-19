from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ProductionLogger:
    """
    Minimal structured JSONL logger for production observability.

    Writes one JSON object per line to the provided file. Intended for
    main-process operational events (milestones, metrics snapshots).
    """

    def __init__(self, output_path: Path | str, default_level: str = "INFO") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_level = default_level

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        try:
            return str(value)
        except Exception:
            return "<unserializable>"

    def _write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_event(self, event: str, level: str | None = None, **context: Any) -> None:
        record = {
            "timestamp": self._now_iso(),
            "level": (level or self.default_level).upper(),
            "event": event,
            "context": self._serialize(context),
        }
        self._write(record)

    def log_performance_metric(self, name: str, value: float, unit: str = "", **context: Any) -> None:
        self.log_event(
            event="performance_metric",
            metric_name=name,
            value=value,
            unit=unit,
            **context,
        )

    def log_processing_milestone(self, milestone: str, images_processed: Optional[int] = None, elapsed_seconds: Optional[float] = None, **context: Any) -> None:
        self.log_event(
            event="processing_milestone",
            milestone=milestone,
            images_processed=images_processed,
            elapsed_seconds=elapsed_seconds,
            **context,
        )


