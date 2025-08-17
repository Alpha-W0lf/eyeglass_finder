from __future__ import annotations

import psutil
from dataclasses import dataclass


@dataclass
class MemoryPoolManager:
    """Lightweight memory pressure helper.

    - Detects memory pressure using percent used and available GB guardrails
    - Provides recommendations to throttle prefetch/in-flight tasks when needed
    """

    pressure_percent_threshold: int = 85  # e.g., 85%
    min_available_gb: float = 4.0         # keep at least 4 GB free

    def is_memory_pressure(self) -> bool:
        vm = psutil.virtual_memory()
        if vm.percent >= self.pressure_percent_threshold:
            return True
        avail_gb = vm.available / (1024 ** 3)
        return avail_gb < self.min_available_gb

    def recommend_prefetch_window(self, desired_window: int) -> int:
        if self.is_memory_pressure():
            # Throttle to a minimal in-flight window to relieve pressure
            return 1
        return max(1, desired_window)

    def current_usage_gb(self) -> float:
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / (1024 ** 3)


