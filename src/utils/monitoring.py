"""Monitors and records system resource utilization (CPU, Memory, Disk I/O).

This module provides a `ResourceMonitor` class that runs in a background
thread to periodically sample the system's CPU, memory, and disk I/O usage.
It is designed to be non-intrusive and can be started and stopped from a main
process to profile a specific workload.
"""

from __future__ import annotations

import threading
import time
import psutil
from typing import List, Dict, Any
from loguru import logger
import json
from pathlib import Path


class ResourceMonitor:
    """A thread-safe class to monitor system resource usage."""

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.data: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._last_disk_io = psutil.disk_io_counters()

    def _monitor(self):
        logger.info(f"Resource monitor started. Sampling every {self.interval}s.")
        start_time = time.time()
        while not self._stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            memory_used_mb = mem.used / (1024 * 1024)

            current_disk_io = psutil.disk_io_counters()
            read_mbps = (current_disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024) / self.interval
            write_mbps = (current_disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024) / self.interval
            self._last_disk_io = current_disk_io

            self.data.append({
                "timestamp": time.time() - start_time,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_used_mb,
                "disk_read_mbps": read_mbps,
                "disk_write_mbps": write_mbps,
            })
            self._stop_event.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        logger.info("Resource monitor stopped.")

    def save_to_json(self, output_path: Path):
        logger.info(f"Saving resource utilization data to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save resource data to {output_path}: {e}")


