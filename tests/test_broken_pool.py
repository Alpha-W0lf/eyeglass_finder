"""A broken worker pool must end the run cleanly with every image counted."""
import json
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd
from loguru import logger

from scripts import process_data
from src.utils.config import load_config
from src.utils.metrics import MetricsManager


class BrokenAfterFirstSubmit:
    """The first chunk's worker dies; every later submit finds the pool broken, as after an OOM kill."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def submit(self, fn, *args):
        self.calls += 1
        if self.calls > 1:
            raise BrokenProcessPool("A child process terminated abruptly")
        future = Future()
        future.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_broken_pool_counts_every_image_as_failed(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"image_url": [f"u{i}" for i in range(6)], "image": [{"bytes": b"x"}] * 6}).to_parquet(
        data / "train-0.parquet", index=False
    )
    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.paths.input_dir, config.paths.output_dir = str(data), str(tmp_path / "out")
    config.data_processing.chunk_size = 2
    config.performance.rampup_enabled = False
    config.performance.prefetch_chunks = 1
    monkeypatch.setattr(process_data, "ProcessPoolExecutor", BrokenAfterFirstSubmit)

    process_data.process_images(config, logger, MetricsManager())

    aggregated = json.loads((tmp_path / "out" / "run_metadata.json").read_text())["aggregated_metrics"]
    assert aggregated["total_images_processed"] == 6
    assert aggregated["images_failed"] == 6
    assert aggregated["failed_chunks"] == 3  # one lost future plus two chunks never submitted


class ScriptedExecutor:
    """Runs no workers: each submit returns a finished future whose outcome comes from `outcome(n, chunk)`."""

    outcome = None

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def submit(self, fn, chunk):
        future = Future()
        try:
            future.set_result(type(self).outcome(self.calls, chunk))
        except Exception as error:
            future.set_exception(error)
        self.calls += 1
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _no_faces(chunk):
    diagnostics = [{"image_id": i, "reason": "no_faces_detected"} for i in chunk.index]
    return len(chunk), 0, [], diagnostics, [], 0.0, 0.0


def _run(tmp_path, monkeypatch, outcome):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"image_url": [f"u{i}" for i in range(6)], "image": [{"bytes": b"x"}] * 6}).to_parquet(
        data / "train-0.parquet", index=False
    )
    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.paths.input_dir, config.paths.output_dir = str(data), str(tmp_path / "out")
    config.data_processing.chunk_size = 2
    config.performance.rampup_enabled = False
    config.performance.prefetch_chunks = 1
    monkeypatch.setattr(ScriptedExecutor, "outcome", staticmethod(outcome))
    monkeypatch.setattr(process_data, "ProcessPoolExecutor", ScriptedExecutor)
    process_data.process_images(config, logger, MetricsManager())
    return json.loads((tmp_path / "out" / "run_metadata.json").read_text())["aggregated_metrics"]


def test_failed_mini_batch_images_are_counted_in_the_run(tmp_path, monkeypatch):
    def outcome(n, chunk):
        if n == 0:
            diagnostics = [{"image_id": i, "reason": "mini_batch_processing_failed"} for i in chunk.index]
            return len(chunk), 0, [], diagnostics, [], 0.0, 0.0
        return _no_faces(chunk)

    aggregated = _run(tmp_path, monkeypatch, outcome)
    assert aggregated["images_failed"] == 2
    assert aggregated["images_with_no_faces"] == 4


def test_lost_worker_chunk_is_counted_in_the_run(tmp_path, monkeypatch):
    def outcome(n, chunk):
        if n == 1:
            raise RuntimeError("worker process died")
        return _no_faces(chunk)

    aggregated = _run(tmp_path, monkeypatch, outcome)
    assert aggregated["images_failed"] == 2
    assert aggregated["images_with_no_faces"] == 4
