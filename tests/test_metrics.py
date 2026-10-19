from src.utils import metrics as metrics_module
from src.utils.metrics import MetricsManager


def test_finalize_computes_images_per_second(monkeypatch):
    manager = MetricsManager(start_time=1000.0)
    manager.total_images_processed = 1000
    monkeypatch.setattr(metrics_module.time, "time", lambda: 1250.0)

    manager.finalize()

    assert manager.total_runtime_seconds == 250.0
    assert manager.images_per_second == 4.0


def test_finalize_with_no_images_reports_zero(monkeypatch):
    manager = MetricsManager(start_time=1000.0)
    monkeypatch.setattr(metrics_module.time, "time", lambda: 1010.0)
    manager.finalize()
    assert manager.images_per_second == 0


def test_stage2_rebuilds_the_run_timings_from_metadata():
    import pandas as pd

    from scripts.generate_run_artifacts import rebuild_metrics

    metadata = {
        "run_summary": {"run_id": "run_x", "run_command": "c", "git_commit_hash": "abc", "environment": {}},
        "config": {},
        "start_time": 0.0,
        "aggregated_metrics": {"total_images_processed": 39258, "final_target_count": 44},
        "worker_performance": {"worker_processing_times": [1.0, 3.0], "avg_chunk_time_seconds": 2.0},
        "performance": {
            "total_runtime_seconds": 178.85,
            "images_per_second": 219.5,
            "total_detection_time_seconds": 489.23,
            "total_classification_time_seconds": 11.55,
        },
    }
    df = pd.DataFrame({"face_score": [0.9, 0.8]})

    metrics = rebuild_metrics(metadata, df)

    assert metrics.images_per_second == 219.5
    assert metrics.total_runtime_seconds == 178.85
    assert metrics.total_detection_time_seconds == 489.23
    assert metrics.final_target_count == 44
    assert metrics.worker_time_avg == 2.0


def test_images_per_second_counts_only_successfully_processed_images(monkeypatch):
    manager = MetricsManager(start_time=1000.0)
    manager.total_images_processed = 100
    manager.images_failed = 40
    monkeypatch.setattr(metrics_module.time, "time", lambda: 1010.0)

    manager.finalize()

    assert manager.images_per_second == 6.0


def test_run_status_is_non_zero_when_any_image_failed():
    from scripts.process_data import run_status

    ok, failed = MetricsManager(), MetricsManager()
    failed.images_failed = 3
    assert run_status(ok) == 0
    assert run_status(failed) == 1
