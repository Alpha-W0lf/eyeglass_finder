"""Failures must be counted, never read as "no faces"."""
import io

import pandas as pd
import pytest
from PIL import Image

from src.modeling.face_detector import FaceDetector
from src.processing import pipeline, worker


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_empty_chunk_returns_the_five_values_callers_unpack():
    assert pipeline.process_images(pd.DataFrame()) == ([], [], [], 0.0, 0.0)


def test_detector_error_is_raised_not_returned_as_no_faces():
    class BrokenModel:
        def predict(self, **kwargs):
            raise RuntimeError("device error")

    detector = object.__new__(FaceDetector)
    detector.model, detector.conf, detector.iou, detector.max_det, detector.imgsz = BrokenModel(), 0.5, 0.5, 300, 640
    with pytest.raises(RuntimeError):
        detector.detect([Image.new("RGB", (32, 32))])


def test_failed_mini_batch_marks_each_image_failed(monkeypatch):
    from pathlib import Path

    from src.utils.config import load_config

    class BrokenDetector:
        def detect(self, image_batch):
            raise RuntimeError("device error")

    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    monkeypatch.setattr(pipeline, "g_config", config)
    monkeypatch.setattr(pipeline, "g_face_detector", BrokenDetector())
    monkeypatch.setattr(pipeline, "g_device", "cpu")
    df = pd.DataFrame({"image": [{"bytes": _png_bytes()}] * 3, "image_url": ["u"] * 3}, index=[7, 8, 9])

    faces, diagnostics, _, _, _ = pipeline.process_images(df)

    assert faces == []
    assert [(d["image_id"], d["reason"]) for d in diagnostics] == [
        (7, "mini_batch_processing_failed"), (8, "mini_batch_processing_failed"), (9, "mini_batch_processing_failed")
    ]


def test_crashed_worker_chunk_marks_each_image_failed(monkeypatch):
    def crash(image_df):
        raise RuntimeError("worker crash")

    monkeypatch.setattr(worker, "g_worker_config", object())
    monkeypatch.setattr(worker, "process_images", crash)
    df = pd.DataFrame({"image_url": ["u", "v"]}, index=[3, 4])

    num_images, faces, valid, diagnostics, _, _, _ = worker.process_chunk_of_data(df)

    assert (num_images, faces, valid) == (2, 0, [])
    assert diagnostics == [{"image_id": 3, "reason": "chunk_failed"}, {"image_id": 4, "reason": "chunk_failed"}]


def test_edge_padding_repeats_border_pixels_and_constant_fills_black():
    image = Image.new("RGB", (2, 1), (200, 10, 10))
    edge = pipeline.pad_image(image, left=1, top=0, right=0, bottom=0, mode="edge")
    black = pipeline.pad_image(image, left=1, top=0, right=0, bottom=0, mode="constant")
    assert edge.size == black.size == (3, 1)
    assert edge.getpixel((0, 0)) == (200, 10, 10)
    assert black.getpixel((0, 0)) == (0, 0, 0)
