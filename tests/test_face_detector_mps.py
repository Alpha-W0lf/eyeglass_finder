"""Apple GPU (MPS) face boxes must match CPU face boxes.

torch 2.3.1 on MPS returned face boxes with a wrong right edge on
letterboxed (non-square) images, which changed which faces passed the
100 px size filter. This test runs only where MPS is available.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDS = REPO_ROOT / "docs/latest_run_showcase/qualitative_analysis/final_targets"

WEIGHTS = REPO_ROOT / load_config(REPO_ROOT / "config" / "production.yaml").model_params.face_detection.model_path

pytestmark = [
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS"),
    # Tests stay offline: fetch the weights once with scripts/precache_models.py.
    pytest.mark.skipif(not WEIGHTS.exists(), reason="face-detector weights not downloaded"),
]


def _portrait_with_face():
    canvas = Image.new("RGB", (300, 451), (128, 128, 128))
    face = sorted(FINDS.glob("*.jpg"))[0]  # any committed face crop works
    canvas.paste(Image.open(face).convert("RGB").resize((150, 150)), (40, 60))
    return canvas


def test_mps_boxes_match_cpu_on_non_square_image():
    from src.modeling.face_detector import FaceDetector

    config = load_config(REPO_ROOT / "config" / "production.yaml").model_params.face_detection
    config = replace(config, model_path=str(REPO_ROOT / config.model_path), min_confidence=0.5)
    image = _portrait_with_face()

    boxes = {}
    for device in ("cpu", "mps"):
        detected, _, _ = FaceDetector(config, device=device).detect([image])
        boxes[device] = detected[0]

    assert len(boxes["cpu"]) >= 1
    assert boxes["mps"].shape == boxes["cpu"].shape
    np.testing.assert_allclose(boxes["mps"], boxes["cpu"], atol=1.0)
