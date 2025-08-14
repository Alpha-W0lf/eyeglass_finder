"""Handles the loading and instantiation of all models for the pipeline.

This module provides centralized functions for loading the pre-trained models
required by the worker processes. This includes:
1.  The YOLOv8-based face detection model.
2.  The eyeglass and sunglass classification models.

These loader functions ensure that models are initialized with the correct
configuration and moved to the appropriate hardware device (e.g., CUDA, MPS, CPU)
for inference.
"""
from typing import Dict
import logging

from src.modeling.face_detector import FaceDetector
from src.utils.config import AppConfig

try:
    from glasses_detector import GlassesClassifier  # type: ignore
except Exception:  # pragma: no cover - optional until vendored
    GlassesClassifier = None  # type: ignore

logger = logging.getLogger(__name__)


def load_face_detector(config: AppConfig, device: str) -> FaceDetector:
    """
    Loads and initializes the face detection model.

    This function accesses `config.model_params.face_detection` to
    instantiate the detector with the correct parameters.
    """
    logger.info(f"Loading face detection model on device '{device}'...")
    face_detection_config = config.model_params.face_detection
    face_detector = FaceDetector(detection_config=face_detection_config, device=device)
    logger.info("Face detection model loaded.")
    return face_detector


def load_glasses_classifiers(device: str) -> dict:
    """
    Initializes and returns the glasses classifier models.

    Returns a dictionary with keys 'eyeglasses' and 'sunglasses'.
    """
    if GlassesClassifier is None:
        raise RuntimeError(
            "glasses_detector is not available yet. Vendor the library before loading classifiers."
        )

    eyeglasses_classifier = GlassesClassifier(kind="eyeglasses", device=device)
    sunglasses_classifier = GlassesClassifier(kind="sunglasses", device=device)

    return {
        "eyeglasses": eyeglasses_classifier,
        "sunglasses": sunglasses_classifier,
    }
