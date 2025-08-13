"""A wrapper for the YOLOv8-based face detection model.

This module provides a `FaceDetector` class that encapsulates the logic for
loading the YOLOv8-Face model using the `ultralytics` library and running
inference. It is designed to be thread-safe for use in a multiprocessing
environment.

The class exposes a simple `detect` method that takes a batch of images
and returns all detected bounding boxes, landmarks, and confidence scores.
"""
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    A wrapper class for the YOLOv8-face model from the ultralytics library.
    This class handles model loading, inference, and result parsing.
    """

    def __init__(self, detection_config: dict, device: str):
        """
        Initializes the YOLOv8 face detector.
        Args:
            detection_config (dict): Configuration for face detection. Note that
                the `min_face_size` parameter is handled by a downstream
                filtering step and is not used by this class.
            device (str): The device to load the model onto ('cuda', 'mps', 'cpu').
        """
        self.model_path = detection_config["model_path"]
        self.min_confidence = detection_config["min_confidence"]
        self.keep_all = detection_config["keep_all"]
        try:
            self.model = YOLO(self.model_path)
            self.model.to(device)
        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            raise

        # --- Set inference parameters from the config ---
        self.conf = detection_config.get("min_confidence", 0.3)
        self.iou = detection_config.get("iou_threshold", 0.5)
        self.max_det = detection_config.get("max_detections", 300)

        # The 'imgsz' parameter in ultralytics can take an integer for square
        # resizing or a (h, w) tuple. We'll support both via the config.
        det_size = detection_config.get("det_size", 640)
        if isinstance(det_size, list):
            self.imgsz = tuple(det_size)
        else:
            self.imgsz = det_size

    def detect(
        self, image_batch: List[Image.Image]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Performs face detection on a batch of images.

        Args:
            image_batch (List[Image.Image]): A batch of images in PIL format.

        Returns:
            A tuple containing lists for each image in the batch:
            - boxes_batch (List[np.ndarray]): Bounding boxes in (x1, y1, x2, y2)
              format.
            - scores_batch (List[np.ndarray]): Confidence scores for each detection.
            - landmarks_batch (List[np.ndarray]): Facial landmarks. If the model
              fails to detect landmarks for a face, an empty array of shape
              (num_detections, 5, 2) is returned for that image to maintain
              structural consistency.
        """
        try:
            results = self.model.predict(
                source=image_batch,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error during face detection inference: {e}")
            return [], [], []

        boxes_batch: List[np.ndarray] = []
        scores_batch: List[np.ndarray] = []
        landmarks_batch: List[np.ndarray] = []

        for result in results:
            boxes_batch.append(result.boxes.xyxy.cpu().numpy())
            scores_batch.append(result.boxes.conf.cpu().numpy())

            if result.keypoints is not None:
                landmarks_batch.append(result.keypoints.xy.cpu().numpy())
            else:
                num_detections = len(result.boxes)
                landmarks_batch.append(np.empty((num_detections, 5, 2)))

        return boxes_batch, scores_batch, landmarks_batch

    def predict(self, image_batch: List[np.ndarray]) -> List:
        """
        Runs batch prediction on a list of images.
        """
        results = self.model.predict(
            source=image_batch, verbose=False, conf=self.conf
        )
        return results
