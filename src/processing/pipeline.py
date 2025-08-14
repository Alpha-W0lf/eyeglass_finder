"""Defines the core, low-level processing functions for the pipeline.

This module provides the fundamental building blocks for the data processing
worker. The functions here are responsible for the heavy lifting of the
computer vision tasks:

1.  `detect_and_crop_faces`: Takes a batch of raw image bytes, runs face
    detection, filters results based on size, and prepares cropped face
    images for classification. It returns a rich dictionary for each
    valid face found.

2.  `classify_faces`: Takes the cropped faces from the detection stage
    and runs them through the eyeglass and sunglass classifiers to
    determine the final `is_target` status.

These functions are designed to be called from within the parallel worker
processes.
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict, Tuple, Callable
import os
from loguru import logger

from src.data_processing.utils import safe_image_open
from src.utils.config import AppConfig


def batch_generator(data_list: list, batch_size: int):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i : i + batch_size]


def detect_and_crop_faces(
    image_bytes_batch: List[bytes],
    image_metadatas: List[Dict],
    face_detector,
    config: AppConfig,
    lock,
) -> Tuple[List[Dict], Dict, int, List[float], List[Dict], List[Dict]]:
    """
    Detects, filters, and crops faces from a batch of images.
    """
    logger.info(
        f"Worker {os.getpid()}: Starting face detection and cropping for {len(image_bytes_batch)} images."
    )

    logger.info(
        f"PIPELINE_TRACKING: Worker {os.getpid()}: PIPELINE START - Processing {len(image_bytes_batch)} images"
    )

    detection_config = config.model_params.face_detection
    inference_batch_size = config.execution.inference_batch_size
    min_face_size = detection_config.min_face_size
    target_size = tuple(detection_config.target_size)

    metrics = {
        "decoding_errors": 0,
        "images_with_no_faces": 0,
        "faces_above_size_threshold": 0,
        "corrupted_batches": 0,
        "corrupted_batch_images": 0,
    }

    all_faces: List[Dict] = []
    total_raw_detections = 0
    all_confidence_scores: List[float] = []

    faces_per_image_stats: List[Dict] = []
    high_face_count_images: List[Dict] = []

    image_batches = batch_generator(image_bytes_batch, inference_batch_size)
    total_batches = (len(image_bytes_batch) + inference_batch_size - 1) // inference_batch_size

    total_images_tracked_in_diagnostics = 0

    for batch_idx, image_batch_bytes in enumerate(image_batches):
        logger.debug(
            f"Worker {os.getpid()}: Processing batch {batch_idx + 1}/{total_batches}"
        )

        batch_size = len(image_batch_bytes)
        logger.debug(
            f"PIPELINE_TRACKING: Worker {os.getpid()}: Batch {batch_idx + 1} contains {batch_size} images"
        )

        image_batch_pil: List[Image.Image] = []
        original_image_modes: List[str] = []
        original_image_dims: List[Tuple[int, int]] = []
        valid_indices_in_batch: List[int] = []
        diagnostic_entries_in_batch = 0

        logger.debug(
            f"Worker {os.getpid()}: Decoding {len(image_batch_bytes)} images for batch {batch_idx + 1}."
        )
        for i, image_bytes in enumerate(image_batch_bytes):
            original_full_batch_index = (batch_idx * inference_batch_size) + i
            image_url = image_metadatas[original_full_batch_index].get("image_url", "N/A")
            logger.debug(
                f"Worker {os.getpid()}: Attempting to process image URL: {image_url}"
            )
            try:
                image, mode = safe_image_open(image_bytes)
                if image.mode != "RGB":
                    logger.debug(
                        f"Worker {os.getpid()}: Converting image from {image.mode} to RGB. URL: {image_url}"
                    )
                    image = image.convert("RGB")

                image_batch_pil.append(image)
                original_image_modes.append(mode)
                original_image_dims.append(image.size)
                valid_indices_in_batch.append(i)
            except Exception as e:
                logger.warning(
                    f"IMAGE_DECODE_FAILURE: Worker {os.getpid()}: FAILED to decode image with URL: {image_url}"
                )
                logger.warning(f"IMAGE_DECODE_FAILURE: Error type: {type(e).__name__}")
                logger.warning(f"IMAGE_DECODE_FAILURE: Error message: {str(e)}")
                logger.warning(
                    f"IMAGE_DECODE_FAILURE: Image bytes length: {len(image_bytes)} bytes"
                )

                if len(image_bytes) == 0:
                    logger.warning(f"IMAGE_DECODE_FAILURE: Empty image bytes detected")
                elif len(image_bytes) < 100:
                    logger.warning(
                        f"IMAGE_DECODE_FAILURE: Suspiciously small image file"
                    )

                metrics["decoding_errors"] = metrics.get("decoding_errors", 0) + 1

                original_idx = (batch_idx * inference_batch_size) + i
                faces_per_image_stats.append(
                    {
                        "image_url": image_url,
                        "num_faces": 0,
                        "failure_reason": "image_decode_failed",
                        "source_file": image_metadatas[original_idx]["source_file"],
                    }
                )
                diagnostic_entries_in_batch += 1
                logger.debug(
                    f"PIPELINE_TRACKING: Worker {os.getpid()}: Added diagnostic entry for decode failure: {image_url}"
                )
        logger.debug(
            f"Worker {os.getpid()}: Successfully decoded {len(image_batch_pil)} images for batch {batch_idx + 1}."
        )

        if not image_batch_pil:
            logger.warning(
                f"CORRUPTED_BATCH: Worker {os.getpid()}: All images in batch {batch_idx + 1} were corrupted. Skipping."
            )
            logger.warning(
                f"CORRUPTED_BATCH: Original batch size was {len(image_batch_bytes)} images"
            )
            logger.warning(
                f"CORRUPTED_BATCH: All {len(image_batch_bytes)} images failed to decode"
            )

            metrics["corrupted_batches"] = metrics.get("corrupted_batches", 0) + 1
            metrics["corrupted_batch_images"] = metrics.get("corrupted_batch_images", 0) + len(
                image_batch_bytes
            )

            for i in range(len(image_batch_bytes)):
                original_idx = (batch_idx * inference_batch_size) + i
                url = image_metadatas[original_idx].get("image_url", "N/A")
                faces_per_image_stats.append(
                    {
                        "image_url": url,
                        "num_faces": 0,
                        "failure_reason": "batch_corrupted",
                    }
                )
                diagnostic_entries_in_batch += 1
                logger.debug(
                    f"PIPELINE_TRACKING: Worker {os.getpid()}: Added diagnostic entry for corrupted batch: {url}"
                )

            total_images_tracked_in_diagnostics += diagnostic_entries_in_batch
            logger.debug(
                f"PIPELINE_TRACKING: Worker {os.getpid()}: Batch {batch_idx + 1} CORRUPTED - added {diagnostic_entries_in_batch} diagnostic entries"
            )
            continue

        MAX_RETRIES = 2
        batch_success = False

        for retry_attempt in range(MAX_RETRIES + 1):
            try:
                batch_boxes, batch_scores, batch_landmarks = face_detector.detect(
                    image_batch_pil
                )
                batch_success = True
                break

            except Exception as e:
                logger.error(
                    f"BATCH_FAILURE: Worker {os.getpid()}: face_detector.detect() failed for batch {batch_idx + 1}, attempt {retry_attempt + 1}/{MAX_RETRIES + 1}"
                )
                logger.error(f"BATCH_FAILURE: Error type: {type(e).__name__}")
                logger.error(f"BATCH_FAILURE: Error message: {str(e)}")
                logger.error(
                    f"BATCH_FAILURE: Batch contained {len(image_batch_pil)} images"
                )

                sample_urls = []
                for i in range(min(3, len(image_batch_pil))):
                    original_idx = (batch_idx * inference_batch_size) + valid_indices_in_batch[i]
                    url = image_metadatas[original_idx].get("image_url", "N/A")
                    sample_urls.append(url)
                logger.error(f"BATCH_FAILURE: Sample URLs: {sample_urls}")

                if retry_attempt == MAX_RETRIES:
                    logger.error(
                        f"FINAL_BATCH_FAILURE: Worker {os.getpid()}: Batch {batch_idx + 1} failed after {MAX_RETRIES} retries. Marking all {len(image_batch_pil)} images as no-faces."
                    )
                    metrics["images_with_no_faces"] += len(image_batch_pil)
                    metrics["failed_inference_batches"] = metrics.get(
                        "failed_inference_batches", 0
                    ) + 1
                    break
                else:
                    logger.warning(
                        f"BATCH_RETRY: Worker {os.getpid()}: Retrying batch {batch_idx + 1} (attempt {retry_attempt + 2}/{MAX_RETRIES + 1})"
                    )
                    import time

                    time.sleep(0.1)

        if not batch_success:
            for i in range(len(image_batch_pil)):
                original_idx = (batch_idx * inference_batch_size) + valid_indices_in_batch[i]
                url = image_metadatas[original_idx].get("image_url", "N/A")
                faces_per_image_stats.append(
                    {
                        "image_url": url,
                        "num_faces": 0,
                        "failure_reason": "batch_inference_failed",
                    }
                )
                diagnostic_entries_in_batch += 1
                logger.debug(
                    f"PIPELINE_TRACKING: Worker {os.getpid()}: Added diagnostic entry for failed batch: {url}"
                )

            total_images_tracked_in_diagnostics += diagnostic_entries_in_batch
            logger.debug(
                f"PIPELINE_TRACKING: Worker {os.getpid()}: Batch {batch_idx + 1} FAILED - added {diagnostic_entries_in_batch} diagnostic entries"
            )
            continue

        for idx, (boxes, scores, landmarks) in enumerate(
            zip(batch_boxes, batch_scores, batch_landmarks)
        ):
            original_full_batch_index = (
                batch_idx * inference_batch_size
            ) + valid_indices_in_batch[idx]

            pil_image = image_batch_pil[idx]
            image_np = np.array(pil_image)
            image_mode = original_image_modes[idx]
            image_width, image_height = original_image_dims[idx]
            url = image_metadatas[original_full_batch_index].get("image_url", "N/A")

            logger.debug(
                f"Worker {os.getpid()}: Processing detection results for {len(boxes) if boxes is not None else 0} faces in URL: {url}."
            )

            if boxes is not None:
                total_raw_detections += len(boxes)
                all_confidence_scores.extend(scores.tolist())

            num_faces_detected = len(boxes) if boxes is not None else 0
            image_confidence_scores = scores.tolist() if boxes is not None else []

            faces_per_image_stats.append(
                {
                    "image_url": url,
                    "num_faces": num_faces_detected,
                    "confidence_scores": image_confidence_scores,
                    "source_file": image_metadatas[original_full_batch_index]["source_file"],
                }
            )
            diagnostic_entries_in_batch += 1
            logger.debug(
                f"PIPELINE_TRACKING: Worker {os.getpid()}: Added diagnostic entry for successful detection: {url} ({num_faces_detected} faces)"
            )

            HIGH_FACE_COUNT_THRESHOLD = 5
            if num_faces_detected > HIGH_FACE_COUNT_THRESHOLD:
                original_image_bytes = image_bytes_batch[original_full_batch_index]
                high_face_count_images.append(
                    {
                        "image_url": url,
                        "num_faces": num_faces_detected,
                        "confidence_scores": image_confidence_scores,
                        "source_file": image_metadatas[original_full_batch_index]["source_file"],
                        "image_bytes": original_image_bytes,
                    }
                )
                logger.info(
                    f"Worker {os.getpid()}: HIGH FACE COUNT detected - {num_faces_detected} faces in image: {url}"
                )

            if boxes is None or boxes.shape[0] == 0:
                metrics["images_with_no_faces"] += 1
                continue

            valid_indices: List[int] = []
            face_dims: List[Tuple[float, float]] = []
            for face_idx, box in enumerate(boxes):
                face_width = box[2] - box[0]
                face_height = box[3] - box[1]
                if face_width >= min_face_size and face_height >= min_face_size:
                    valid_indices.append(face_idx)
                    face_dims.append((face_width, face_height))

            metrics["faces_above_size_threshold"] += len(valid_indices)

            if not valid_indices:
                continue

            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            landmarks = landmarks[valid_indices]

            for box, score, landmark_points, (face_w, face_h) in zip(
                boxes, scores, landmarks, face_dims
            ):
                if len(landmark_points) < 2:
                    continue

                x1, y1, x2, y2 = [int(c) for c in box]
                cropped_face = image_np[y1:y2, x1:x2]

                if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                    resized_face = cv2.resize(cropped_face, target_size)
                    resized_face_bgr = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                    _, jpeg_buffer = cv2.imencode(".jpg", resized_face_bgr)
                    cropped_face_jpeg = jpeg_buffer.tobytes()

                    all_faces.append(
                        {
                            "cropped_face": resized_face,
                            "cropped_face_jpeg": cropped_face_jpeg,
                            "face_bbox": [int(c) for c in box],
                            "face_score": float(score),
                            "face_size": (face_w, face_h),
                            "original_image_size": (image_width, image_height),
                            "original_image_mode": image_mode,
                            "image_url": url,
                            "source_file": image_metadatas[original_full_batch_index]["source_file"],
                        }
                    )

        total_images_tracked_in_diagnostics += diagnostic_entries_in_batch
        logger.debug(
            f"PIPELINE_TRACKING: Worker {os.getpid()}: Batch {batch_idx + 1} SUCCESS - added {diagnostic_entries_in_batch} diagnostic entries"
        )

    if not all_faces:
        logger.info("Worker %s: No faces found in this batch of images.", os.getpid())

    logger.info(
        f"Worker {os.getpid()}: Face detection complete. Processed {len(image_bytes_batch)} images, found {len(all_faces)} valid faces."
    )

    logger.info(
        f"PIPELINE_TRACKING: Worker {os.getpid()}: PIPELINE END - Input: {len(image_bytes_batch)} images, Diagnostic entries: {total_images_tracked_in_diagnostics}"
    )
    if total_images_tracked_in_diagnostics != len(image_bytes_batch):
        logger.error(
            f"PIPELINE_TRACKING: Worker {os.getpid()}: DATA INTEGRITY ERROR - Missing {len(image_bytes_batch) - total_images_tracked_in_diagnostics} images from diagnostics!"
        )
        logger.error(
            f"PIPELINE_TRACKING: Worker {os.getpid()}: Expected: {len(image_bytes_batch)}, Got: {total_images_tracked_in_diagnostics}"
        )
    else:
        logger.info(
            f"PIPELINE_TRACKING: Worker {os.getpid()}: DATA INTEGRITY OK - All images tracked in diagnostics"
        )

    return (
        all_faces,
        metrics,
        total_raw_detections,
        all_confidence_scores,
        faces_per_image_stats,
        high_face_count_images,
    )


def classify_faces(
    cropped_faces: List[Dict],
    eyeglasses_classifier: Callable,
    sunglasses_classifier: Callable,
    config: AppConfig,
) -> Tuple[List[Dict], Dict]:
    """
    Classifies faces and returns the enriched results along with metrics.
    """
    metrics = {
        "faces_classified": len(cropped_faces),
        "faces_with_eyeglasses": 0,
        "faces_rejected_as_sunglasses": 0,
        "faces_passing_confidence_thresholds": 0,
    }
    results: List[Dict] = []

    if not cropped_faces:
        logger.warning(
            f"Worker {os.getpid()}: classify_faces called with no faces to process."
        )
        return results, metrics

    face_batch_np = np.array([face_info["cropped_face"] for face_info in cropped_faces])
    eyeglasses_preds = eyeglasses_classifier.predict(face_batch_np)

    class_config = config.model_params.classification
    present_label = class_config.present_label

    eyeglasses_candidates: List[Dict] = []
    for i, face_info in enumerate(cropped_faces):
        if eyeglasses_preds[i] == present_label:
            eyeglasses_candidates.append(face_info)

    metrics["faces_with_eyeglasses"] = len(eyeglasses_candidates)

    if not eyeglasses_candidates:
        return [], metrics

    candidate_batch_np = np.array([face["cropped_face"] for face in eyeglasses_candidates])
    sunglasses_preds = sunglasses_classifier.predict(candidate_batch_np)

    for i, candidate_info in enumerate(eyeglasses_candidates):
        sunglass_label = sunglasses_preds[i]

        if sunglass_label == present_label:
            metrics["faces_rejected_as_sunglasses"] += 1
            final_result = {
                "eyeglasses_prediction": True,
                "sunglasses_prediction": True,
                "is_target": False,
            }
        else:
            final_result = {
                "eyeglasses_prediction": True,
                "sunglasses_prediction": False,
                "is_target": True,
            }

        combined_result = candidate_info.copy()
        combined_result.update(final_result)
        results.append(combined_result)

    metrics["faces_passing_confidence_thresholds"] = len(
        [r for r in results if r["is_target"]]
    )
    logger.debug(
        f"Worker {os.getpid()}: Classification finished. Found {metrics['faces_passing_confidence_thresholds']} final target faces."
    )
    return results, metrics
