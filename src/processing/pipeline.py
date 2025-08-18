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
import gc
import io
import os
import time
import torch
from typing import Dict, List, Tuple

from PIL import Image
from loguru import logger
import pandas as pd
from torchvision import transforms
import psutil

from src.modeling.face_detector import FaceDetector
from glasses_detector import GlassesClassifier
from src.processing.batch_classifier import BatchFaceClassifier
from src.utils.config import AppConfig, config_from_dict
from src.utils.device import get_best_available_device
from src.utils.logging_setup import get_logger
from src.utils.metrics import log_memory_usage

# Globals for worker processes
g_config: AppConfig = None
g_face_detector: FaceDetector = None
g_glasses_classifier: GlassesClassifier = None
g_batch_classifier: BatchFaceClassifier | None = None
g_device: torch.device = None


def initialize_pipeline_worker(config_dict: Dict):
    """Initializes the pipeline worker with models and config."""
    global g_config, g_face_detector, g_glasses_classifier, g_device
    
    # This function is called once per worker process.
    # It initializes models and other resources.
    g_config = config_from_dict(config_dict)
    
    log_memory_usage(f"Worker {os.getpid()}: Initializing...")
    
    g_device = get_best_available_device(g_config)
    logger.info(f"Worker {os.getpid()} selected device: {g_device}")

    # Load models on the selected device with a minimal functional check.
    def _load_models_on(device_any):
        global g_face_detector, g_glasses_classifier, g_batch_classifier
        g_face_detector = FaceDetector(
            detection_config=g_config.model_params.face_detection,
            device=device_any,
        )
        log_memory_usage(f"Worker {os.getpid()}: After loading FaceDetector model.")

        # Choose classifier kind based on config (default eyeglasses)
        try:
            clf_kind = getattr(g_config.model_params.classification, "kind", "eyeglasses")
        except Exception:
            clf_kind = "eyeglasses"
        g_glasses_classifier = GlassesClassifier(kind=str(clf_kind), device=device_any)
        g_glasses_classifier.model.eval()
        log_memory_usage(f"Worker {os.getpid()}: After loading GlassesClassifier model.")
        # Initialize batch wrapper
        g_batch_classifier = BatchFaceClassifier(g_glasses_classifier, device_any)

    def _quick_device_check():
        try:
            # Create a tiny blank image and run a quick detection to surface device issues early
            test_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
            _ = g_face_detector.detect(image_batch=[test_img])
            return True
        except Exception:
            logger.warning(
                f"Worker {os.getpid()} device check failed on {g_device}. Falling back to CPU.",
                exc_info=True,
            )
            return False

    # Attempt load and validate on selected device
    _load_models_on(g_device)
    if not _quick_device_check():
        g_device = "cpu"
        _load_models_on(g_device)
        logger.info(f"Worker {os.getpid()} switched to fallback device: {g_device}")

    # Log active eyewear probability threshold for clarity
    try:
        thr = float(getattr(g_config.model_params.classification, "eyewear_prob_threshold", 0.5))
        logger.info(f"Worker {os.getpid()} classifier prob threshold: {thr}")
    except Exception:
        pass


def process_images(image_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict], float, float]:
    """
    Processes a DataFrame of images to detect faces and classify eyewear.
    This version is optimized for memory efficiency by processing images in mini-batches
    and loading image data just-in-time.

    Args:
        image_df: DataFrame with image data, must include 'image_bytes' and an index (image_id).

    Returns:
        A tuple containing two lists:
        - A list of dictionaries, where each dictionary represents a detected face with eyewear classification.
        - A list of dictionaries for diagnostic information.
    """
    logger.info(f"Worker {os.getpid()}: process_images received {len(image_df)} images.")
    global g_config, g_face_detector, g_glasses_classifier, g_batch_classifier, g_device
    
    valid_faces = []
    diagnostics = []
    high_face_images = []
    total_detection_time_s = 0.0
    total_classification_time_s = 0.0
    
    detection_config = g_config.model_params.face_detection
    sampling_cfg = getattr(g_config, "sampling", None)
    batch_size = g_config.performance.inference_batch_size
    total_images = len(image_df)
    total_faces_detected = 0
    images_with_faces = 0
    
    if total_images == 0:
        return [], []

    log_memory_usage(f"Worker {os.getpid()}: Starting processing of {total_images} images.")
    
    # Mini-batch processing loop
    for i in range(0, total_images, batch_size):
        mini_batch_df = image_df.iloc[i:i + batch_size]
        
        # JIT Loading: Convert image bytes to PIL images only for the current mini-batch
        import warnings as _warnings
        mini_batch_images = []
        mini_batch_bytes = []
        for _, row in mini_batch_df.iterrows():
            try:
                # Extract raw bytes from supported representations
                raw_field = row.image
                raw_bytes = None
                if isinstance(raw_field, dict):
                    raw_bytes = raw_field.get('bytes')
                elif isinstance(raw_field, (bytes, bytearray, memoryview)):
                    raw_bytes = bytes(raw_field)
                if not raw_bytes:
                    raise ValueError("Unsupported image field format")

                # Open robustly and normalize modes while suppressing benign PIL warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", UserWarning)
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                # Handle palette images with transparency more explicitly to avoid warnings
                if getattr(pil_img, "mode", None) == "P" and isinstance(getattr(pil_img, "info", {}), dict) and "transparency" in pil_img.info:
                    pil_img = pil_img.convert("RGBA").convert("RGB")
                else:
                    pil_img = pil_img.convert("RGB")
                mini_batch_images.append(pil_img)
                mini_batch_bytes.append(raw_bytes)
            except Exception:
                # Log the error and add a placeholder for index alignment
                logger.warning(f"Could not decode image {row.name}. Skipping.")
                diagnostics.append({"image_id": row.name, "reason": "image_decoding_error"})
                mini_batch_images.append(None)  # Keep list size consistent
                mini_batch_bytes.append(None)

        mini_batch_tuples = [(row.name, mini_batch_bytes[idx]) for idx, (_, row) in enumerate(mini_batch_df.iterrows())]
        
        log_memory_usage(f"Worker {os.getpid()}: Loaded mini-batch {i//batch_size + 1}/{total_images//batch_size + 1}")

        try:
            # Detect faces in the current mini-batch
            valid_images_in_batch = [img for img in mini_batch_images if img is not None]
            if not valid_images_in_batch:
                continue # Skip this mini-batch if all images failed to decode

            # Face detection timing
            _detect_start = time.perf_counter()
            all_boxes, all_scores, _ = g_face_detector.detect(image_batch=valid_images_in_batch)
            total_detection_time_s += (time.perf_counter() - _detect_start)
            log_memory_usage(f"Worker {os.getpid()}: After face detection in mini-batch {i//batch_size + 1}")

            

            # Process detections for this mini-batch
            valid_image_idx = 0
            for j, original_image in enumerate(mini_batch_images):
                if original_image is None:
                    continue # Skip placeholders for failed images
                
                original_image_id, _ = mini_batch_tuples[j]
                row = mini_batch_df.iloc[j]
                
                # Retrieve the results for the current valid image
                boxes = all_boxes[valid_image_idx]
                scores = all_scores[valid_image_idx]
                valid_image_idx += 1
                
                if boxes is None or len(boxes) == 0:
                    diagnostics.append({"image_id": original_image_id, "reason": "no_faces_detected"})
                    continue

                num_faces = len(boxes)
                total_faces_detected += num_faces
                images_with_faces += 1
                # Record raw detection count per image for accurate distribution downstream
                try:
                    diagnostics.append({"image_id": original_image_id, "reason": "faces_detected", "count": int(num_faces)})
                except Exception:
                    pass
                # Capture high face count images for diagnostics (threshold >= 6)
                try:
                    if num_faces >= 6:
                        high_face_images.append({
                            "image_id": int(original_image_id),
                            "image_bytes": row.image['bytes'],
                            "num_faces": int(num_faces),
                            "image_url": row.get("image_url", None),
                        })
                except Exception:
                    # Best-effort only; do not impact main flow
                    pass
  
                _class_start = time.perf_counter()
                # Accumulate faces for batch classification
                face_records = []  # tuples: (face_idx, bbox, conf, crop_for_artifacts, clf_input_image, face_size)
                for face_idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                    conf = scores[face_idx]
                    face_width, face_height = xmax - xmin, ymax - ymin
                    if face_width < detection_config.min_face_size or face_height < detection_config.min_face_size:
                        diagnostics.append({"image_id": original_image_id, "reason": "face_too_small"})
                        continue
                    try:
                        # Build centered square crop with margin for artifacts (config-gated)
                        if sampling_cfg and getattr(sampling_cfg, "square_crop", False):
                            cx = (xmin + xmax) / 2.0
                            cy = (ymin + ymax) / 2.0
                            side = max(face_width, face_height) * (1.0 + float(getattr(sampling_cfg, "crop_margin", 0.0)))
                            half = side / 2.0
                            x0 = int(round(cx - half)); y0 = int(round(cy - half)); x1 = int(round(cx + half)); y1 = int(round(cy + half))
                            W, H = original_image.size
                            if getattr(sampling_cfg, "pad_mode", None):
                                from PIL import ImageOps
                                pad_left = max(0, -x0); pad_top = max(0, -y0); pad_right = max(0, x1 - W); pad_bottom = max(0, y1 - H)
                                if pad_left or pad_top or pad_right or pad_bottom:
                                    fill_mode = sampling_cfg.pad_mode
                                    if fill_mode == "edge":
                                        padded = ImageOps.expand(original_image, border=(pad_left, pad_top, pad_right, pad_bottom))
                                    else:
                                        padded = ImageOps.expand(original_image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
                                    x0 += pad_left; y0 += pad_top; x1 += pad_left; y1 += pad_top
                                    crop_for_artifacts = padded.crop((x0, y0, x1, y1))
                                else:
                                    crop_for_artifacts = original_image.crop((max(0,x0), max(0,y0), min(W,x1), min(H,y1)))
                            else:
                                x0c = max(0, x0); y0c = max(0, y0); x1c = min(W, x1); y1c = min(H, y1)
                                crop_for_artifacts = original_image.crop((x0c, y0c, x1c, y1c))
    else:
                            crop_for_artifacts = original_image.crop((xmin, ymin, xmax, ymax))

                        # Choose classifier input image depending on flag
                        clf_input_image = crop_for_artifacts if (sampling_cfg and getattr(sampling_cfg, "apply_to_classifier", False)) else original_image.crop((xmin, ymin, xmax, ymax))
                        face_records.append((face_idx, (xmin, ymin, xmax, ymax), conf, crop_for_artifacts, clf_input_image, (face_width, face_height)))
                    except Exception:
                        diagnostics.append({"image_id": original_image_id, "reason": "face_processing_error"})

                # Run batch classification on accumulated faces for this image
                if face_records:
                    try:
                        clf_imgs = [rec[4] for rec in face_records]
                        # Dynamic batch sizing heuristic based on available system memory
                        base_bs = int(getattr(g_config.performance, 'face_classification_batch_size', 16) or 16)
                        try:
                            avail_gb = psutil.virtual_memory().available / (1024**3)
                        except Exception:
                            avail_gb = 8.0
                        if avail_gb < 4.0:
                            batch_bs = max(4, base_bs // 4)
                        elif avail_gb < 8.0:
                            batch_bs = max(8, base_bs // 2)
        else:
                            batch_bs = base_bs
                        logger.debug(f"Worker {os.getpid()} classification batch size: {batch_bs} (avail_gb={avail_gb:.2f})")
                        log_memory_usage(f"Worker {os.getpid()}: Before batch classify ({len(clf_imgs)} faces, bs={batch_bs})")
                        eyewear_probas = g_batch_classifier.classify_batch(clf_imgs, batch_size=batch_bs)
                        log_memory_usage(f"Worker {os.getpid()}: After batch classify")
                    except Exception:
                        # Fallback to per-face classification if batch path fails
                        eyewear_probas = []
                        for rec in face_records:
                            eyewear_probas.append(float(g_glasses_classifier.predict(rec[4], format="proba")))

                    # Build outputs
                    for rec, eyewear_proba in zip(face_records, eyewear_probas):
                        face_idx, (xmin, ymin, xmax, ymax), conf, crop_for_artifacts, _clf_img, (face_width, face_height) = rec
                        try:
                            # Resize artifact crop to target size for consistency
                            try:
                                target_w, target_h = detection_config.target_size[0], detection_config.target_size[1]
                                resized_for_artifact = crop_for_artifacts.resize((int(target_w), int(target_h)))
                            except Exception:
                                resized_for_artifact = crop_for_artifacts
                            # Serialize to JPEG bytes
                            jpeg_buf = io.BytesIO()
                            resized_for_artifact.save(jpeg_buf, format="JPEG")
                            cropped_jpeg_bytes = jpeg_buf.getvalue()
                            jpeg_buf.close()
                            # Thresholding
                            try:
                                prob_threshold = float(getattr(g_config.model_params.classification, "eyewear_prob_threshold", 0.5))
                            except Exception:
                                prob_threshold = 0.5
                            is_target = bool(float(eyewear_proba) >= prob_threshold)
                            valid_faces.append({
                                "image_id": original_image_id,
                                "image_url": row.get("image_url", None),
                                "source_file": row.get("source_file", None),
                                "original_image_mode": getattr(original_image, "mode", None),
                                "face_index": face_idx,
                                "face_bbox": [int(c) for c in [xmin, ymin, xmax, ymax]],
                                "face_size": [int(face_width), int(face_height)],
                                "face_score": float(conf.item()) if hasattr(conf, "item") else float(conf),
                                "has_eyewear_confidence": float(eyewear_proba),
                                "glasses_confidence": float(eyewear_proba),
                "sunglasses_prediction": False,
                                "is_target": is_target,
                                "cropped_face_jpeg": cropped_jpeg_bytes,
                            })
                        except Exception:
                            diagnostics.append({"image_id": original_image_id, "reason": "face_processing_error"})
                total_classification_time_s += (time.perf_counter() - _class_start)
          
        except Exception as e:
            # Broad exception to catch issues within the mini-batch loop (e.g., image decoding)
            # Log the error and continue to the next mini-batch if possible
            logger.error(f"Failed to process mini-batch starting at index {i}. Error: {e}", exc_info=True)
            # Add diagnostics for all images in the failed mini-batch
            for _, row in mini_batch_df.iterrows():
                diagnostics.append({"image_id": row.name, "reason": "mini_batch_processing_failed"})
        finally:
            # CRITICAL: Cleanup for the mini-batch to control memory
            if 'mini_batch_df' in locals():
                del mini_batch_df
            if 'mini_batch_images' in locals():
                del mini_batch_images
            if 'mini_batch_tuples' in locals():
                del mini_batch_tuples
            if 'all_boxes' in locals():
                del all_boxes
            if 'all_scores' in locals():
                del all_scores
            
            # Handle both string ("mps") and torch.device cases safely
            device_type = g_device if isinstance(g_device, str) else getattr(g_device, "type", None)
            if device_type == "mps":
                torch.mps.empty_cache()
                
            gc.collect()
            log_memory_usage(f"Worker {os.getpid()}: End of mini-batch {i//batch_size + 1} after cleanup")

    logger.info(f"Worker {os.getpid()}: Finished processing. Detected {total_faces_detected} faces in {images_with_faces}/{total_images} images.")
    return valid_faces, diagnostics, high_face_images, total_detection_time_s, total_classification_time_s
