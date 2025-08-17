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

from src.modeling.face_detector import FaceDetector
from glasses_detector import GlassesClassifier
from src.utils.config import AppConfig, config_from_dict
from src.utils.device import get_best_available_device
from src.utils.logging_setup import get_logger
from src.utils.metrics import log_memory_usage

# Globals for worker processes
g_config: AppConfig = None
g_face_detector: FaceDetector = None
g_glasses_classifier: GlassesClassifier = None
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
    
    g_face_detector = FaceDetector(
        detection_config=g_config.model_params.face_detection,
        device=g_device
    )
    log_memory_usage(f"Worker {os.getpid()}: After loading FaceDetector model.")
    
    g_glasses_classifier = GlassesClassifier(device=g_device)
    # The model inside the wrapper must be set to evaluation mode
    g_glasses_classifier.model.eval()
    log_memory_usage(f"Worker {os.getpid()}: After loading GlassesClassifier model.")


def process_images(image_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
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
    global g_config, g_face_detector, g_glasses_classifier, g_device
    
    valid_faces = []
    diagnostics = []
    
    detection_config = g_config.model_params.face_detection
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
        mini_batch_images = []
        for _, row in mini_batch_df.iterrows():
            try:
                # Attempt to open the image from bytes
                image = Image.open(io.BytesIO(row.image['bytes'])).convert("RGB")
                mini_batch_images.append(image)
            except Exception:
                # Log the error and add a placeholder for index alignment
                logger.warning(f"Could not decode image {row.name}. Skipping.")
                diagnostics.append({"image_id": row.name, "reason": "image_decoding_error"})
                mini_batch_images.append(None) # Keep list size consistent

        mini_batch_tuples = [(row.name, row.image['bytes']) for _, row in mini_batch_df.iterrows()]
        
        log_memory_usage(f"Worker {os.getpid()}: Loaded mini-batch {i//batch_size + 1}/{total_images//batch_size + 1}")

        try:
            # Detect faces in the current mini-batch
            valid_images_in_batch = [img for img in mini_batch_images if img is not None]
            if not valid_images_in_batch:
                continue # Skip this mini-batch if all images failed to decode

            all_boxes, all_scores, _ = g_face_detector.detect(image_batch=valid_images_in_batch)
            log_memory_usage(f"Worker {os.getpid()}: After face detection in mini-batch {i//batch_size + 1}")

            

            # Process detections for this mini-batch
            valid_image_idx = 0
            for j, original_image in enumerate(mini_batch_images):
                if original_image is None:
                    continue # Skip placeholders for failed images
                
                original_image_id, _ = mini_batch_tuples[j]
                
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
  
                for face_idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                    conf = scores[face_idx]
                    face_width, face_height = xmax - xmin, ymax - ymin
  
                    if face_width < detection_config.min_face_size or face_height < detection_config.min_face_size:
                        diagnostics.append({"image_id": original_image_id, "reason": "face_too_small"})
                        continue

                    try:
                        cropped_face = original_image.crop((xmin, ymin, xmax, ymax))
                        # Request probability from the classifier to get a numeric output
                        eyewear_proba = g_glasses_classifier.predict(cropped_face, format="proba")

                        valid_faces.append({
                            "image_id": original_image_id,
                            "face_index": face_idx,
                            "bounding_box": [int(c) for c in [xmin, ymin, xmax, ymax]],
                            "confidence": float(conf.item()) if hasattr(conf, "item") else float(conf),
                            "has_eyewear_confidence": float(eyewear_proba),
                            "glasses_confidence": float(eyewear_proba),
                            "sunglasses_confidence": None,
                        })
                    except Exception as e:
                        logger.error(
                            f"WORKER CRASH POINT: An unexpected error occurred while processing face {face_idx} for image {original_image_id}.",
                            exc_info=True
                        )
                        diagnostics.append({
                            "image_id": original_image_id,
                            "reason": "face_processing_error"
                        })
          
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
    return valid_faces, diagnostics
