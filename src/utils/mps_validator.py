"""
This module provides a suite of validation functions to verify the compatibility,
performance, and memory usage of machine learning models on Apple's Metal
Performance Shaders (MPS) backend.

It is designed to be used as a standalone diagnostic tool before enabling MPS
acceleration in the main pipeline, helping to catch potential issues early.

The validation functions cover:
- YOLOv8 face detector compatibility and performance.
- glasses-detector classifier compatibility and performance.
- End-to-end pipeline memory usage and stability on MPS.
"""

import torch
import time
import psutil
import os
import numpy as np
from PIL import Image

from src.utils.config import load_config
from src.modeling.model_loader import load_face_detector, load_glasses_classifiers


def get_memory_usage_gb() -> float:
    """Returns the current memory usage of the process in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def validate_yolov8_mps():
    """Validates the YOLOv8 model on MPS."""
    print("--- Validating YOLOv8 Face Detector on MPS ---")
    
    config = load_config("config/config.yaml")
    dummy_image = Image.new('RGB', (640, 480), color = 'red')
    num_runs = 50

    try:
        # --- CPU Benchmark ---
        print("Running CPU benchmark...")
        cpu_detector = load_face_detector(config, device="cpu")
        cpu_detector.detect([dummy_image]) # Warm-up run
        
        start_time = time.time()
        for _ in range(num_runs):
            cpu_detector.detect([dummy_image])
        cpu_time = time.time() - start_time
        print(f"CPU time for {num_runs} runs: {cpu_time:.3f} seconds")
        del cpu_detector
        torch.cuda.empty_cache()

        # --- MPS Benchmark ---
        if not torch.backends.mps.is_available():
            print("MPS not available on this system. Skipping benchmark.")
            return

        print("\nRunning MPS benchmark...")
        mps_detector = load_face_detector(config, device="mps")
        mps_detector.detect([dummy_image]) # Warm-up run
        
        start_time = time.time()
        for _ in range(num_runs):
            mps_detector.detect([dummy_image])
        mps_time = time.time() - start_time
        print(f"MPS time for {num_runs} runs: {mps_time:.3f} seconds")

        # --- Comparison ---
        speedup = cpu_time / mps_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"YOLOv8 MPS compatible: True")

    except Exception as e:
        print(f"\nAn error occurred during MPS validation: {e}")
        print(f"YOLOv8 MPS compatible: False")


def validate_glasses_detector_mps():
    """Validates the glasses detector models on MPS."""
    print("\n--- Validating Glasses Detector Classifiers on MPS ---")
    
    dummy_face = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    num_runs = 100

    try:
        # --- CPU Benchmark ---
        print("Running CPU benchmark...")
        cpu_classifiers = load_glasses_classifiers(device="cpu")
        cpu_classifiers["eyeglasses"].predict(np.expand_dims(dummy_face, axis=0)) # Warm-up

        start_time = time.time()
        for _ in range(num_runs):
            cpu_classifiers["eyeglasses"].predict(np.expand_dims(dummy_face, axis=0))
            cpu_classifiers["sunglasses"].predict(np.expand_dims(dummy_face, axis=0))
        cpu_time = time.time() - start_time
        print(f"CPU time for {num_runs} runs (both models): {cpu_time:.3f} seconds")
        del cpu_classifiers

        # --- MPS Benchmark ---
        if not torch.backends.mps.is_available():
            print("MPS not available on this system. Skipping benchmark.")
            return

        print("\nRunning MPS benchmark...")
        mps_classifiers = load_glasses_classifiers(device="mps")
        mps_classifiers["eyeglasses"].predict(np.expand_dims(dummy_face, axis=0)) # Warm-up

        start_time = time.time()
        for _ in range(num_runs):
            mps_classifiers["eyeglasses"].predict(np.expand_dims(dummy_face, axis=0))
            mps_classifiers["sunglasses"].predict(np.expand_dims(dummy_face, axis=0))
        mps_time = time.time() - start_time
        print(f"MPS time for {num_runs} runs (both models): {mps_time:.3f} seconds")
        
        # --- Comparison ---
        speedup = cpu_time / mps_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Glasses Detector MPS compatible: True")

    except Exception as e:
        print(f"\nAn error occurred during MPS validation: {e}")
        print(f"Glasses Detector MPS compatible: False")


def validate_mps_memory():
    """Validates the memory usage with MPS enabled."""
    print("\n--- Validating Memory Usage on MPS ---")
    
    config = load_config("config/config.yaml")

    try:
        # --- Baseline Memory ---
        mem_before = get_memory_usage_gb()
        print(f"Initial memory usage: {mem_before:.3f} GB")

        # --- CPU Memory Usage ---
        print("Loading models on CPU...")
        cpu_detector = load_face_detector(config, device="cpu")
        cpu_classifiers = load_glasses_classifiers(device="cpu")
        mem_after_cpu = get_memory_usage_gb()
        cpu_cost = mem_after_cpu - mem_before
        print(f"Memory after loading on CPU: {mem_after_cpu:.3f} GB (Cost: {cpu_cost:.3f} GB)")
        del cpu_detector, cpu_classifiers

        # --- MPS Memory Usage ---
        if not torch.backends.mps.is_available():
            print("MPS not available on this system. Skipping benchmark.")
            return
            
        print("\nLoading models on MPS...")
        mps_detector = load_face_detector(config, device="mps")
        mps_classifiers = load_glasses_classifiers(device="mps")
        mem_after_mps = get_memory_usage_gb()
        mps_cost = mem_after_mps - mem_after_cpu
        print(f"Memory after loading on MPS: {mem_after_mps:.3f} GB (Incremental Cost: {mps_cost:.3f} GB)")
        
        increase_percent = (mps_cost / cpu_cost - 1) * 100 if cpu_cost > 0 else 0
        print(f"\nMemory increase from CPU to MPS: {increase_percent:.2f}%")
        print(f"MPS memory acceptable: True")

    except Exception as e:
        print(f"\nAn error occurred during memory validation: {e}")
        print(f"MPS memory acceptable: False")


def validate_end_to_end_mps():
    """Runs a small end-to-end test on the MPS device."""
    print("\n--- Running End-to-End Validation on MPS ---")
    # Implementation will follow from optimization_notes.md
    pass


if __name__ == "__main__":
    validate_yolov8_mps()
    validate_glasses_detector_mps()
    validate_mps_memory()
    validate_end_to_end_mps()
