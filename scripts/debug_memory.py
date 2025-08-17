import os
import sys
from pathlib import Path

import psutil
import torch

# Ensure src is on the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.modeling.model_loader import load_face_detector, load_glasses_classifiers
from src.utils.device import get_best_available_device

def get_memory_usage_gb() -> float:
    """Returns the current memory usage of the process in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def main():
    """
    Measures the memory footprint of loading the ML models used in the pipeline.
    """
    print("--- Memory Diagnostic Script ---")

    # --- Baseline Memory ---
    mem_before = get_memory_usage_gb()
    print(f"Initial memory usage: {mem_before:.3f} GB")

    # --- Load Config ---
    config_path = "config/config.yaml"
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    mem_after_config = get_memory_usage_gb()
    print(f"Memory after loading config: {mem_after_config:.3f} GB")
    print(f"  -> Config loading cost: {mem_after_config - mem_before:.3f} GB")

    # --- Determine Device ---
    device = get_best_available_device()
    print(f"\nTarget device for models: '{device}'")

    # --- Load Face Detector ---
    print("\nLoading Face Detector model (YOLOv8)...")
    mem_before_fd = get_memory_usage_gb()
    face_detector = load_face_detector(config, device=device)
    torch.cuda.empty_cache() if device == 'cuda' else None # Clear any caching
    mem_after_fd = get_memory_usage_gb()
    print(f"Memory after loading Face Detector: {mem_after_fd:.3f} GB")
    print(f"  -> Face Detector memory cost: {mem_after_fd - mem_before_fd:.3f} GB")


    # --- Load Classifiers ---
    print("\nLoading Eyeglass/Sunglass Classifier models...")
    mem_before_cls = get_memory_usage_gb()
    classifiers = load_glasses_classifiers(device=device)
    torch.cuda.empty_cache() if device == 'cuda' else None # Clear any caching
    mem_after_cls = get_memory_usage_gb()
    print(f"Memory after loading Classifiers: {mem_after_cls:.3f} GB")
    print(f"  -> Classifiers memory cost: {mem_after_cls - mem_before_cls:.3f} GB")

    # --- Final Report ---
    total_model_cost = (mem_after_fd - mem_before_fd) + (mem_after_cls - mem_before_cls)
    print("\n--- Summary ---")
    print(f"Total memory cost for one set of models: {total_model_cost:.3f} GB")

    num_workers = config.execution.num_workers
    projected_total = total_model_cost * (num_workers + 1) # +1 for the main process
    print(f"\nProjected memory for {num_workers} workers (+1 main process):")
    print(f"  -> {total_model_cost:.3f} GB/worker * {num_workers + 1} processes = {projected_total:.3f} GB")
    
    print("\nNote: This is a conservative estimate and doesn't include data processing overhead.")
    print("--- End of Report ---")


if __name__ == "__main__":
    main()
