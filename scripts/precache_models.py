"""Download the pinned face-detector weights named in a config (run once at setup or image build)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.weights import ensure_weights
from src.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/production.yaml")
    args = parser.parse_args()
    path = ensure_weights(load_config(args.config).model_params.face_detection.model_path)
    print(f"Weights ready: {path}")
