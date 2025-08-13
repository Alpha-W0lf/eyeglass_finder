from pathlib import Path
import shutil
import os

DEFAULT_MODELS = [
    ("yolov8n-face-lindevs.pt", "models/yolov8n-face-lindevs.pt"),
    ("yolov8s-face.pt", "models/yolov8s-face.pt"),
]


def precache_from_local(local_dir: Path) -> None:
    local_dir = Path(local_dir)
    for src_name, dest_path in DEFAULT_MODELS:
        src = local_dir / src_name
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {src} -> {dest}")
        else:
            print(f"Missing {src}; skipped")


if __name__ == "__main__":
    # Use MODELS_DIR env var if provided; otherwise default to local ./models
    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    precache_from_local(models_dir)
