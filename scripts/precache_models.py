from pathlib import Path
import shutil

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
    # Adjust this path if your local models live elsewhere
    precache_from_local(Path("/Users/tom/Documents/Git/stability_ai_take_home/models"))
