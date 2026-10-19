"""Pinned download of third-party model weights.

Weights are not committed. Each known file is fetched from its upstream
release on first use and checked against a pinned SHA-256.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# file name -> (upstream URL, SHA-256). Source and licence: CREDITS.md, "Third-party models".
MODEL_WEIGHTS = {
    "yolov8n-face-lindevs.pt": (
        "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt",
        "b038ca653b503453a94f6e12d76feca6840b2a97d7a1322b4498c5e922f29832",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path: str | Path) -> Path:
    """Return `path`, downloading the pinned file first if it is missing. Raise if the bytes don't match."""
    path = Path(path)
    spec = MODEL_WEIGHTS.get(path.name)
    if path.exists():
        if spec and _sha256(path) != spec[1]:
            raise ValueError(f"{path} does not match the pinned SHA-256 for {path.name}; delete it to re-download.")
        return path
    if spec is None:
        raise FileNotFoundError(f"{path} is missing and has no pinned download (known: {sorted(MODEL_WEIGHTS)}).")
    url, expected = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    logger.info(f"Downloading {path.name} from {url}")
    urllib.request.urlretrieve(url, partial)
    actual = _sha256(partial)
    if actual != expected:
        partial.unlink()
        raise ValueError(f"Downloaded {path.name} has SHA-256 {actual}, expected {expected}.")
    partial.replace(path)
    return path
