from __future__ import annotations

from typing import List

from PIL import Image


class BatchFaceClassifier:
    def __init__(self, glasses_classifier, device):
        self.classifier = glasses_classifier
        self.device = device

    def classify_batch(self, face_images: List[Image.Image], batch_size: int = 16) -> List[float]:
        """
        Minimal wrapper to classify faces in small batches.
        Returns a list of eyewear probabilities (floats).
        """
        if not face_images:
            return []
        probs: List[float] = []
        # For now, call into existing classifier one-by-one, preserving API.
        # Future work: move to a vectorized path if the underlying library supports it.
        for img in face_images:
            p = self.classifier.predict(img, format="proba")
            probs.append(float(p))
        return probs


