from PIL import Image
from transformers import pipeline

from app.services.detector import BaseDetector, DetectionResult


class PyTorchDetector(BaseDetector):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def load(self) -> None:
        self._pipe = pipeline(
            "image-classification",
            model=self.model_name,
            device=self.device,
        )
        self._loaded = True

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        processed = self.preprocess(image)
        predictions = self._pipe(processed)
        top = predictions[0]
        score = top["score"]

        label = top["label"].lower()
        if "human" in label or "real" in label or "natural" in label:
            score = 1.0 - score

        return DetectionResult(score=score, model_name=self.model_name)
