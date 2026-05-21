from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class DetectionResult:
    score: float
    model_name: str

    def __post_init__(self):
        clamped = max(0.0, min(1.0, self.score))
        if clamped != self.score:
            object.__setattr__(self, "score", clamped)
        if not self.model_name:
            raise ValueError("model_name must not be empty")


class BaseDetector(ABC):
    def __init__(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        ...

    def preprocess(self, image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
        """Convert image to RGB and resize. Drops alpha/grayscale channels."""
        return image.convert("RGB").resize(size, Image.LANCZOS)
