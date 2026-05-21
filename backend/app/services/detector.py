from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class DetectionResult:
    score: float
    model_name: str

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))


class BaseDetector(ABC):
    def __init__(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        ...

    def preprocess(self, image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
        return image.convert("RGB").resize(size, Image.LANCZOS)
