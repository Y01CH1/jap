from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.detector import BaseDetector, DetectionResult


class FakeDetector(BaseDetector):
    def load(self):
        self._loaded = True

    def detect(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(score=0.85, model_name="fake")


class TestDetectionResult:
    def test_creation(self):
        result = DetectionResult(score=0.75, model_name="test_model")
        assert result.score == 0.75
        assert result.model_name == "test_model"

    def test_score_clamped(self):
        result = DetectionResult(score=1.5, model_name="test")
        assert 0.0 <= result.score <= 1.0


class TestBaseDetector:
    def test_subclass_must_implement_detect(self):
        detector = FakeDetector()
        img = Image.new("RGB", (64, 64))
        result = detector.detect(img)
        assert isinstance(result, DetectionResult)
        assert result.score == 0.85

    def test_load_sets_loaded_flag(self):
        detector = FakeDetector()
        assert not detector.is_loaded
        detector.load()
        assert detector.is_loaded

    def test_preprocess_resizes_image(self):
        detector = FakeDetector()
        img = Image.new("RGB", (800, 600))
        processed = detector.preprocess(img, size=(224, 224))
        assert processed.size == (224, 224)
