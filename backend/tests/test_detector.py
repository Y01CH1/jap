import pytest
from PIL import Image

from app.services.detector import BaseDetector, DetectionResult


class FakeDetector(BaseDetector):
    def load(self) -> None:
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

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name must not be empty"):
            DetectionResult(score=0.5, model_name="")


class TestBaseDetector:
    def test_fake_detector_returns_expected_result(self):
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
