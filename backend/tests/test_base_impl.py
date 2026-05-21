from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.services.models.base_impl import PyTorchDetector


class TestPyTorchDetector:
    @patch("app.services.models.base_impl.pipeline")
    def test_load_creates_pipeline(self, mock_pipeline):
        detector = PyTorchDetector(model_name="test/model", device="cpu")
        detector.load()
        mock_pipeline.assert_called_once_with(
            "image-classification", model="test/model", device="cpu"
        )
        assert detector.is_loaded

    @patch("app.services.models.base_impl.pipeline")
    def test_detect_returns_detection_result(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "ai", "score": 0.92}]
        mock_pipeline.return_value = mock_pipe

        detector = PyTorchDetector(model_name="test/model", device="cpu")
        detector.load()
        img = Image.new("RGB", (224, 224))
        result = detector.detect(img)

        assert result.score == 0.92
        assert result.model_name == "test/model"

    @patch("app.services.models.base_impl.pipeline")
    def test_detect_with_human_label_inverts_score(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "human", "score": 0.88}]
        mock_pipeline.return_value = mock_pipe

        detector = PyTorchDetector(model_name="test/model", device="cpu")
        detector.load()
        img = Image.new("RGB", (224, 224))
        result = detector.detect(img)

        assert result.score == 0.12

    @patch("app.services.models.base_impl.pipeline")
    def test_detect_raises_if_not_loaded(self, mock_pipeline):
        detector = PyTorchDetector(model_name="test/model", device="cpu")
        img = Image.new("RGB", (64, 64))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.detect(img)
