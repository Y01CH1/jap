import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def test_image():
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def client_with_mock_detector():
    from app.main import app
    from app.services.detector import DetectionResult

    mock_detector = MagicMock()
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = DetectionResult(score=0.85, model_name="mock")
    mock_detector.model_name = "mock"

    app.state.detector = mock_detector
    app.state.rate_limiter = MagicMock()
    app.state.rate_limiter.is_allowed.return_value = True

    return TestClient(app)


class TestDetectionAPI:
    def test_detect_returns_result(self, client_with_mock_detector, test_image):
        response = client_with_mock_detector.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 0.85
        assert data["verdict"] == "likely_ai"
        assert data["model_used"] == "mock"

    def test_detect_no_file_returns_422(self, client_with_mock_detector):
        response = client_with_mock_detector.post("/api/v1/detect")
        assert response.status_code == 422

    def test_detect_file_too_large_returns_413(self, client_with_mock_detector):
        big_data = b"x" * (17 * 1024 * 1024)
        response = client_with_mock_detector.post(
            "/api/v1/detect",
            files={"file": ("big.jpg", io.BytesIO(big_data), "image/jpeg")},
        )
        assert response.status_code == 413

    def test_detect_rate_limited_returns_429(self, client_with_mock_detector, test_image):
        client_with_mock_detector.app.state.rate_limiter.is_allowed.return_value = False
        response = client_with_mock_detector.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
        )
        assert response.status_code == 429

    def test_health_returns_ok(self, client_with_mock_detector):
        response = client_with_mock_detector.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
