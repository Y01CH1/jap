import pytest
from app.schemas.detection import DetectionResponse, Verdict, score_to_verdict


class TestDetectionResponse:
    def test_verdict_likely_ai(self):
        resp = DetectionResponse(
            id="req_001",
            score=0.92,
            verdict=Verdict.LIKELY_AI,
            model_used="default",
            inference_time_ms=310,
        )
        assert resp.verdict == Verdict.LIKELY_AI

    def test_verdict_likely_real(self):
        resp = DetectionResponse(
            id="req_002",
            score=0.12,
            verdict=Verdict.LIKELY_REAL,
            model_used="default",
            inference_time_ms=280,
        )
        assert resp.verdict == Verdict.LIKELY_REAL

    def test_verdict_uncertain(self):
        resp = DetectionResponse(
            id="req_003",
            score=0.55,
            verdict=Verdict.UNCERTAIN,
            model_used="default",
            inference_time_ms=290,
        )
        assert resp.verdict == Verdict.UNCERTAIN

    def test_score_from_detection_returns_correct_verdict(self):
        assert score_to_verdict(0.95) == Verdict.LIKELY_AI
        assert score_to_verdict(0.71) == Verdict.LIKELY_AI
        assert score_to_verdict(0.70) == Verdict.UNCERTAIN
        assert score_to_verdict(0.55) == Verdict.UNCERTAIN
        assert score_to_verdict(0.30) == Verdict.UNCERTAIN
        assert score_to_verdict(0.29) == Verdict.LIKELY_REAL
        assert score_to_verdict(0.0) == Verdict.LIKELY_REAL
