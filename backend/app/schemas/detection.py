from enum import Enum

from pydantic import BaseModel, Field

from app.core.config import settings


class Verdict(str, Enum):
    LIKELY_AI = "likely_ai"
    LIKELY_REAL = "likely_real"
    UNCERTAIN = "uncertain"


def score_to_verdict(score: float) -> Verdict:
    if score > settings.verdict_ai_threshold:
        return Verdict.LIKELY_AI
    if score < settings.verdict_real_threshold:
        return Verdict.LIKELY_REAL
    return Verdict.UNCERTAIN


class DetectionResponse(BaseModel):
    id: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    model_used: str
    inference_time_ms: float


class ErrorResponse(BaseModel):
    detail: str
    code: str
