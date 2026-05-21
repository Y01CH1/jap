import asyncio
import time
import uuid
from io import BytesIO

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

from app.core.config import settings
from app.schemas.detection import DetectionResponse, ErrorResponse, score_to_verdict

router = APIRouter(prefix="/api/v1")


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def detect(request: Request, file: UploadFile = File(...)):
    rate_limiter = request.app.state.rate_limiter
    client_ip = request.client.host

    if not rate_limiter.is_allowed(client_ip):
        return _error_response(429, "Rate limit exceeded. Try again later.", "rate_limited")

    if not file.filename:
        return _error_response(422, "Filename is missing", "missing_filename")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        return _error_response(422, f"Unsupported format: .{ext}", "invalid_format")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_file_size_mb * 1024 * 1024 + 1024:
                return _error_response(413, "File too large", "file_too_large")
        except ValueError:
            pass

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    buf = BytesIO()
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = await file.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        buf.write(chunk)
        remaining -= len(chunk)
    if buf.tell() > max_bytes:
        return _error_response(413, "File too large", "file_too_large")
    contents = buf.getvalue()

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
        image = Image.open(BytesIO(contents))
    except Exception:
        return _error_response(422, "Invalid or corrupted image", "invalid_image")
    detector = request.app.state.detector

    start = time.perf_counter()
    result = await asyncio.to_thread(detector.detect, image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return DetectionResponse(
        id=f"req_{uuid.uuid4().hex[:12]}",
        score=round(result.score, 4),
        verdict=score_to_verdict(result.score),
        model_used=result.model_name,
        inference_time_ms=round(elapsed_ms, 1),
    )


@router.get("/health")
async def health(request: Request):
    detector = request.app.state.detector
    return {
        "status": "ok",
        "model_loaded": detector.is_loaded,
        "model_name": getattr(detector, "model_name", "unknown"),
    }


def _error_response(status_code: int, detail: str, code: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "code": code},
    )
