from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers.detection import router as detection_router
from app.services.models.base_impl import PyTorchDetector
from app.services.rate_limiter import InMemoryRateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = PyTorchDetector(model_name=settings.model_name, device=settings.device)
    detector.load()
    app.state.detector = detector

    app.state.rate_limiter = InMemoryRateLimiter(
        max_requests=settings.rate_limit_per_minute, window_seconds=60
    )

    yield

    app.state.rate_limiter.cleanup()


app = FastAPI(title="JAP - Judge AI Pictures", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)
