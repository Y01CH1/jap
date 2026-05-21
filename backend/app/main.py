from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.detection import router as detection_router

app = FastAPI(title="JAP - Judge AI Pictures", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)
