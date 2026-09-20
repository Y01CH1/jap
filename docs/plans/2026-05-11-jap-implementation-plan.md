# JAP (Judge AI Pictures) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Web app where users upload images and get an AI-generation probability score, with FastAPI backend and React frontend.

**Architecture:** Python FastAPI backend serves a single detection endpoint that wraps a HuggingFace pretrained model behind an abstract detector interface. React + Vite frontend calls it via proxy, manages detection state with a custom hook, and renders results with SVG components. No database, no auth — session-only history in browser memory.

**Tech Stack:** Python 3.11+, FastAPI, PyTorch, HuggingFace transformers, React 18, TypeScript, Vite, react-dropzone, axios, CSS modules

> **修订说明（2026-09-20）：** 本计划已对照 `master` 上的实际实现校正。补齐了 venv 创建与 torch/torchvision 配对安装步骤（原 Task 1 Step 4 直接 `pip install` 到全局 Python，是本项目环境损坏的根源）；修正了 Task 3/4 的代码片段与 Task 7 的测试计数；Task 6 的路由替换为实际落地的加固版本，Task 10/11 补上了 object URL 回收逻辑。
>
> **注意：** Task 1 的 `requirements.txt` 写的是**修正后**的版本，仓库当前的 `backend/requirements.txt` 仍是旧的宽松写法（`torch>=2.5.0` / `torchvision>=0.20.0` 分开写）。按本计划重建环境时以计划为准。

---

### Task 1: Backend project scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`

- [ ] **Step 1: Write requirements.txt**

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.19
# torch 与 torchvision 必须版本配对。两者各写各的 >= 会让 pip 自由解析，
# 装出二进制不匹配的组合，报错：operator torchvision::nms does not exist
torch==2.5.0
torchvision==0.20.0
# 锁在 4.x：transformers 5.x 改了 pipeline 的部分签名，本计划代码未针对 v5 验证
transformers>=4.47.0,<5
Pillow>=11.0.0
pydantic>=2.10.0
pydantic-settings>=2.7.0
```

- [ ] **Step 2: Write core/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "prithivMLmodi/AI-or-Not"
    device: str = "cpu"  # 有 GPU 时用环境变量覆盖：JAP_DEVICE=cuda
    max_file_size_mb: int = 16
    rate_limit_per_minute: int = 20
    allowed_extensions: set[str] = {"jpg", "jpeg", "png", "webp"}
    verdict_ai_threshold: float = 0.7
    verdict_real_threshold: float = 0.3

    model_config = {"env_prefix": "JAP_"}


settings = Settings()
```

- [ ] **Step 3: Write minimal main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="JAP - Judge AI Pictures", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
```

- [ ] **Step 4: Create venv, install dependencies & verify**

不要直接 `pip install` 到全局 Python —— backend 必须有独立的 venv，否则 torch 这类重依赖会污染其它项目，且版本冲突时无法隔离排查。

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell；Git Bash 用 source .venv/Scripts/activate
pip install -r requirements.txt
```

装完先单独验证 torch/torchvision 这对二进制包能一起导入，再起服务 —— 这一步失败的话，`transformers` 的导入会跟着炸：

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Then: `uvicorn app.main:app --port 8000`
Expected: server starts, `GET http://localhost:8000/api/v1/health` returns `{"status":"ok","version":"0.1.0"}`

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/
git commit -m "feat: scaffold backend project with FastAPI entry point"
```

---

### Task 2: Pydantic schemas

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/detection.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_schemas.py`

- [ ] **Step 1: Write the test file for schemas**

```python
import pytest
from app.schemas.detection import DetectionResponse, Verdict


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
        from app.schemas.detection import score_to_verdict

        assert score_to_verdict(0.95) == Verdict.LIKELY_AI
        assert score_to_verdict(0.71) == Verdict.LIKELY_AI
        assert score_to_verdict(0.70) == Verdict.UNCERTAIN
        assert score_to_verdict(0.55) == Verdict.UNCERTAIN
        assert score_to_verdict(0.30) == Verdict.UNCERTAIN
        assert score_to_verdict(0.29) == Verdict.LIKELY_REAL
        assert score_to_verdict(0.0) == Verdict.LIKELY_REAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_schemas.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write schemas/detection.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_schemas.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/ backend/tests/
git commit -m "feat: add Pydantic detection schemas with verdict logic"
```

---

### Task 3: Detector abstract interface

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/detector.py`
- Create: `backend/tests/test_detector.py`

- [ ] **Step 1: Write the test for detector interface**

```python
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

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name must not be empty"):
            DetectionResult(score=0.5, model_name="")


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_detector.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write services/detector.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class DetectionResult:
    score: float
    model_name: str

    def __post_init__(self):
        # frozen dataclass 不能直接赋值，需用 object.__setattr__ 绕过
        clamped = max(0.0, min(1.0, self.score))
        if clamped != self.score:
            object.__setattr__(self, "score", clamped)
        if not self.model_name:
            raise ValueError("model_name must not be empty")


class BaseDetector(ABC):
    def __init__(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        ...

    def preprocess(self, image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
        """Convert image to RGB and resize. Drops alpha/grayscale channels."""
        return image.convert("RGB").resize(size, Image.LANCZOS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_detector.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/detector.py backend/tests/test_detector.py
git commit -m "feat: add BaseDetector abstract interface"
```

---

### Task 4: PyTorch detector implementation

**Files:**
- Create: `backend/app/services/models/__init__.py`
- Create: `backend/app/services/models/base_impl.py`
- Create: `backend/tests/test_base_impl.py`

- [ ] **Step 1: Write the test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_base_impl.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write services/models/base_impl.py**

```python
from PIL import Image
from transformers import pipeline

from app.services.detector import BaseDetector, DetectionResult


class PyTorchDetector(BaseDetector):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def load(self) -> None:
        self._pipe = pipeline(
            "image-classification",
            model=self.model_name,
            device=self.device,
        )
        self._loaded = True

    def detect(self, image: Image.Image) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        processed = self.preprocess(image)
        predictions = self._pipe(processed)
        top = predictions[0]
        score = top["score"]

        label = top["label"].lower()
        if "human" in label or "real" in label or "natural" in label:
            score = 1.0 - score

        return DetectionResult(score=score, model_name=self.model_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_base_impl.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/models/ backend/tests/test_base_impl.py
git commit -m "feat: add PyTorchDetector using HuggingFace pipeline"
```

---

### Task 5: Rate limiter

**Files:**
- Create: `backend/app/services/rate_limiter.py`
- Create: `backend/tests/test_rate_limiter.py`

- [ ] **Step 1: Write the test**

```python
import time

import pytest

from app.services.rate_limiter import InMemoryRateLimiter


class TestInMemoryRateLimiter:
    def test_allows_requests_within_limit(self):
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1")

    def test_blocks_when_limit_exceeded(self):
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("10.0.0.1")
        assert limiter.is_allowed("10.0.0.1")
        assert not limiter.is_allowed("10.0.0.1")

    def test_different_ips_have_separate_limits(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("1.1.1.1")
        assert limiter.is_allowed("2.2.2.2")

    def test_window_expires(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("3.3.3.3")
        assert not limiter.is_allowed("3.3.3.3")
        time.sleep(1.1)
        assert limiter.is_allowed("3.3.3.3")

    def test_cleanup_removes_expired_entries(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=0)
        limiter.is_allowed("4.4.4.4")
        limiter.cleanup()
        assert len(limiter._requests) == 0

    def test_get_remaining_returns_correct_count(self):
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("5.5.5.5") == 5
        limiter.is_allowed("5.5.5.5")
        assert limiter.get_remaining("5.5.5.5") == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_rate_limiter.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write services/rate_limiter.py**

```python
import time
from collections import defaultdict


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        return max(0, self.max_requests - len(self._requests[key]))

    def cleanup(self):
        now = time.time()
        window_start = now - self.window_seconds
        for key in list(self._requests.keys()):
            self._requests[key] = [t for t in self._requests[key] if t > window_start]
            if not self._requests[key]:
                del self._requests[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_rate_limiter.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rate_limiter.py backend/tests/test_rate_limiter.py
git commit -m "feat: add in-memory IP rate limiter with sliding window"
```

---

### Task 6: Detection API router

**Files:**
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/detection.py`
- Create: `backend/tests/test_detection_api.py`

- [ ] **Step 1: Write the API test**

```python
import io
from unittest.mock import MagicMock, patch

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_detection_api.py -v`
Expected: FAIL — module not found or import error

- [ ] **Step 3: Write routers/detection.py**

```python
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

    # 先看 Content-Length 快速拒绝；缺失或畸形都不阻断，交给下面的分块读取兜底
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_file_size_mb * 1024 * 1024 + 1024:
                return _error_response(413, "File too large", "file_too_large")
        except ValueError:
            pass

    # 分块读取：没有 Content-Length 的请求也不能让 await file.read() 无界缓冲
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

    # 推理是同步阻塞调用，丢到线程池，避免卡住事件循环
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_detection_api.py -v`
Expected: all 5 tests PASS

> **已知覆盖缺口：** 上面 5 个测试都没覆盖 Step 3 新增的两条路径 —— 畸形 `Content-Length`（走不到 `int()` 的 `ValueError` 分支）和缺 `Content-Length` 时的分块读取限流。这两条目前只能靠人工构造请求验证，回归时不会报警。建议补两个测试用例。

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/ backend/tests/test_detection_api.py
git commit -m "feat: add detection API endpoint with rate limiting"
```

---

### Task 7: Wire up main.py lifecycle

**Files:**
- Modify: `backend/app/main.py` (full rewrite)

- [ ] **Step 1: Rewrite main.py with startup/shutdown**

```python
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
```

- [ ] **Step 2: Verify backend starts**

Run: `cd backend && timeout 5 uvicorn app.main:app --port 8000 2>&1 || true`
Expected: "Application startup complete" in output (model download may take time on first run)

- [ ] **Step 3: Run full backend test suite**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all 25 tests from Tasks 2-6 PASS

- [ ] **Step 4: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: wire up app lifecycle with model loading and rate limiter"
```

---

### Task 8: Frontend project scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.app.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/vite-env.d.ts`

- [ ] **Step 1: Write package.json**

```json
{
  "name": "jap-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "axios": "^1.7.9",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-dropzone": "^14.3.5"
  },
  "devDependencies": {
    "@types/react": "^18.3.18",
    "@types/react-dom": "^18.3.5",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "~5.6.2",
    "vite": "^6.0.0"
  }
}
```

- [ ] **Step 2: Write tsconfig.json**

```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

- [ ] **Step 3: Write tsconfig.app.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Write tsconfig.node.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 5: Write vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 6: Write index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>JAP - Judge AI Pictures</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔍</text></svg>" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Write src/main.tsx**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 8: Write src/vite-env.d.ts**

```typescript
/// <reference types="vite/client" />
```

- [ ] **Step 9: Install dependencies & verify**

Run: `cd frontend && npm install && npm run dev`
Expected: Vite dev server starts on port 5173, blank page loads

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React + Vite + TypeScript frontend"
```

---

### Task 9: API service layer

**Files:**
- Create: `frontend/src/services/api.ts`
- Create: `frontend/src/types.ts`

- [ ] **Step 1: Write types.ts**

```typescript
export type Verdict = 'likely_ai' | 'likely_real' | 'uncertain';

export interface DetectionResponse {
  id: string;
  score: number;
  verdict: Verdict;
  model_used: string;
  inference_time_ms: number;
}

export interface ErrorResponse {
  detail: string;
  code: string;
}

export type DetectionState = 'idle' | 'uploading' | 'detecting' | 'done' | 'error';

export interface HistoryEntry {
  id: string;
  fileName: string;
  thumbnailUrl: string;
  score: number;
  verdict: Verdict;
  timestamp: number;
}
```

- [ ] **Step 2: Write api.ts**

```typescript
import axios, { AxiosProgressEvent } from 'axios';
import { DetectionResponse } from '../types';

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export async function detectImage(
  file: File,
  onProgress?: (percent: number) => void
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await client.post<DetectionResponse>('/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (event: AxiosProgressEvent) => {
      if (event.total && onProgress) {
        onProgress(Math.round((event.loaded * 100) / event.total));
      }
    },
  });

  return data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await client.get('/health');
    return true;
  } catch {
    return false;
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/ frontend/src/types.ts
git commit -m "feat: add API service layer and TypeScript types"
```

---

### Task 10: useDetection hook

**Files:**
- Create: `frontend/src/hooks/useDetection.ts`

- [ ] **Step 1: Write useDetection.ts**

```typescript
import { useCallback, useState, useRef, useEffect } from 'react';
import { DetectionResponse, DetectionState, HistoryEntry } from '../types';
import { detectImage } from '../services/api';

interface UseDetectionReturn {
  state: DetectionState;
  progress: number;
  result: DetectionResponse | null;
  error: string | null;
  history: HistoryEntry[];
  submit: (file: File) => Promise<void>;
  reset: () => void;
}

export function useDetection(): UseDetectionReturn {
  const [state, setState] = useState<DetectionState>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const objectUrls = useRef<string[]>([]);

  const submit = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setState('uploading');
    setProgress(0);

    const thumbnailUrl = URL.createObjectURL(file);
    objectUrls.current.push(thumbnailUrl);

    try {
      const data = await detectImage(file, (pct) => {
        setProgress(pct);
        if (pct >= 100) setState('detecting');
      });
      setResult(data);

      setHistory((prev) => [
        {
          id: data.id,
          fileName: file.name,
          thumbnailUrl,
          score: data.score,
          verdict: data.verdict,
          timestamp: Date.now(),
        },
        ...prev,
      ]);

      setState('done');
    } catch (err: unknown) {
      URL.revokeObjectURL(thumbnailUrl);
      objectUrls.current = objectUrls.current.filter((u) => u !== thumbnailUrl);

      const msg =
        err && typeof err === 'object' && 'message' in err
          ? (err as { message: string }).message
          : 'Unknown error';
      setError(msg);
      setState('error');
    }
  }, []);

  const reset = useCallback(() => {
    setState('idle');
    setProgress(0);
    setResult(null);
    setError(null);
  }, []);

  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  return { state, progress, result, error, history, submit, reset };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/
git commit -m "feat: add useDetection hook for detection state machine"
```

---

### Task 11: UploadZone component

**Files:**
- Create: `frontend/src/components/UploadZone.tsx`
- Create: `frontend/src/components/UploadZone.module.css`

- [ ] **Step 1: Write UploadZone.tsx**

```tsx
import { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { DetectionState } from '../types';
import styles from './UploadZone.module.css';

interface UploadZoneProps {
  state: DetectionState;
  progress: number;
  onUpload: (file: File) => void;
}

export function UploadZone({ state, progress, onUpload }: UploadZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length === 0) return;
      const file = accepted[0];
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      onUpload(file);
    },
    [onUpload]
  );

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
    maxSize: 16 * 1024 * 1024,
    disabled: state === 'uploading' || state === 'detecting',
  });

  const isBusy = state === 'uploading' || state === 'detecting';

  return (
    <div className={styles.wrapper}>
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${isBusy ? styles.busy : ''}`}
      >
        <input {...getInputProps()} />
        {isBusy ? (
          <div className={styles.progress}>
            <div className={styles.spinner} />
            <p>{state === 'uploading' ? `Uploading... ${progress}%` : 'Analyzing...'}</p>
            {state === 'uploading' && (
              <div className={styles.bar}>
                <div className={styles.fill} style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        ) : isDragActive ? (
          <p>Drop image here</p>
        ) : (
          <div>
            <p className={styles.cta}>Drag & drop an image, or click to select</p>
            <p className={styles.hint}>JPG, PNG, WebP up to 16MB</p>
          </div>
        )}
      </div>
      {preview && !isBusy && (
        <div className={styles.preview}>
          <img src={preview} alt="Preview" />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Write UploadZone.module.css**

```css
.wrapper {
  width: 100%;
  max-width: 480px;
  margin: 0 auto;
}

.dropzone {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.dropzone:hover,
.active {
  border-color: #6366f1;
  background: #eef2ff;
}

.busy {
  cursor: default;
  border-color: #6366f1;
  background: #eef2ff;
}

.cta {
  font-size: 16px;
  color: #1e293b;
  margin: 0 0 8px;
}

.hint {
  font-size: 13px;
  color: #94a3b8;
  margin: 0;
}

.progress {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #475569;
  font-size: 14px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e2e8f0;
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.bar {
  width: 100%;
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
}

.fill {
  height: 100%;
  background: #6366f1;
  border-radius: 3px;
  transition: width 0.2s;
}

.preview {
  margin-top: 16px;
  border-radius: 8px;
  overflow: hidden;
  max-height: 300px;
}

.preview img {
  width: 100%;
  height: auto;
  object-fit: contain;
  max-height: 300px;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/UploadZone.tsx frontend/src/components/UploadZone.module.css
git commit -m "feat: add UploadZone component with drag-and-drop"
```

---

### Task 12: ResultCard component

**Files:**
- Create: `frontend/src/components/ResultCard.tsx`
- Create: `frontend/src/components/ResultCard.module.css`

- [ ] **Step 1: Write ResultCard.tsx**

```tsx
import { DetectionResponse } from '../types';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  result: DetectionResponse;
  onRetry: () => void;
}

function RingChart({ score }: { score: number }) {
  const r = 42;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score);

  let color = '#eab308'; // uncertain
  if (score > 0.7) color = '#ef4444';
  else if (score < 0.3) color = '#22c55e';

  return (
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="8" />
      <circle
        cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
        strokeLinecap="round" strokeDasharray={circumference}
        strokeDashoffset={offset} transform="rotate(-90 60 60)"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
      />
      <text x="60" y="56" textAnchor="middle" fontSize="22" fontWeight="700" fill="#1e293b">
        {Math.round(score * 100)}%
      </text>
      <text x="60" y="76" textAnchor="middle" fontSize="11" fill="#64748b">
        AI probability
      </text>
    </svg>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const labels: Record<string, { text: string; className: string }> = {
    likely_ai: { text: 'Likely AI-Generated', className: 'ai' },
    likely_real: { text: 'Likely Real', className: 'real' },
    uncertain: { text: 'Uncertain', className: 'uncertain' },
  };
  const { text, className } = labels[verdict] ?? labels.uncertain;

  return <span className={`${styles.badge} ${styles[className]}`}>{text}</span>;
}

export function ResultCard({ result, onRetry }: ResultCardProps) {
  return (
    <div className={styles.card}>
      <RingChart score={result.score} />
      <VerdictBadge verdict={result.verdict} />
      <p className={styles.meta}>inference: {result.inference_time_ms}ms</p>
      <button className={styles.retry} onClick={onRetry}>
        Test another image
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Write ResultCard.module.css**

```css
.card {
  text-align: center;
  padding: 32px 24px;
  max-width: 360px;
  margin: 0 auto;
}

.badge {
  display: inline-block;
  margin: 16px 0 8px;
  padding: 6px 18px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
}

.ai {
  background: #fef2f2;
  color: #dc2626;
}

.real {
  background: #f0fdf4;
  color: #16a34a;
}

.uncertain {
  background: #fefce8;
  color: #ca8a04;
}

.meta {
  font-size: 13px;
  color: #94a3b8;
  margin: 4px 0 20px;
}

.retry {
  background: none;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 8px 20px;
  font-size: 14px;
  color: #475569;
  cursor: pointer;
  transition: border-color 0.2s;
}

.retry:hover {
  border-color: #6366f1;
  color: #6366f1;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ResultCard.tsx frontend/src/components/ResultCard.module.css
git commit -m "feat: add ResultCard with SVG ring chart and verdict badge"
```

---

### Task 13: HistoryList component

**Files:**
- Create: `frontend/src/components/HistoryList.tsx`
- Create: `frontend/src/components/HistoryList.module.css`

- [ ] **Step 1: Write HistoryList.tsx**

```tsx
import { HistoryEntry } from '../types';
import styles from './HistoryList.module.css';

interface HistoryListProps {
  history: HistoryEntry[];
}

const verdictLabel: Record<string, string> = {
  likely_ai: 'AI',
  likely_real: 'Real',
  uncertain: '?',
};

export function HistoryList({ history }: HistoryListProps) {
  if (history.length === 0) return null;

  return (
    <div className={styles.section}>
      <h3 className={styles.heading}>This session</h3>
      <div className={styles.list}>
        {history.map((entry) => (
          <div key={entry.id} className={styles.item}>
            <img src={entry.thumbnailUrl} alt={entry.fileName} className={styles.thumb} />
            <div className={styles.info}>
              <span className={styles.name}>{entry.fileName}</span>
              <span className={styles.score}>{Math.round(entry.score * 100)}% AI</span>
            </div>
            <span className={`${styles.tag} ${styles[entry.verdict]}`}>
              {verdictLabel[entry.verdict]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Write HistoryList.module.css**

```css
.section {
  max-width: 480px;
  margin: 32px auto 0;
  padding: 0 16px;
}

.heading {
  font-size: 14px;
  font-weight: 600;
  color: #64748b;
  margin: 0 0 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: #f8fafc;
  border-radius: 8px;
}

.thumb {
  width: 40px;
  height: 40px;
  border-radius: 6px;
  object-fit: cover;
}

.info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow: hidden;
}

.name {
  font-size: 13px;
  color: #334155;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.score {
  font-size: 12px;
  color: #94a3b8;
}

.tag {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 10px;
  flex-shrink: 0;
}

.likely_ai {
  background: #fef2f2;
  color: #dc2626;
}

.likely_real {
  background: #f0fdf4;
  color: #16a34a;
}

.uncertain {
  background: #fefce8;
  color: #ca8a04;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/HistoryList.tsx frontend/src/components/HistoryList.module.css
git commit -m "feat: add HistoryList component for session detection log"
```

---

### Task 14: App.tsx assembly

**Files:**
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.module.css`

- [ ] **Step 1: Write App.tsx**

```tsx
import { useDetection } from './hooks/useDetection';
import { UploadZone } from './components/UploadZone';
import { ResultCard } from './components/ResultCard';
import { HistoryList } from './components/HistoryList';
import styles from './App.module.css';

export default function App() {
  const { state, progress, result, error, history, submit, reset } = useDetection();

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.logo}>JAP</h1>
        <p className={styles.subtitle}>Judge AI Pictures — Spot synthetic media, instantly.</p>
      </header>

      <main className={styles.main}>
        {(state === 'idle' || state === 'uploading' || state === 'detecting') && (
          <UploadZone state={state} progress={progress} onUpload={submit} />
        )}

        {state === 'done' && result && (
          <ResultCard result={result} onRetry={reset} />
        )}

        {state === 'error' && (
          <div className={styles.error}>
            <p>{error || 'Something went wrong'}</p>
            <button onClick={reset}>Try again</button>
          </div>
        )}

        <HistoryList history={history} />
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Write App.module.css**

```css
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #ffffff;
  color: #1e293b;
  -webkit-font-smoothing: antialiased;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  text-align: center;
  padding: 40px 16px 0;
}

.logo {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #6366f1;
}

.subtitle {
  font-size: 14px;
  color: #94a3b8;
  margin-top: 4px;
}

.main {
  flex: 1;
  padding: 32px 16px 48px;
}

.error {
  text-align: center;
  max-width: 360px;
  margin: 0 auto;
  padding: 32px 24px;
  color: #dc2626;
}

.error button {
  margin-top: 12px;
  background: none;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 8px 20px;
  font-size: 14px;
  cursor: pointer;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.module.css
git commit -m "feat: assemble App with detection state flow"
```

---

### Task 15: End-to-end verification

先把能自动化的部分跑成一条命令。**任何一步失败都不要进入人工验证** —— 下面的手动步骤没有客观判据，这个任务上一轮就是因此被整体落下的。

- [ ] **Step 1: Automated gate**

```powershell
cd backend
python -m pytest tests/ -q                 # 期望：25 passed
python -c "import app.main; print('import OK')"
cd ..\frontend
npx tsc -b                                 # 期望：无输出
```

Expected: `25 passed`、`import OK`、tsc 无报错。三项全过才继续。

- [ ] **Step 2: Start backend**

Run: `cd backend && uvicorn app.main:app --port 8000`
Wait for model to load (first run downloads from HuggingFace).
Expected: `curl http://localhost:8000/api/v1/health` 返回 `"model_loaded": true`。

- [ ] **Step 3: Start frontend**

Run: `cd frontend && npm run dev`
Expected: Vite starts on port 5173.

- [ ] **Step 4: Manual smoke test in browser**

Open `http://localhost:5173`, upload a test image, verify:
- UploadZone shows drag state and progress bar
- ResultCard displays ring chart, verdict badge, and inference time
- "Test another image" resets to upload state
- HistoryList accumulates entries

- [ ] **Step 5: API smoke test via curl**

```bash
# Create a test image
python -c "from PIL import Image; Image.new('RGB', (256,256), 'blue').save('/tmp/test.jpg')"

# Health check
curl http://localhost:8000/api/v1/health

# Detection
curl -X POST http://localhost:8000/api/v1/detect -F "file=@/tmp/test.jpg"
```

Expected: health returns model info, detect returns valid JSON with score and verdict.

- [ ] **Step 6: Commit if anything changed**

```bash
git status
# If any fixes were made, commit them
```
