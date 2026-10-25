"""FastAPI application for Industrial Defect Detection."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.config import settings
from src.api.schemas import (
    BoundingBox,
    DefectItem,
    DetectionResponse,
    HealthResponse,
    LatencyBreakdown,
)
from src.api.security import sanitize_and_read_image
from src.models.benchmark import ONNXDefectDetector
from src.vision.annotate import draw_detections

logger = logging.getLogger("src.api")

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade REST API for real-time industrial surface defect detection using YOLOv8 & ONNX Runtime.",
    lifespan=lifespan,
)
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Registry adapter: Starlette types exception handlers as
    (Request, Exception) while slowapi narrows the second parameter
    (a contravariance mismatch in its annotations). The registry only
    dispatches RateLimitExceeded to this key, so delegate directly."""
    assert isinstance(exc, RateLimitExceeded)
    return _rate_limit_exceeded_handler(request, exc)


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# CORS configuration: explicit origins from settings, never a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
)

# Global model holder
model_engine: Any = None
model_type: str = "none"
_model_lock = threading.Lock()


def load_model() -> None:
    """Initializes the detection model exactly once, thread-safely.

    Favors ONNX Runtime with PyTorch fallback. A failed load leaves the
    engine unset so a later retry (e.g. model exported after startup)
    can still succeed.
    """
    global model_engine, model_type

    with _model_lock:
        if model_engine is not None:
            return  # already loaded — concurrent callers no-op

        onnx_path = Path(settings.MODEL_PATH)
        if onnx_path.exists():
            try:
                model_engine = ONNXDefectDetector(
                    onnx_path,
                    confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                    iou_threshold=settings.IOU_THRESHOLD,
                )
                model_type = "onnx_runtime"
                logger.info("Loaded ONNX model from: %s", onnx_path)
                return
            except Exception:
                logger.exception("Failed to load ONNX model from %s", onnx_path)

        pt_path = Path(settings.FALLBACK_PYTORCH_PATH)
        if pt_path.exists():
            try:
                from ultralytics import YOLO

                model_engine = YOLO(str(pt_path))
                model_type = "pytorch_yolo"
                logger.info("Loaded PyTorch YOLO model from: %s", pt_path)
                return
            except Exception:
                logger.exception("Failed to load PyTorch model from %s", pt_path)

        model_engine = None
        model_type = "not_initialized"
        logger.warning("No model available — set MODEL_PATH or FALLBACK_PYTORCH_PATH")




@app.get("/health", response_model=HealthResponse)
async def health_check() -> Response:
    """Returns service health: 200 when a model is loaded, 503 otherwise.

    503 on missing model lets Docker HEALTHCHECK / orchestrators mark the
    container unhealthy instead of routing traffic to a model-less service.
    """
    payload = HealthResponse(
        status="healthy" if model_engine is not None else "unhealthy",
        app_version=settings.APP_VERSION,
        model_loaded=model_engine is not None,
        model_type=model_type,
        classes=settings.CLASS_NAMES,
    )
    code = status.HTTP_200_OK if model_engine is not None else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=payload.model_dump())


def run_detection(img_bgr: np.ndarray) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Runs inference via loaded model engine."""
    global model_engine, model_type

    if model_engine is None:
        # Load attempt if model was exported after server startup
        load_model()

    if model_engine is None:
        # Model genuinely unavailable: fail loudly instead of inventing results.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )

    if model_type == "onnx_runtime":
        return model_engine.predict(img_bgr)

    # PyTorch fallback
    import time
    t0 = time.perf_counter()
    results = model_engine.predict(img_bgr, conf=settings.CONFIDENCE_THRESHOLD, iou=settings.IOU_THRESHOLD)
    t1 = time.perf_counter()

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            })

    latencies = {
        # ultralytics bundles preprocessing and NMS inside predict(), so the
        # externally-observed cost of those stages is ~0; total is measured.
        "preprocess_ms": 0.0,
        "inference_ms": (t1 - t0) * 1000.0,
        "postprocess_ms": 0.0,
        "total_ms": (t1 - t0) * 1000.0,
    }
    return detections, latencies


@app.post("/v1/detect", response_model=DetectionResponse)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def detect_defects(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    """Analyzes product image and returns defect bounding boxes and confidence scores.

    Deliberately a plain (sync) def: Starlette runs sync handlers in its
    threadpool so inference never blocks the event loop.
    """
    img_bgr, width, height = sanitize_and_read_image(file)
    raw_detections, latencies = run_detection(img_bgr)

    items = []
    for d in raw_detections:
        cls_id = d["class_id"]
        cls_name = settings.CLASS_NAMES[cls_id] if 0 <= cls_id < len(settings.CLASS_NAMES) else f"unknown_{cls_id}"
        items.append(
            DefectItem(
                class_id=cls_id,
                class_name=cls_name,
                confidence=d["confidence"],
                box=BoundingBox(**d["box"]),
            )
        )

    return DetectionResponse(
        filename=file.filename or "unknown",
        image_width=width,
        image_height=height,
        defect_count=len(items),
        defects=items,
        latency=LatencyBreakdown(**latencies),
    )


@app.post("/v1/detect/annotated")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def detect_and_annotate(request: Request, file: UploadFile = File(...)) -> Response:
    """Returns JPEG image with highlighted defect bounding boxes and labels.

    Sync def on purpose: runs in the threadpool, see detect_defects.
    """
    img_bgr, _, _ = sanitize_and_read_image(file)
    raw_detections, _ = run_detection(img_bgr)

    annotated = draw_detections(
        img_bgr, raw_detections, settings.CLASS_NAMES, settings.CLASS_COLORS
    )

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(content=encoded.tobytes(), media_type="image/jpeg")
