"""API configuration settings and operational constraints."""

from typing import Dict, List, Set, Tuple

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Industrial Defect Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model paths
    MODEL_PATH: str = "models/industrial_yolov8n.onnx"
    FALLBACK_PYTORCH_PATH: str = "models/industrial_yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45

    # Classes
    CLASS_NAMES: List[str] = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled_in_scale",
        "scratches",
    ]

    # Single source of truth for the defect palette (BGR, OpenCV order).
    # Both the API's annotated endpoint and the dashboard render from this.
    CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 255),     # crazing (red)
        1: (0, 165, 255),   # inclusion (orange)
        2: (0, 255, 255),   # patches (yellow)
        3: (255, 0, 255),   # pitted_surface (magenta)
        4: (255, 255, 0),   # rolled_in_scale (cyan)
        5: (0, 255, 0),     # scratches (green)
    }

    # Security Boundaries & Validation
    MAX_IMAGE_SIZE_MB: int = 15
    MAX_IMAGE_DIMENSION: int = 4096
    MIN_IMAGE_DIMENSION: int = 32
    ALLOWED_CONTENT_TYPES: Set[str] = {
        "image/jpeg",
        "image/png",
        "image/bmp",
    }
    RATE_LIMIT_DEFAULT: str = "60/minute"

    # CORS: explicit origins only (override via env ALLOWED_ORIGINS='["..."]').
    # A wildcard is never allowed and the API uses no cookies/auth, so
    # credentials are not granted.
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit dashboard
        "http://localhost:3000",  # local frontends
    ]


settings = Settings()
