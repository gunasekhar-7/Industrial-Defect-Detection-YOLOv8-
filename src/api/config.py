"""API configuration settings and operational constraints."""

from pathlib import Path
from typing import List, Set
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


settings = Settings()
