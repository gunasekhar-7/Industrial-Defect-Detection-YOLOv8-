"""Pydantic schemas for request/response serialization and validation."""

from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: int = Field(..., description="Top-left X coordinate in pixels")
    y_min: int = Field(..., description="Top-left Y coordinate in pixels")
    x_max: int = Field(..., description="Bottom-right X coordinate in pixels")
    y_max: int = Field(..., description="Bottom-right Y coordinate in pixels")


class DefectItem(BaseModel):
    class_id: int = Field(..., description="Defect class index")
    class_name: str = Field(..., description="Human-readable defect name")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    box: BoundingBox


class LatencyBreakdown(BaseModel):
    preprocess_ms: float = Field(..., description="Preprocessing time in ms")
    inference_ms: float = Field(..., description="Model inference time in ms")
    postprocess_ms: float = Field(..., description="NMS post-processing time in ms")
    total_ms: float = Field(..., description="End-to-end total latency in ms")


class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    defect_count: int
    defects: List[DefectItem]
    latency: LatencyBreakdown


class HealthResponse(BaseModel):
    status: str
    app_version: str
    model_loaded: bool
    model_type: str
    classes: List[str]
