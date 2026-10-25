"""Integration tests for FastAPI endpoints and security validation boundaries."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.api.app import app


@pytest.fixture
def client():
    # Context manager runs lifespan -> load_model() so the 200 path is covered.
    with TestClient(app) as c:
        yield c


def create_test_image_bytes(format_type: str = "JPEG", size=(640, 640), color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_type)
    return buf.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "classes" in data
    assert len(data["classes"]) == 6


def test_health_503_when_model_missing(client, monkeypatch):
    """Health must report unhealthy (503) when no model is loaded, so orchestrators fail."""
    monkeypatch.setattr("src.api.app.model_engine", None)
    monkeypatch.setattr("src.api.app.model_type", "not_initialized")
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_detect_503_when_model_missing(client, monkeypatch):
    """Detect must fail loudly (503) instead of returning fake zero-detection results."""
    monkeypatch.setattr("src.api.app.model_engine", None)
    monkeypatch.setattr("src.api.app.load_model", lambda: None)
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/v1/detect",
        files={"file": ("test_part.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "model not loaded"


def test_detect_annotated_503_when_model_missing(client, monkeypatch):
    """Annotated detect must also 503 when no model is loaded."""
    monkeypatch.setattr("src.api.app.model_engine", None)
    monkeypatch.setattr("src.api.app.load_model", lambda: None)
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/v1/detect/annotated",
        files={"file": ("test_part.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "model not loaded"


def test_detect_valid_image(client):
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/v1/detect",
        files={"file": ("test_part.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image_width"] == 640
    assert data["image_height"] == 640
    assert "defects" in data
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0


def test_detect_annotated_valid_image(client):
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/v1/detect/annotated",
        files={"file": ("test_part.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 100


def test_security_reject_invalid_mime(client):
    """Ensure non-image file uploads are blocked with 415 Unsupported Media Type."""
    malicious_payload = b"#!/bin/bash\necho 'hacked'"
    response = client.post(
        "/v1/detect",
        files={"file": ("script.sh", malicious_payload, "text/plain")},
    )
    assert response.status_code == 415


def test_security_reject_empty_file(client):
    """Ensure empty files are blocked with 400 Bad Request."""
    response = client.post(
        "/v1/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_security_reject_oversized_dimensions(client):
    """Ensure decompression bombs exceeding MAX_IMAGE_DIMENSION are rejected."""
    # Create image header that claims 5000x5000
    img = Image.new("RGB", (5000, 5000), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    response = client.post(
        "/v1/detect",
        files={"file": ("huge.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceed permitted limits" in response.json()["detail"]
