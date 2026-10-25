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


def test_detect_handlers_do_not_block_event_loop():
    """Detect handlers must be sync defs: Starlette runs those in its
    threadpool, while async handlers would block the event loop for the
    full ~200ms inference, stalling every other request (e.g. /health)."""
    import inspect

    from src.api import app as app_module

    paths = {"/v1/detect", "/v1/detect/annotated"}
    found = set()
    for route in app_module.app.routes:
        p = getattr(route, "path", None)
        if p in paths:
            found.add(p)
            assert not inspect.iscoroutinefunction(route.endpoint), (
                f"{p} is an async handler and blocks the event loop"
            )
    assert found == paths, f"missing routes: {paths - found}"


class _CountingReader:
    """File-like wrapper that counts how many bytes the app pulls in."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._buf.read(size)
        self.bytes_read += len(chunk)
        return chunk


class _SpyUpload:
    """Duck-typed UploadFile exposing only the .file attribute sanitize uses."""

    def __init__(self, data: bytes):
        self.file = _CountingReader(data)


def test_payload_cap_reads_bounded_amount():
    """Oversized upload must be rejected after reading at most cap + 1 chunk,
    not after buffering the entire body (the old DoS-prone behavior)."""
    from fastapi import HTTPException

    from src.api.config import settings
    from src.api.security import sanitize_and_read_image

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    # Valid JPEG magic prefix so the payload reaches the SIZE cap (not the
    # magic-byte check) — this test is about bounded reads on oversize bodies.
    payload = b"\xff\xd8\xff" + b"x" * (20 * 1024 * 1024)  # ~20 MiB, limit 15 MiB
    spy = _SpyUpload(payload)

    with pytest.raises(HTTPException) as exc:
        sanitize_and_read_image(spy)

    assert exc.value.status_code == 413
    # Old code read all 20 MiB; new code must stop within cap + one 1 MiB chunk.
    assert spy.file.bytes_read <= max_bytes + 1024 * 1024, (
        f"read {spy.file.bytes_read} bytes — payload cap did not abort early"
    )


def test_detect_rejects_oversized_payload(client):
    """End-to-end: a body over MAX_IMAGE_SIZE_MB gets 413."""
    # JPEG magic prefix: exercises the size cap, not the magic-byte check.
    huge = b"\xff\xd8\xff" + b"x" * (16 * 1024 * 1024)  # ~16 MiB > 15 MiB limit
    response = client.post(
        "/v1/detect",
        files={"file": ("huge.jpg", huge, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_cors_configured_origins_only(client):
    """CORS must echo an explicitly configured origin, never '*', and must
    not grant CORS to arbitrary origins."""
    from src.api.config import settings

    assert "*" not in settings.ALLOWED_ORIGINS, "wildcard origin is never allowed"

    allowed = settings.ALLOWED_ORIGINS[0]
    r = client.options(
        "/v1/detect",
        headers={
            "Origin": allowed,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == allowed

    r2 = client.options(
        "/v1/detect",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r2.headers, (
        "unconfigured origin must get no CORS grant"
    )


def test_load_model_is_thread_safe_and_once(client, monkeypatch):
    """Concurrent first requests must construct the model session exactly
    once (lock + already-loaded short-circuit), not once per caller."""
    import threading
    import time

    from src.api import app as app_module

    constructions = 0

    class CountingDetector:
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            time.sleep(0.05)  # widen the race window

    monkeypatch.setattr(app_module, "ONNXDefectDetector", CountingDetector)
    monkeypatch.setattr(app_module, "model_engine", None)
    monkeypatch.setattr(app_module, "model_type", "none")

    threads = [threading.Thread(target=app_module.load_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert constructions == 1, f"model constructed {constructions} times, expected 1"
    # Second call after load must be a no-op (short-circuit).
    app_module.load_model()
    assert constructions == 1, "load_model re-ran despite model already loaded"


def test_dimension_errors_are_honest(client):
    """Too-small images report 'below minimum'; oversized images keep
    'exceed permitted limits' — the old code said 'exceed' for both."""
    # Below minimum (16x16 < MIN_IMAGE_DIMENSION=32)
    tiny = Image.new("RGB", (16, 16), color=(10, 10, 10))
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    r = client.post(
        "/v1/detect",
        files={"file": ("tiny.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 400
    assert "below minimum" in r.json()["detail"], r.json()["detail"]

    # Over maximum (5000x5000 > MAX_IMAGE_DIMENSION=4096)
    huge_img = Image.new("RGB", (5000, 5000), color=(100, 100, 100))
    buf2 = io.BytesIO()
    huge_img.save(buf2, format="JPEG")
    r2 = client.post(
        "/v1/detect",
        files={"file": ("huge.jpg", buf2.getvalue(), "image/jpeg")},
    )
    assert r2.status_code == 400
    assert "exceed permitted limits" in r2.json()["detail"], r2.json()["detail"]


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
