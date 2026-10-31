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


def _run_middleware(content_length: bytes | None) -> tuple[bool, list]:
    """Drive BodySizeLimitMiddleware with a synthetic ASGI scope.

    Returns (inner_app_reached, messages_sent).
    """
    import asyncio

    from src.api.security import BodySizeLimitMiddleware

    reached = False

    async def inner(scope, receive, send):
        nonlocal reached
        reached = True

    messages: list = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    headers = [] if content_length is None else [(b"content-length", content_length)]
    scope = {"type": "http", "method": "POST", "path": "/v1/detect", "headers": headers}
    asyncio.run(BodySizeLimitMiddleware(inner)(scope, receive, send))
    return reached, messages


def test_body_limit_rejects_before_form_parsing():
    """A declared body over the cap must 413 without the app ever running,
    so Starlette never spools the multipart file to disk."""
    reached, messages = _run_middleware(b"%d" % (64 * 1024 * 1024))

    assert not reached, "oversize body reached the app — it would be spooled to disk"
    assert messages[0]["status"] == 413


def test_body_limit_allows_small_and_unlength_chunked_bodies():
    """Small bodies and chunked uploads (no Content-Length) reach the app,
    where the streaming cap still applies."""
    reached_small, messages_small = _run_middleware(b"1024")
    assert reached_small and not messages_small

    reached_chunked, messages_chunked = _run_middleware(None)
    assert reached_chunked and not messages_chunked


def test_body_limit_stops_oversize_chunked_body_before_parser():
    """Chunked bodies must be capped as they stream, before multipart spooling."""
    import asyncio

    from src.api.security import BodySizeLimitMiddleware

    chunks = iter(
        [
            {"type": "http.request", "body": b"12345678", "more_body": True},
            {"type": "http.request", "body": b"90123", "more_body": False},
        ]
    )
    downstream_bytes = 0
    messages = []

    async def receive():
        return next(chunks)

    async def inner(scope, downstream_receive, send):
        nonlocal downstream_bytes
        while True:
            message = await downstream_receive()
            if message["type"] == "http.disconnect":
                return
            downstream_bytes += len(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"bad multipart"})

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": []}
    middleware = BodySizeLimitMiddleware(inner, max_bytes=10)
    asyncio.run(middleware(scope, receive, send))

    assert downstream_bytes == 8, "over-limit body fragment reached the downstream parser"
    assert messages[0]["status"] == 413


def test_body_limit_still_sends_real_asgi_disconnect():
    """Do not replace client disconnects with endless synthetic disconnects."""
    import asyncio

    from src.api.security import BodySizeLimitMiddleware

    calls = 0
    messages = []

    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": b"12345678", "more_body": True}
        return {"type": "http.disconnect"}

    async def inner(scope, downstream_receive, send):
        await downstream_receive()
        message = await downstream_receive()
        assert message["type"] == "http.disconnect"

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": []}
    asyncio.run(BodySizeLimitMiddleware(inner, max_bytes=10)(scope, receive, send))
    assert not messages


def test_detect_rejects_oversize_via_content_length(client, monkeypatch):
    """End-to-end: the header gate answers 413 for a body over the budget."""
    from src.api.config import settings

    monkeypatch.setattr(settings, "MAX_IMAGE_SIZE_MB", 1)
    payload = b"\xff\xd8\xff" + b"x" * (3 * 1024 * 1024)
    response = client.post(
        "/v1/detect",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


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


def test_oversize_413_raises_no_deprecation_warning():
    """Review item 4: the 413 path must use HTTP_413_CONTENT_TOO_LARGE —
    the deprecated REQUEST_ENTITY name emits StarletteDeprecationWarning."""
    import warnings

    from fastapi import HTTPException

    from src.api.security import sanitize_and_read_image

    payload = b"\xff\xd8\xff" + b"x" * (16 * 1024 * 1024)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(HTTPException) as exc:
            sanitize_and_read_image(_SpyUpload(payload))
    assert exc.value.status_code == 413
    stale = [w for w in caught if "HTTP_413_REQUEST_ENTITY_TOO_LARGE" in str(w.message)]
    assert not stale, "deprecated Starlette status constant still referenced"


def test_body_limit_ignores_unparseable_content_length():
    """A garbage Content-Length must not bypass the gate nor crash it: the
    header is ignored and the streaming cap still applies."""
    reached, messages = _run_middleware(b"not-a-number")

    assert reached, "unparseable Content-Length should fall through to the streaming cap"
    assert not messages


def test_body_limit_reraises_downstream_failures():
    """A body within budget must surface the app's own exception unchanged,
    not convert it into a misleading 413."""
    import asyncio

    from src.api.security import BodySizeLimitMiddleware

    async def inner(scope, receive, send):
        raise RuntimeError("handler exploded")

    async def receive():
        return {"type": "http.request", "body": b"tiny", "more_body": False}

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": []}
    with pytest.raises(RuntimeError, match="handler exploded"):
        asyncio.run(BodySizeLimitMiddleware(inner, max_bytes=1024)(scope, receive, send))


def test_corrupt_structure_reported_separately_from_magic_bytes():
    """Valid magic bytes over a broken body must say 'corrupted', not claim
    the format is unsupported."""
    from fastapi import HTTPException

    from src.api.security import sanitize_and_read_image

    payload = b"\xff\xd8\xff" + b"\x00" * 40
    with pytest.raises(HTTPException) as exc:
        sanitize_and_read_image(_SpyUpload(payload))

    assert exc.value.status_code == 400
    assert "Corrupted or invalid image structure" in exc.value.detail


def test_undecodable_pixels_are_rejected(monkeypatch):
    """A readable header whose pixels cv2 cannot decode is a 400, not a crash."""
    from fastapi import HTTPException

    from src.api import security

    payload = create_test_image_bytes("PNG", size=(64, 64))
    monkeypatch.setattr(security.cv2, "imdecode", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as exc:
        security.sanitize_and_read_image(_SpyUpload(payload))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to decode image data."


def test_load_model_survives_broken_onnx_and_absent_fallback(tmp_path, monkeypatch):
    """A corrupt ONNX export must not wedge startup: loading fails, the
    PyTorch fallback is tried, and a missing one leaves the service
    explicitly uninitialised so /health reports 503."""
    from src.api import app as app_module

    onnx = tmp_path / "broken.onnx"
    onnx.write_bytes(b"not an onnx graph")
    monkeypatch.setattr(app_module.settings, "MODEL_PATH", str(onnx))
    monkeypatch.setattr(app_module.settings, "FALLBACK_PYTORCH_PATH", str(tmp_path / "absent.pt"))
    monkeypatch.setattr(app_module, "model_engine", None)
    monkeypatch.setattr(app_module, "model_type", "none")

    def boom(*_args, **_kwargs):
        raise RuntimeError("bad graph")

    monkeypatch.setattr(app_module, "ONNXDefectDetector", boom)
    app_module.load_model()

    assert app_module.model_engine is None
    assert app_module.model_type == "not_initialized"


def test_load_model_falls_back_to_pytorch(tmp_path, monkeypatch):
    """With no ONNX model, an existing .pt must be loaded through YOLO."""
    import sys
    import types

    from src.api import app as app_module

    weights = tmp_path / "fallback.pt"
    weights.write_bytes(b"fake weights")
    monkeypatch.setattr(app_module.settings, "MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(app_module.settings, "FALLBACK_PYTORCH_PATH", str(weights))
    monkeypatch.setattr(app_module, "model_engine", None)
    monkeypatch.setattr(app_module, "model_type", "none")

    loaded = {}

    class FakeYOLO:
        def __init__(self, path):
            loaded["path"] = path

    stub = types.ModuleType("ultralytics")
    stub.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", stub)

    app_module.load_model()

    assert loaded["path"] == str(weights)
    assert app_module.model_type == "pytorch_yolo"
    assert isinstance(app_module.model_engine, FakeYOLO)


def test_load_model_reports_uninitialised_when_pytorch_fallback_fails(tmp_path, monkeypatch):
    """A .pt that ultralytics cannot load must leave the service uninitialised
    instead of keeping a half-built engine."""
    import sys
    import types

    from src.api import app as app_module

    weights = tmp_path / "fallback.pt"
    weights.write_bytes(b"fake weights")
    monkeypatch.setattr(app_module.settings, "MODEL_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(app_module.settings, "FALLBACK_PYTORCH_PATH", str(weights))
    monkeypatch.setattr(app_module, "model_engine", None)
    monkeypatch.setattr(app_module, "model_type", "none")

    def broken_yolo(_path):
        raise ValueError("unsupported checkpoint")

    stub = types.ModuleType("ultralytics")
    stub.YOLO = broken_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", stub)

    app_module.load_model()

    assert app_module.model_engine is None
    assert app_module.model_type == "not_initialized"


def test_run_detection_pytorch_fallback_parses_boxes(monkeypatch):
    """The PyTorch fallback must translate ultralytics boxes into the same
    detection dict shape the ONNX path returns."""
    import numpy as np

    from src.api import app as app_module

    class FakeBox:
        cls = np.array([2.0])
        conf = np.array([0.9012])
        xyxy = np.array([[1.0, 2.0, 30.0, 40.0]])

    class FakeResult:
        boxes = [FakeBox()]

    class FakeEngine:
        def predict(self, img, conf=None, iou=None):
            assert conf == app_module.settings.CONFIDENCE_THRESHOLD
            return [FakeResult()]

    monkeypatch.setattr(app_module, "model_engine", FakeEngine())
    monkeypatch.setattr(app_module, "model_type", "pytorch_yolo")

    detections, latencies = app_module.run_detection(np.zeros((8, 8, 3), dtype=np.uint8))

    assert detections == [
        {
            "class_id": 2,
            "confidence": 0.9012,
            "box": {"x_min": 1, "y_min": 2, "x_max": 30, "y_max": 40},
        }
    ]
    assert latencies["preprocess_ms"] == 0.0
    assert latencies["postprocess_ms"] == 0.0
    assert latencies["total_ms"] == latencies["inference_ms"]


def test_detect_labels_out_of_range_class_ids_as_unknown(client, monkeypatch):
    """A model emitting an unexpected class id must be labelled, not crash
    with an IndexError deep inside the response builder."""
    from src.api import app as app_module

    latencies = {
        "preprocess_ms": 0.0,
        "inference_ms": 0.0,
        "postprocess_ms": 0.0,
        "total_ms": 0.0,
    }

    class FakeEngine:
        def predict(self, img):
            return (
                [
                    {
                        "class_id": 99,
                        "confidence": 0.5,
                        "box": {"x_min": 1, "y_min": 1, "x_max": 4, "y_max": 4},
                    }
                ],
                latencies,
            )

    monkeypatch.setattr(app_module, "model_engine", FakeEngine())
    monkeypatch.setattr(app_module, "model_type", "onnx_runtime")

    response = client.post(
        "/v1/detect",
        files={"file": ("part.jpg", create_test_image_bytes("JPEG"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["defects"][0]["class_name"] == "unknown_99"


def test_annotated_returns_500_when_jpeg_encoding_fails(client, monkeypatch):
    """If cv2 cannot encode the annotated frame the endpoint must answer a
    clean 500 rather than returning an empty 200 image."""
    from src.api import app as app_module

    class FakeEngine:
        def predict(self, img):
            return [], {
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": 0.0,
            }

    monkeypatch.setattr(app_module, "model_engine", FakeEngine())
    monkeypatch.setattr(app_module, "model_type", "onnx_runtime")
    monkeypatch.setattr(app_module.cv2, "imencode", lambda *_args, **_kwargs: (False, None))

    response = client.post(
        "/v1/detect/annotated",
        files={"file": ("part.jpg", create_test_image_bytes("JPEG"), "image/jpeg")},
    )

    assert response.status_code == 500


def test_rate_limit_exceeded_returns_429(client):
    """The registered slowapi handler (not Starlette's default) answers 429,
    so a throttled client gets the documented error payload."""
    from fastapi import Request

    from src.api import app as app_module

    @app_module.app.get("/_tests/rate-limited")
    @app_module.limiter.limit("1/minute")
    def rate_limited(request: Request) -> dict:
        return {"ok": True}

    assert client.get("/_tests/rate-limited").status_code == 200
    throttled = client.get("/_tests/rate-limited")

    assert throttled.status_code == 429
    assert "Rate limit exceeded" in throttled.text
