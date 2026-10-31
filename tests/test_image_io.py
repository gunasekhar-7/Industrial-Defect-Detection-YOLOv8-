"""Header-first upload guard (review item 3): oversize is rejected before decode."""
import io

import pytest
from PIL import Image

from src.vision.image_io import check_upload_size, decode_image


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_oversize_rejected_before_pixel_decode(monkeypatch):
    import src.vision.image_io as mod

    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(mod.cv2, "imdecode", spy)
    with pytest.raises(ValueError, match="exceed maximum 4096px"):
        decode_image(_jpeg(5000, 5000), 4096)
    assert calls["n"] == 0, "guard must reject before cv2.imdecode runs"


def test_valid_image_decodes_to_bgr():
    img = decode_image(_jpeg(640, 480), 4096)
    assert img.shape == (480, 640, 3)


def test_corrupt_bytes_rejected():
    with pytest.raises(ValueError, match="unreadable image"):
        decode_image(b"\xff\xd8\xffgarbage", 4096)


def test_empty_bytes_rejected():
    with pytest.raises(ValueError, match="empty image data"):
        decode_image(b"", 4096)


def test_decoder_failure_after_valid_header_is_reported(monkeypatch):
    import src.vision.image_io as mod

    monkeypatch.setattr(mod.cv2, "imdecode", lambda *_args: None)
    with pytest.raises(ValueError, match="failed to decode image data"):
        decode_image(_jpeg(24, 24), 4096)


def test_upload_size_guard_boundaries():
    """At the cap is allowed; one byte over is rejected with the limit named."""
    limit = 1024 * 1024
    check_upload_size(limit, limit)
    with pytest.raises(ValueError, match="over the 1MB upload limit"):
        check_upload_size(limit + 1, limit)
