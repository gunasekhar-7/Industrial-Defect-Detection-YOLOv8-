"""Header-first upload guard (review item 3): oversize is rejected before decode."""
import io

import pytest
from PIL import Image

from src.vision.image_io import decode_image


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
    with pytest.raises(ValueError):
        decode_image(b"\xff\xd8\xffgarbage", 4096)
