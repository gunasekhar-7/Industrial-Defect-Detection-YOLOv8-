"""Header-first decoding for untrusted image uploads.

Reads only the image header (Pillow's lazy open) to enforce a maximum
dimension BEFORE any pixel buffer is allocated, so an oversized file
cannot exhaust memory via cv2.imdecode. Used by the Streamlit dashboard;
raises ValueError with a human-readable reason on every rejection.
"""

import io

import cv2
import numpy as np
from PIL import Image


def decode_image(data: bytes, max_dim: int) -> np.ndarray:
    """Decode ``data`` into a BGR image, rejecting anything wider or taller
    than ``max_dim`` before pixel decoding, per review item 3.

    Raises ValueError for empty, unreadable, oversize, or undecodable input.
    """
    if not data:
        raise ValueError("empty image data")

    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
    except Exception as exc:
        raise ValueError(f"unreadable image: {exc}") from exc

    if width > max_dim or height > max_dim:
        raise ValueError(f"image dimensions {width}x{height} exceed maximum {max_dim}px")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image data")
    return img
