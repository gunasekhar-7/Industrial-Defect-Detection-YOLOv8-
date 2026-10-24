"""Security validation boundaries and image sanitization for API endpoints."""

import io
from typing import Tuple
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image

from src.api.config import settings


# Magic byte signatures for image formats
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
}


def verify_image_magic_bytes(header: bytes) -> str:
    """Verifies file header matches legitimate image magic signatures."""
    for magic, mime in MAGIC_NUMBERS.items():
        if header.startswith(magic):
            return mime
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Unsupported file format. Only JPEG, PNG, and BMP are accepted.",
    )


async def sanitize_and_read_image(file: UploadFile) -> Tuple[np.ndarray, int, int]:
    """Reads, verifies magic bytes, and validates image dimensions within safe operational bounds."""
    # Read raw content
    content = await file.read()
    content_length = len(content)

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if content_length > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size ({content_length / (1024 * 1024):.2f}MB) exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit.",
        )

    if content_length < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty or too small to be a valid image.",
        )

    # Magic byte verification
    verify_image_magic_bytes(content[:16])

    # Decompression bomb safety check via PIL without loading full pixel buffer into memory
    try:
        with Image.open(io.BytesIO(content)) as pil_img:
            width, height = pil_img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image structure: {str(e)}",
        )

    if (
        width > settings.MAX_IMAGE_DIMENSION
        or height > settings.MAX_IMAGE_DIMENSION
        or width < settings.MIN_IMAGE_DIMENSION
        or height < settings.MIN_IMAGE_DIMENSION
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions ({width}x{height}) exceed permitted limits "
                f"[{settings.MIN_IMAGE_DIMENSION} - {settings.MAX_IMAGE_DIMENSION} px]."
            ),
        )

    # Decode with OpenCV to standard BGR
    np_arr = np.frombuffer(content, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image data.",
        )

    return img_bgr, width, height
