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
        detail="Unsupported file format. Only JPEG, PNG, and BMP are accepted.",
    )


def sanitize_and_read_image(file: UploadFile) -> Tuple[np.ndarray, int, int]:
    """Reads, verifies magic bytes, and validates image dimensions within safe operational bounds.

    Synchronous by design: called from sync (threadpool) route handlers.
    The body is consumed in 1 MiB chunks and reading aborts the moment the
    running total exceeds MAX_IMAGE_SIZE_MB, so an oversized upload is
    rejected without ever being fully buffered in memory.
    """
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    chunk_size = 1024 * 1024

    chunks = []
    total = 0
    header = b""
    magic_checked = False

    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size ({total / (1024 * 1024):.2f}MB) exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit.",
            )
        if not magic_checked:
            header += chunk
            if len(header) >= 16:
                verify_image_magic_bytes(header[:16])
                magic_checked = True
        chunks.append(chunk)

    content_length = total

    if content_length < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty or too small to be a valid image.",
        )

    # Magic byte verification (deferred if the file was tiny but >= 16 bytes
    # spread across chunks is impossible; header holds the first 16+ bytes)
    if not magic_checked:
        verify_image_magic_bytes(header)

    content = b"".join(chunks)

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
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions ({width}x{height}) exceed permitted limits "
                f"[{settings.MIN_IMAGE_DIMENSION} - {settings.MAX_IMAGE_DIMENSION} px]."
            ),
        )

    if (
        width < settings.MIN_IMAGE_DIMENSION
        or height < settings.MIN_IMAGE_DIMENSION
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions ({width}x{height}) are below minimum "
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
