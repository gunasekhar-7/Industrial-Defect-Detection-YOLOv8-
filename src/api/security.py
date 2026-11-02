"""Security validation boundaries and image sanitization for API endpoints."""

import io
from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.config import settings

# Magic byte signatures for image formats
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
}

# Bytes allowed above the image cap for multipart framing (boundaries, part
# headers, sibling form fields) before the whole-request limit trips.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


def body_limit_bytes() -> int:
    """Whole-request budget: the configured image cap plus multipart overhead."""
    return settings.MAX_IMAGE_SIZE_MB * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES


def _declared_content_length(scope: Scope) -> int | None:
    """Content-Length from the raw ASGI headers, or None when absent/invalid."""
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class BodySizeLimitMiddleware:
    """Reject declared-oversize bodies before Starlette parses the form.

    The streaming cap in :func:`sanitize_and_read_image` bounds memory, but
    Starlette spools the whole multipart body to a temporary file before the
    endpoint runs, so bytes still reach disk. Answering 413 from
    ``Content-Length`` short-circuits that entirely. Bodies without a length
    (``Transfer-Encoding: chunked``) use the same total-body cap in this
    middleware before any request bytes reach the form parser.
    """

    def __init__(self, app: ASGIApp, max_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        limit = self.max_bytes if self.max_bytes is not None else body_limit_bytes()
        declared = _declared_content_length(scope)
        if declared is not None and declared > limit:
            response = JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={
                    "detail": (
                        f"Request body is {declared} bytes, which exceeds the "
                        f"{limit}-byte request budget "
                        f"({settings.MAX_IMAGE_SIZE_MB}MB image cap plus multipart overhead)."
                    )
                },
            )
            await response(scope, receive, send)
            return

        total_bytes = 0
        over_limit = False

        async def limited_receive() -> Message:
            nonlocal total_bytes, over_limit
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > limit:
                    # Do not expose any part of this chunk to Starlette's form
                    # parser, which would otherwise spool it to a temp file.
                    over_limit = True
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message: Message) -> None:
            # The downstream app may parse the synthetic terminal request and
            # send a 400; suppress it so the caller receives the size-limit 413.
            if not over_limit:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not over_limit:
                raise

        if over_limit:
            response = JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={
                    "detail": (
                        f"Request body exceeds the {limit}-byte request budget "
                        f"({settings.MAX_IMAGE_SIZE_MB}MB image cap plus multipart overhead)."
                    )
                },
            )
            await response(scope, receive, send)


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
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
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

    # Magic bytes were verified in the loop: `header` accumulates the first
    # chunk, so reaching 16+ total bytes always means 16+ header bytes.
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
