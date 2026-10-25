"""Tests for the shared detection-annotation helper (single source of truth)."""

import numpy as np

from src.api.config import settings
from src.vision.annotate import draw_detections


def _det(x1: int, y1: int, x2: int, y2: int, cls_id: int = 0, conf: float = 0.9) -> dict:
    return {
        "class_id": cls_id,
        "confidence": conf,
        "box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
    }


def test_drawing_changes_pixels_at_box_corner() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [_det(20, 20, 80, 80)], settings.CLASS_NAMES, settings.CLASS_COLORS)
    color = settings.CLASS_COLORS[0]
    # 2px-thick outline means the pixel at the corner itself is painted.
    assert tuple(int(v) for v in out[20, 20]) == tuple(color)
    # Original must not be mutated.
    assert tuple(int(v) for v in img[20, 20]) == (0, 0, 0)


def test_empty_detections_returns_untouched_copy() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    out = draw_detections(img, [], settings.CLASS_NAMES, settings.CLASS_COLORS)
    assert out is not img
    assert np.array_equal(out, img)


def test_unknown_class_id_falls_back_without_crashing() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [_det(10, 50, 90, 90, cls_id=99)], settings.CLASS_NAMES, settings.CLASS_COLORS)
    assert out.shape == img.shape
    # Fallback color (green) painted on the outline.
    assert tuple(int(v) for v in out[50, 10]) == (0, 255, 0)


def test_class_colors_come_from_config_once() -> None:
    # Single source of truth: settings carry the palette used everywhere.
    assert set(settings.CLASS_COLORS) == set(range(len(settings.CLASS_NAMES)))
    for color in settings.CLASS_COLORS.values():
        assert len(color) == 3
