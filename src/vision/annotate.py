"""Bounding-box annotation for detections — the one place boxes are drawn.

The API's ``/v1/detect/annotated`` endpoint and the Streamlit dashboard
both render detections through :func:`draw_detections`, so label style,
thickness and palette handling cannot drift between the two surfaces.
"""

from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np

BGR = Tuple[int, int, int]
# Detection payloads arrive as parsed JSON from the detector, so the
# values are intentionally Any rather than a rigid schema.
Detection = Dict[str, Any]

_FALLBACK_COLOR: BGR = (0, 255, 0)
_BOX_THICKNESS = 2
_LABEL_SCALE = 0.5
_LABEL_THICKNESS = 1


def draw_detections(
    img_bgr: np.ndarray,
    detections: Sequence[Detection],
    class_names: Sequence[str],
    colors: Dict[int, BGR],
) -> np.ndarray:
    """Return a copy of ``img_bgr`` with a labelled box per detection.

    Each detection is a dict with ``class_id`` (int), ``confidence``
    (float) and ``box`` (``x_min``/``y_min``/``x_max``/``y_max``).
    Unknown class ids fall back to ``_FALLBACK_COLOR`` instead of
    raising, so a mis-indexed model output still renders.
    """
    annotated = img_bgr.copy()
    for d in detections:
        cls_id = int(d["class_id"])
        conf = float(d["confidence"])
        box = d["box"]
        x1 = int(box["x_min"])
        y1 = int(box["y_min"])
        x2 = int(box["x_max"])
        y2 = int(box["y_max"])

        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        color = colors.get(cls_id, _FALLBACK_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
        label = f"{name} {conf:.2f}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, _LABEL_THICKNESS)
        # Shift the label block left when it would run past the right edge,
        # otherwise the text is silently clipped mid-word.
        bg_x1 = max(0, min(x1, annotated.shape[1] - (tw + 6)))
        cv2.rectangle(
            annotated,
            (bg_x1, max(0, y1 - 20)),
            (bg_x1 + tw + 6, max(0, y1)),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (bg_x1 + 3, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            _LABEL_SCALE,
            (0, 0, 0),
            _LABEL_THICKNESS,
            cv2.LINE_AA,
        )
    return annotated
