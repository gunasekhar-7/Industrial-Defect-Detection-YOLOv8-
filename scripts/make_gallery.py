"""Render a ground-truth vs prediction gallery for the weakest classes.

Reviewers ask the same question about a defect detector: what does it get
wrong, and how? This renders the two weakest classes from
``reports/eval_report.json`` — two validation frames each — with ground
truth on the left and the model's predictions on the right.

    python scripts/make_gallery.py

Writes ``docs/failure_gallery.png`` (embedded in the README).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Run as a plain script: make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.config import settings  # noqa: E402
from src.models.benchmark import ONNXDefectDetector  # noqa: E402
from src.vision.annotate import draw_detections  # noqa: E402

GT_COLOR = (255, 255, 255)
GT_LABEL_COLOR = (0, 0, 0)
PANEL_PAD = 8
CAPTION_H = 22


def read_labels(label_path: Path, width: int, height: int) -> List[Dict]:
    """Convert a YOLO label file into detection dicts with confidence 1.0."""
    truth: List[Dict] = []
    if not label_path.exists():
        return truth
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:])
        truth.append(
            {
                "class_id": cls_id,
                "confidence": 1.0,
                "box": {
                    "x_min": int((xc - w / 2) * width),
                    "y_min": int((yc - h / 2) * height),
                    "x_max": int((xc + w / 2) * width),
                    "y_max": int((yc + h / 2) * height),
                },
            }
        )
    return truth


def panel(img_bgr: np.ndarray, caption: str) -> np.ndarray:
    bar = np.full((CAPTION_H, img_bgr.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        bar, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA
    )
    return np.vstack([bar, img_bgr])


def build_gallery(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    classes: List[str],
    frames_per_class: int = 2,
) -> np.ndarray:
    detector = ONNXDefectDetector(model_path)
    rows: List[np.ndarray] = []
    for class_id, class_name in classes:
        chosen: List[Path] = []
        for label_path in sorted(Path(labels_dir).glob("*.txt")):
            if len(chosen) == frames_per_class:
                break
            ids = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}
            if class_id in ids:
                chosen.append(label_path)
        cells = []
        for label_path in chosen:
            img_path = Path(images_dir) / f"{label_path.stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            truth = [t for t in read_labels(label_path, img.shape[1], img.shape[0]) if t["class_id"] == class_id]
            detections, _ = detector.predict(img)
            predicted = [d for d in detections if d["class_id"] == class_id]
            left = draw_detections(
                img, truth, settings.CLASS_NAMES, {class_id: GT_COLOR}, show_confidence=False
            )
            right = draw_detections(img, predicted, settings.CLASS_NAMES, settings.CLASS_COLORS)
            pair = np.hstack([panel(left, f"{class_name}: ground truth"), panel(right, f"{class_name}: predicted")])
            cells.append(pair)
        if cells:
            rows.append(np.hstack(cells))
    if not rows:
        raise SystemExit("no frames matched the requested classes")
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the GT-vs-prediction gallery")
    parser.add_argument("--onnx", default="models/industrial_yolov8n.onnx")
    parser.add_argument("--images", default="data/processed/images/val")
    parser.add_argument("--labels", default="data/processed/labels/val")
    parser.add_argument("--out", default="docs/failure_gallery.png")
    parser.add_argument("--classes", nargs="*", default=None)
    args = parser.parse_args(argv)

    if args.classes:
        wanted = [(settings.CLASS_NAMES.index(name), name) for name in args.classes]
    else:
        report = json.loads(Path("reports/eval_report.json").read_text(encoding="utf-8"))
        ranked: List[Tuple[float, str]] = sorted(
            (v["mAP50_95"], name) for name, v in report["per_class"].items()
        )
        wanted = [(settings.CLASS_NAMES.index(name), name) for _, name in ranked[:2]]

    gallery = build_gallery(args.onnx, args.images, args.labels, wanted)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), gallery)
    print(f"Wrote {out} ({gallery.shape[1]}x{gallery.shape[0]}): {[name for _, name in wanted]}")


if __name__ == "__main__":
    main()
