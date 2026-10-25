"""Dataset validation and integrity checking for YOLO industrial defect detection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np


@dataclass
class ValidationReport:
    total_images: int = 0
    total_labels: int = 0
    missing_labels: List[str] = field(default_factory=list)
    orphaned_labels: List[str] = field(default_factory=list)
    corrupted_images: List[str] = field(default_factory=list)
    out_of_bounds_boxes: List[Tuple[str, int, List[float]]] = field(default_factory=list)
    class_distribution: Dict[int, int] = field(default_factory=dict)
    images_with_defects: int = 0
    clean_images: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            len(self.corrupted_images) == 0
            and len(self.out_of_bounds_boxes) == 0
            and len(self.orphaned_labels) == 0
        )


class DatasetValidator:
    """Validates YOLO dataset integrity, coordinates and class balances."""

    VALID_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, dataset_root: str | Path, num_classes: int = 6):
        self.root = Path(dataset_root)
        self.num_classes = num_classes

    def validate_split(self, split: str = "train") -> ValidationReport:
        """Validates an individual split directory (e.g. 'train', 'val', 'test')."""
        images_dir = self.root / "images" / split
        labels_dir = self.root / "labels" / split

        report = ValidationReport()
        if not images_dir.exists():
            return report

        image_files = {
            f.stem: f
            for f in images_dir.iterdir()
            if f.suffix.lower() in self.VALID_IMAGE_EXTS
        }
        label_files = {
            f.stem: f
            for f in labels_dir.iterdir()
            if f.suffix.lower() == ".txt"
        } if labels_dir.exists() else {}

        report.total_images = len(image_files)
        report.total_labels = len(label_files)

        # Check orphaned labels (label file exists with no corresponding image)
        for stem in label_files:
            if stem not in image_files:
                report.orphaned_labels.append(str(label_files[stem]))

        # Validate images and their bounding box labels
        for stem, img_path in image_files.items():
            # Test image readability
            img = cv2.imread(str(img_path))
            if img is None:
                report.corrupted_images.append(str(img_path))
                continue

            h, w = img.shape[:2]
            if h == 0 or w == 0:
                report.corrupted_images.append(str(img_path))
                continue

            # Check corresponding label
            lbl_path = label_files.get(stem)
            if not lbl_path or not lbl_path.exists():
                # Clean product with zero defects
                report.clean_images += 1
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                report.clean_images += 1
                continue

            report.images_with_defects += 1
            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    report.out_of_bounds_boxes.append((str(lbl_path), line_idx, []))
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                except ValueError:
                    report.out_of_bounds_boxes.append((str(lbl_path), line_idx, []))
                    continue

                # Coordinate bounds validation [0, 1]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    report.out_of_bounds_boxes.append((str(lbl_path), line_idx, [xc, yc, bw, bh]))

                if 0 <= cls_id < self.num_classes:
                    report.class_distribution[cls_id] = report.class_distribution.get(cls_id, 0) + 1

        return report
