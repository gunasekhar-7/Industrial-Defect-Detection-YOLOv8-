"""Realistic synthetic industrial surface defect generator.

Generates industrial metallic surface textures and procedurally stamps defects
(crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches)
with strict ground-truth bounding boxes in YOLO format.
Used for CI pipelines, local testing, and offline benchmarking.
"""

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

CLASS_MAP = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
}


def create_metallic_background(width: int = 640, height: int = 640) -> np.ndarray:
    """Generates a realistic brushed metal / cold-rolled steel surface."""
    base_val = random.randint(120, 180)
    img = np.full((height, width), base_val, dtype=np.uint8)

    # Add directional brushed grain (horizontal streaks)
    grain = np.random.normal(0, random.uniform(8, 16), (height, width // 4)).astype(np.float32)
    grain = cv2.resize(grain, (width, height), interpolation=cv2.INTER_LINEAR)

    # Add non-uniform illumination gradient (vignetting / overhead factory lamp)
    y, x = np.ogrid[:height, :width]
    center_x = random.randint(width // 4, 3 * width // 4)
    center_y = random.randint(height // 4, 3 * height // 4)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(width ** 2 + height ** 2)
    illumination = 1.0 - 0.25 * (dist / max_dist)

    composite = (img.astype(np.float32) + grain) * illumination
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)


def stamp_scratch(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws sharp scratch / gouge defect across surface."""
    h, w = img.shape[:2]
    x1 = random.randint(30, w - 80)
    y1 = random.randint(30, h - 80)
    length = random.randint(60, 180)
    angle = random.uniform(-np.pi, np.pi)
    x2 = int(np.clip(x1 + length * np.cos(angle), 10, w - 10))
    y2 = int(np.clip(y1 + length * np.sin(angle), 10, h - 10))

    color_val = random.choice([30, 240])  # dark scratch or bright reflection
    thickness = random.randint(2, 4)
    cv2.line(img, (x1, y1), (x2, y2), (color_val, color_val, color_val), thickness, cv2.LINE_AA)

    # Box around scratch
    min_x = max(0, min(x1, x2) - 8)
    max_x = min(w, max(x1, x2) + 8)
    min_y = max(0, min(y1, y2) - 8)
    max_y = min(h, max(y1, y2) + 8)
    return min_x, min_y, max_x - min_x, max_y - min_y


def stamp_crazing(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws network of branching micro-cracks (crazing)."""
    h, w = img.shape[:2]
    cx = random.randint(80, w - 80)
    cy = random.randint(80, h - 80)
    radius = random.randint(35, 75)

    pts = []
    curr_x, curr_y = cx, cy
    for _ in range(random.randint(6, 12)):
        nx = int(np.clip(curr_x + random.randint(-radius // 2, radius // 2), 10, w - 10))
        ny = int(np.clip(curr_y + random.randint(-radius // 2, radius // 2), 10, h - 10))
        pts.append((nx, ny))
        cv2.line(img, (curr_x, curr_y), (nx, ny), (40, 40, 40), 1, cv2.LINE_AA)
        curr_x, curr_y = nx, ny

    all_x = [p[0] for p in pts] + [cx]
    all_y = [p[1] for p in pts] + [cy]
    min_x = max(0, min(all_x) - 6)
    max_x = min(w, max(all_x) + 6)
    min_y = max(0, min(all_y) - 6)
    max_y = min(h, max(all_y) + 6)
    return min_x, min_y, max_x - min_x, max_y - min_y


def stamp_inclusion(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws dark embedded foreign particulate matter."""
    h, w = img.shape[:2]
    cx = random.randint(40, w - 60)
    cy = random.randint(40, h - 60)
    size = random.randint(15, 45)

    # Irregular polygon
    num_pts = random.randint(5, 8)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
    radii = size + np.random.uniform(-size // 3, size // 3, num_pts)
    poly = np.array([
        [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
        for a, r in zip(angles, radii)
    ], dtype=np.int32)

    val = random.randint(20, 60)
    cv2.fillPoly(img, [poly], (val, val, val))

    min_x = max(0, int(np.min(poly[:, 0])) - 4)
    max_x = min(w, int(np.max(poly[:, 0])) + 4)
    min_y = max(0, int(np.min(poly[:, 1])) - 4)
    max_y = min(h, int(np.max(poly[:, 1])) + 4)
    return min_x, min_y, max_x - min_x, max_y - min_y


def stamp_patches(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws surface oxidation / irregular discoloration patch."""
    h, w = img.shape[:2]
    cx = random.randint(80, w - 80)
    cy = random.randint(80, h - 80)
    ax1 = random.randint(30, 80)
    ax2 = random.randint(20, 50)
    angle = random.randint(0, 180)

    overlay = img.copy()
    val = random.choice([40, 210])
    cv2.ellipse(overlay, (cx, cy), (ax1, ax2), angle, 0, 360, (val, val, val), -1)
    alpha = random.uniform(0.35, 0.6)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    box_w = int(2 * max(ax1, ax2) * 1.1)
    box_h = int(2 * max(ax1, ax2) * 1.1)
    min_x = max(0, cx - box_w // 2)
    max_x = min(w, cx + box_w // 2)
    min_y = max(0, cy - box_h // 2)
    max_y = min(h, cy + box_h // 2)
    return min_x, min_y, max_x - min_x, max_y - min_y


def stamp_pitted_surface(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws cluster of cavitation / corrosion pit craters."""
    h, w = img.shape[:2]
    cx = random.randint(60, w - 60)
    cy = random.randint(60, h - 60)
    cluster_radius = random.randint(25, 55)

    pts = []
    for _ in range(random.randint(8, 20)):
        px = int(np.clip(cx + random.randint(-cluster_radius, cluster_radius), 5, w - 5))
        py = int(np.clip(cy + random.randint(-cluster_radius, cluster_radius), 5, h - 5))
        r = random.randint(2, 5)
        pts.append((px, py))
        # Pitted crater has dark pit + slight shadow/highlight bevel
        cv2.circle(img, (px, py), r, (30, 30, 30), -1)
        cv2.circle(img, (px + 1, py + 1), max(1, r - 1), (190, 190, 190), 1)

    all_x = [p[0] for p in pts]
    all_y = [p[1] for p in pts]
    min_x = max(0, min(all_x) - 6)
    max_x = min(w, max(all_x) + 6)
    min_y = max(0, min(all_y) - 6)
    max_y = min(h, max(all_y) + 6)
    return min_x, min_y, max_x - min_x, max_y - min_y


def stamp_rolled_in_scale(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Draws rolled-in metallic mill scale bands."""
    h, w = img.shape[:2]
    x = random.randint(30, w - 120)
    y = random.randint(30, h - 50)
    bw = random.randint(70, 160)
    bh = random.randint(15, 45)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (45, 45, 45), -1)
    # Add horizontal texture inside scale
    for sy in range(y, y + bh, 3):
        cv2.line(overlay, (x, sy), (x + bw, sy), (20, 20, 20), 1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return x, y, bw, bh


STAMP_FUNCS = {
    0: stamp_crazing,
    1: stamp_inclusion,
    2: stamp_patches,
    3: stamp_pitted_surface,
    4: stamp_rolled_in_scale,
    5: stamp_scratch,
}


def generate_sample(width: int = 640, height: int = 640) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Generates an image with 0 to 3 stamped defects and returns YOLO format annotations."""
    img = create_metallic_background(width, height)
    boxes = []

    # 15% clean samples, 85% defective samples
    if random.random() < 0.15:
        return img, boxes

    num_defects = random.randint(1, 3)
    for _ in range(num_defects):
        cls_id = random.randint(0, 5)
        stamp_fn = STAMP_FUNCS[cls_id]
        bx, by, bw, bh = stamp_fn(img)

        # Normalize to YOLO format: x_center, y_center, w, h in [0, 1]
        xc = (bx + bw / 2.0) / width
        yc = (by + bh / 2.0) / height
        norm_w = bw / float(width)
        norm_h = bh / float(height)

        # Safeguard bounds
        xc = float(np.clip(xc, 0.001, 0.999))
        yc = float(np.clip(yc, 0.001, 0.999))
        norm_w = float(np.clip(norm_w, 0.005, 0.999))
        norm_h = float(np.clip(norm_h, 0.005, 0.999))

        boxes.append((cls_id, xc, yc, norm_w, norm_h))

    return img, boxes


def generate_dataset(
    output_root: str | Path,
    num_train: int = 240,
    num_val: int = 50,
    num_test: int = 50,
    img_size: int = 640,
    seed: int = 42,
) -> None:
    """Creates a structured YOLO dataset split into train/val/test."""
    random.seed(seed)
    np.random.seed(seed)
    root = Path(output_root)

    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }

    for split_name, count in splits.items():
        img_dir = root / "images" / split_name
        lbl_dir = root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, boxes = generate_sample(img_size, img_size)
            stem = f"sample_{split_name}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)

            lbl_path = lbl_dir / f"{stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                for cls_id, xc, yc, bw, bh in boxes:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Generated dataset at {root}: train={num_train}, val={num_val}, test={num_test}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the synthetic CI dataset (replaces the old
    synthetic-dataset entrypoint removed in review item 5). Defaults match
    the previous entrypoint exactly: 240/50/50 images at data/processed,
    seed 42."""
    parser = argparse.ArgumentParser(
        description="Generate the synthetic industrial defect dataset (CI/tests)"
    )
    parser.add_argument(
        "--out", default="data/processed", help="dataset output root"
    )
    args = parser.parse_args(argv)
    generate_dataset(args.out)


if __name__ == "__main__":
    main()
