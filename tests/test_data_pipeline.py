"""Unit tests for dataset generation, augmentation, and validation."""

import pytest
import numpy as np
from src.data.generator import generate_sample, create_metallic_background, stamp_scratch
from src.data.validator import DatasetValidator, ValidationReport


def test_metallic_background():
    bg = create_metallic_background(640, 640)
    assert bg.shape == (640, 640, 3)
    assert bg.dtype == np.uint8
    assert bg.mean() > 50  # Valid metallic luminance range


def test_scratch_bounding_box():
    bg = create_metallic_background(640, 640)
    bx, by, bw, bh = stamp_scratch(bg)
    assert 0 <= bx < 640
    assert 0 <= by < 640
    assert 0 < bw <= 640
    assert 0 < bh <= 640


def test_generate_sample_bounds():
    img, boxes = generate_sample(640, 640)
    assert img.shape == (640, 640, 3)
    for cls_id, xc, yc, bw, bh in boxes:
        assert 0 <= cls_id <= 5
        assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc}"
        assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc}"
        assert 0.0 < bw <= 1.0, f"bw out of bounds: {bw}"
        assert 0.0 < bh <= 1.0, f"bh out of bounds: {bh}"


def test_validator_clean_report(tmp_path):
    """Validator on a freshly generated dataset: unconditional assertions
    against real files in tmp_path (no vacuous skip when data is absent)."""
    from src.data.generator import generate_dataset

    generate_dataset(tmp_path, num_train=10, num_val=3, num_test=3)

    validator = DatasetValidator(tmp_path)
    report = validator.validate_split("train")

    assert report.total_images == 10
    assert report.total_labels == 10
    assert len(report.corrupted_images) == 0
    assert len(report.out_of_bounds_boxes) == 0
    assert len(report.orphaned_labels) == 0
    assert report.images_with_defects + report.clean_images == report.total_images
