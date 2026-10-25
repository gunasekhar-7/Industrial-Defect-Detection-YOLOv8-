"""Unit tests for dataset generation, augmentation, and validation."""

import numpy as np

from src.data.generator import create_metallic_background, generate_sample, stamp_scratch
from src.data.validator import DatasetValidator
from src.models.evaluate import count_split_images, read_training_args


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


def test_count_split_images_resolves_path_prefix(tmp_path):
    """Eval's image count reads `path:` + `split:` from the dataset YAML,
    counting image files only and reporting 0 for missing directories."""
    val_dir = tmp_path / "images" / "val"
    val_dir.mkdir(parents=True)
    for i in range(3):
        (val_dir / f"v{i}.jpg").write_bytes(b"x")
    (val_dir / "notes.txt").write_bytes(b"not an image")

    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\n# val: images/other\n",
        encoding="utf-8",
    )

    assert count_split_images(yaml_file, "val") == 3
    assert count_split_images(yaml_file, "train") == 0  # directory absent


def test_count_split_images_missing_entry(tmp_path):
    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text("path: data/processed\ntrain: images/train\n", encoding="utf-8")
    assert count_split_images(yaml_file, "test") == 0
    assert count_split_images(tmp_path / "nope.yaml", "val") == 0


def test_read_training_args_from_run_directory(tmp_path):
    """Executed training config is read from the run's args.yaml next to
    the weights; a missing file degrades to an empty dict."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir.parent / "args.yaml").write_text(
        "task: detect\nmodel: yolov8n.pt\nepochs: 50\nbatch: 16\n"
        "imgsz: 320\ndevice: cpu\noptimizer: auto\ntime: null\n",
        encoding="utf-8",
    )

    args = read_training_args(weights_dir / "best.pt")
    assert args["model"] == "yolov8n.pt"
    assert args["epochs"] == 50
    assert args["batch"] == 16
    assert args["imgsz"] == 320
    assert args["device"] == "cpu"
    assert "time" not in args  # only the fields we quote are read

    # Different run directory with no args.yaml next to its weights.
    assert read_training_args(tmp_path / "other_run" / "weights" / "best.pt") == {}
