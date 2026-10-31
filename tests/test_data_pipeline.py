"""Unit tests for dataset generation, augmentation, and validation."""

import numpy as np
from PIL import Image

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


def test_validator_rejects_out_of_range_class_id(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "sample.jpg")
    (labels / "sample.txt").write_text("6 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    report = DatasetValidator(tmp_path).validate_split("train")

    assert not report.is_valid
    assert report.invalid_class_ids == [(str(labels / "sample.txt"), 0, 6)]


def test_validator_rejects_boxes_extending_outside_image(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "sample.jpg")
    (labels / "sample.txt").write_text("0 0.9 0.5 0.5 0.25\n", encoding="utf-8")

    report = DatasetValidator(tmp_path).validate_split("train")

    assert not report.is_valid
    assert len(report.out_of_bounds_boxes) == 1


def test_validator_allows_missing_labels_for_background_images(tmp_path):
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "clean.jpg")

    report = DatasetValidator(tmp_path).validate_split("train")

    assert report.is_valid
    assert report.clean_images == 1
    assert report.missing_labels == [str(images / "clean.jpg")]


def test_validator_rejects_empty_or_missing_split(tmp_path):
    validator = DatasetValidator(tmp_path)
    (tmp_path / "images" / "empty").mkdir(parents=True)

    assert not validator.validate_split("empty").is_valid
    assert not validator.validate_split("missing").is_valid


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


def test_validator_reports_orphaned_labels_and_corrupt_images(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "broken.jpg").write_bytes(b"not a jpeg")
    (labels / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = DatasetValidator(tmp_path).validate_split("train")

    assert not report.is_valid
    assert report.corrupted_images == [str(images / "broken.jpg")]
    assert report.orphaned_labels == [str(labels / "orphan.txt")]


def test_validator_reports_zero_sized_image(tmp_path, monkeypatch):
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    (images / "zero.jpg").write_bytes(b"placeholder")
    monkeypatch.setattr("src.data.validator.cv2.imread", lambda _path: np.zeros((0, 10, 3), dtype=np.uint8))

    report = DatasetValidator(tmp_path).validate_split("train")

    assert report.corrupted_images == [str(images / "zero.jpg")]
    assert not report.is_valid


def test_validator_reports_malformed_and_non_numeric_labels(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "badshape.jpg")
    Image.new("RGB", (32, 32)).save(images / "badnumber.jpg")
    (labels / "badshape.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    (labels / "badnumber.txt").write_text("x 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = DatasetValidator(tmp_path).validate_split("train")

    assert len(report.out_of_bounds_boxes) == 2
    assert all(entry[2] == [] for entry in report.out_of_bounds_boxes)
