"""Provenance parsers: the image count and executed training args in the report.

These two functions are what make `reports/eval_report.json` traceable, so
they are tested on their own without importing ultralytics.
"""

from pathlib import Path

from src.models.evaluate import count_split_images, read_training_args


def test_count_split_images_resolves_relative_split(tmp_path: Path):
    data = tmp_path / "data"
    val = data / "images" / "val"
    val.mkdir(parents=True)
    for name in ("a.jpg", "b.png", "c.JPEG"):
        (val / name).write_bytes(b"x")
    (val / "notes.txt").write_text("not an image", encoding="utf-8")

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        f"# comment\npath: {data.as_posix()}\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )

    assert count_split_images(dataset, "val") == 3


def test_count_split_images_degrades_to_zero(tmp_path: Path):
    """A missing split directory or unreadable YAML reports 0, never raises."""
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\nval: images/absent\n", encoding="utf-8")

    assert count_split_images(dataset, "val") == 0
    assert count_split_images(dataset, "test") == 0
    assert count_split_images(tmp_path / "missing.yaml", "val") == 0


def test_read_training_args_parses_executed_config(tmp_path: Path):
    weights = tmp_path / "runs" / "detect" / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    (weights.parent.parent / "args.yaml").write_text(
        "model: yolov8n.pt\n"
        "data: configs/dataset.yaml\n"
        "epochs: 50\n"
        "batch: 16\n"
        "imgsz: 320\n"
        "optimizer: auto\n"
        "device: cpu\n"
        "plots: true\n",
        encoding="utf-8",
    )

    assert read_training_args(weights) == {
        "model": "yolov8n.pt",
        "epochs": 50,
        "batch": 16,
        "imgsz": 320,
        "optimizer": "auto",
        "device": "cpu",
    }


def test_read_training_args_handles_null_and_missing_file(tmp_path: Path):
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    (weights.parent.parent / "args.yaml").write_text("device: null\nepochs: 50\n", encoding="utf-8")

    assert read_training_args(weights) == {"device": None, "epochs": 50}
    assert read_training_args(tmp_path / "other" / "weights" / "best.pt") == {}
