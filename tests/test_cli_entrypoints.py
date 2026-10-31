"""CLI entry points: each ``python -m`` guard must reach its ``main()``.

These run the real module files as ``__main__`` with ``sys.argv`` set, so
the argument parsers, the wiring into the library functions, and the
``if __name__ == "__main__"`` guards are all executed rather than assumed.
"""

import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_neu_det import make_fixture

ROOT = Path(__file__).resolve().parents[1]


def _run_module_as_main(monkeypatch, relative_path: str, *argv: str):
    monkeypatch.setattr(sys, "argv", [relative_path, *argv])
    return runpy.run_path(str(ROOT / relative_path), run_name="__main__")


def test_export_cli_entry_point_requires_weights(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        _run_module_as_main(
            monkeypatch,
            "src/models/export.py",
            "--weights",
            str(tmp_path / "missing.pt"),
            "--imgsz",
            "320",
            "--opset",
            "17",
            "--half",
            "--dynamic",
        )


def test_evaluate_cli_entry_point_writes_report_where_asked(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.touch()
    calls = {}
    metrics = SimpleNamespace(
        box=SimpleNamespace(map50=0.5, map=0.25, mp=0.4, mr=0.3, maps=[]),
        names={},
    )

    class FakeYOLO:
        device = SimpleNamespace(type="cpu")

        def __init__(self, path):
            calls["weights"] = path

        def val(self, **kwargs):
            calls["val"] = kwargs
            return metrics

    stub = types.ModuleType("ultralytics")
    stub.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    report = tmp_path / "reports" / "cli_eval.json"

    _run_module_as_main(
        monkeypatch,
        "src/models/evaluate.py",
        "--weights",
        str(weights),
        "--split",
        "test",
        "--imgsz",
        "320",
        "--out",
        str(report),
    )

    assert calls["val"]["split"] == "test"
    assert calls["val"]["imgsz"] == 320
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["split"] == "test"
    assert written["imgsz"] == 320


def test_benchmark_cli_entry_point_requires_a_model(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        _run_module_as_main(
            monkeypatch,
            "src/models/benchmark.py",
            "--onnx",
            str(tmp_path / "missing.onnx"),
            "--warmup",
            "1",
            "--runs",
            "2",
            "--json",
            str(tmp_path / "bench.json"),
        )


def test_train_cli_entry_point_requires_dataset(tmp_path, monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    stub = types.ModuleType("ultralytics")
    stub.YOLO = lambda _path: None
    monkeypatch.setitem(sys.modules, "ultralytics", stub)

    with pytest.raises(FileNotFoundError, match="Dataset config not found"):
        _run_module_as_main(
            monkeypatch,
            "src/models/train.py",
            "--data",
            str(tmp_path / "missing.yaml"),
            "--imgsz",
            "320",
            "--epochs",
            "1",
        )


def test_generator_cli_entry_point_builds_the_documented_dataset(tmp_path, monkeypatch):
    """`python -m src.data.generator` must produce the 240/50/50 layout that
    the CI pipeline trains on, at the requested output root."""
    out = tmp_path / "generated"

    _run_module_as_main(monkeypatch, "src/data/generator.py", "--out", str(out))

    counts = {
        split: len(list((out / "images" / split).glob("*.jpg")))
        for split in ("train", "val", "test")
    }
    assert counts == {"train": 240, "val": 50, "test": 50}
    for split in ("train", "val", "test"):
        labels = list((out / "labels" / split).glob("*.txt"))
        assert len(labels) == counts[split]
        # Clean frames legitimately get an empty label file, so require that
        # the split produced at least some defect annotations.
        assert any(p.read_text(encoding="utf-8").strip() for p in labels)


def test_quantize_cli_entry_point_writes_the_int8_graph(tmp_path, monkeypatch):
    import onnxruntime.quantization as ortq

    source = tmp_path / "detector.onnx"
    source.write_bytes(b"fp32 graph")
    target = tmp_path / "artifacts" / "detector.int8.onnx"

    def fake_quantize(**kwargs):
        Path(kwargs["model_output"]).write_bytes(b"int8 graph")
        return Path(kwargs["model_output"])

    monkeypatch.setattr(ortq, "quantize_dynamic", fake_quantize)

    _run_module_as_main(
        monkeypatch,
        "src/models/quantize.py",
        "--onnx",
        str(source),
        "--out",
        str(target),
        "--per-channel",
    )

    assert target.read_bytes() == b"int8 graph"


def test_quantize_cli_compares_fp32_and_int8_on_a_frame_directory(tmp_path, monkeypatch):
    import onnxruntime.quantization as ortq
    from PIL import Image

    from src.models import benchmark

    source = tmp_path / "detector.onnx"
    source.write_bytes(b"fp32 graph")
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (16, 16), (5, 5, 5)).save(frames / "frame_0.jpg", format="JPEG")
    report = tmp_path / "reports" / "tradeoff.json"

    def fake_quantize(**kwargs):
        Path(kwargs["model_output"]).write_bytes(b"int8 graph")

    class FakeDetector:
        def __init__(self, _path):
            pass

        def predict(self, _img):
            return [], {"inference_ms": 1.0, "total_ms": 2.0}

    monkeypatch.setattr(ortq, "quantize_dynamic", fake_quantize)
    monkeypatch.setattr(benchmark, "ONNXDefectDetector", FakeDetector)

    _run_module_as_main(
        monkeypatch,
        "src/models/quantize.py",
        "--onnx",
        str(source),
        "--compare",
        str(frames),
        "--report",
        str(report),
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert set(payload["models"]) == {"fp32", "int8"}
    assert payload["detection_agreement"] == {"frames_compared": 1, "identical_classes": 1}


def test_neu_det_cli_entry_point_converts_a_local_archive(tmp_path, monkeypatch):
    source = make_fixture(tmp_path / "source")
    out = tmp_path / "out"

    _run_module_as_main(
        monkeypatch,
        "src/data/neu_det.py",
        "--archive",
        str(source),
        "--out",
        str(out),
        "--seed",
        "5",
    )

    images = list((out / "images").rglob("*.jpg"))
    labels = list((out / "labels").rglob("*.txt"))
    assert len(images) == 6
    assert len(labels) == 6
    assert {p.parent.name for p in images} == {"train", "val", "test"}
