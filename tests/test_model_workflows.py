"""Unit coverage for model workflows using lightweight Ultralytics fakes."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models import evaluate, export, train


def _fake_ultralytics(monkeypatch, model_class):
    module = types.ModuleType("ultralytics")
    module.YOLO = model_class
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_train_model_selects_cpu_and_forwards_options(tmp_path, monkeypatch):
    data = tmp_path / "dataset.yaml"
    data.write_text("path: .\n", encoding="utf-8")
    calls = {}

    class FakeModel:
        def __init__(self, name):
            calls["model_name"] = name

        def train(self, **kwargs):
            calls["train"] = kwargs
            return "trained"

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    _fake_ultralytics(monkeypatch, FakeModel)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    result = train.train_model(
        model_name="base.pt", data_yaml=data, epochs=3, imgsz=320,
        batch=4, name="fixture", device=None,
    )

    assert result == "trained"
    assert calls["model_name"] == "base.pt"
    assert calls["train"] == {
        "data": str(data.resolve()), "epochs": 3, "imgsz": 320,
        "batch": 4, "project": "", "name": "fixture", "device": "cpu",
        "save": True, "plots": True, "verbose": True,
    }
    assert "MLFLOW_EXPERIMENT_NAME" in __import__("os").environ


def test_train_model_selects_cuda_and_rejects_missing_dataset(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name):
            calls.append(name)

        def train(self, **kwargs):
            calls.append(kwargs)

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    _fake_ultralytics(monkeypatch, FakeModel)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(FileNotFoundError, match="Dataset config not found"):
        train.train_model(data_yaml=tmp_path / "missing.yaml")
    assert calls == []

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\n", encoding="utf-8")
    train.train_model(data_yaml=dataset, device=None)
    assert calls[1]["device"] == "0"


def test_evaluate_model_writes_provenance_report(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"fake weights")
    run = weights.parent.parent
    (run / "args.yaml").write_text("epochs: 8\nimgsz: 320\n", encoding="utf-8")
    data = tmp_path / "dataset.yaml"
    images = tmp_path / "images" / "val"
    images.mkdir(parents=True)
    (images / "one.jpg").write_bytes(b"image")
    data.write_text(f"path: {tmp_path.as_posix()}\nval: images/val\n", encoding="utf-8")
    calls = {}

    box = SimpleNamespace(map50=0.7, map=0.4, mp=0.6, mr=0.5, maps=[0.3])
    metrics = SimpleNamespace(box=box, names={0: "defect", 1: "unmeasured"})

    class FakeModel:
        device = SimpleNamespace(type="cpu")

        def __init__(self, path):
            calls["weights"] = path

        def val(self, **kwargs):
            calls["val"] = kwargs
            return metrics

    _fake_ultralytics(monkeypatch, FakeModel)
    monkeypatch.setattr(evaluate.platform, "processor", lambda: "test-cpu")
    report_file = tmp_path / "nested" / "eval.json"

    report = evaluate.evaluate_model(weights, data, "val", 320, report_path=report_file)

    assert report["model"] == "best.pt"
    assert report["images"] == 1
    assert report["training"] == {"epochs": 8, "imgsz": 320}
    assert report["per_class"] == {"defect": {"mAP50_95": 0.3}}
    assert calls["val"]["device"] == "cpu"
    assert json.loads(report_file.read_text(encoding="utf-8")) == report
    assert report["hardware"].find("test-cpu") >= 0


def test_evaluate_model_defaults_report_location_and_cuda_device(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    calls = {}
    metrics = SimpleNamespace(
        box=SimpleNamespace(map50=0.1, map=0.2, mp=0.3, mr=0.4, maps=[]),
        names={},
    )

    class FakeModel:
        device = SimpleNamespace(type="cuda")

        def __init__(self, _path):
            pass

        def val(self, **kwargs):
            calls.update(kwargs)
            return metrics

    _fake_ultralytics(monkeypatch, FakeModel)
    monkeypatch.setattr(evaluate.platform, "processor", lambda: "")
    monkeypatch.setattr(evaluate.platform, "machine", lambda: "fallback-cpu")
    report = evaluate.evaluate_model(weights, data_yaml=tmp_path / "missing.yaml")

    assert calls["device"] == "0"
    assert report["hardware"].find("fallback-cpu") >= 0
    assert (weights.parent / "eval_report.json").is_file()


def test_evaluate_rejects_missing_weights(tmp_path, monkeypatch):
    _fake_ultralytics(monkeypatch, lambda _path: None)
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        evaluate.evaluate_model(tmp_path / "missing.pt")


def test_export_to_onnx_forwards_options_and_checks_graph(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.touch()
    output = tmp_path / "best.onnx"
    calls = {}

    class FakeModel:
        def __init__(self, path):
            calls["weights"] = path

        def export(self, **kwargs):
            calls["export"] = kwargs
            return str(output)

    _fake_ultralytics(monkeypatch, FakeModel)
    fake_graph = object()

    def load_graph(path):
        calls["loaded"] = path
        return fake_graph

    def check_graph(model):
        calls["checked"] = model

    monkeypatch.setattr(export.onnx, "load", load_graph)
    monkeypatch.setattr(export.onnx.checker, "check_model", check_graph)

    result = export.export_to_onnx(
        weights, imgsz=320, opset=17, half=True, dynamic=True, simplify=False
    )

    assert result == output
    assert calls["weights"] == str(weights.resolve())
    assert calls["export"] == {
        "format": "onnx", "imgsz": 320, "opset": 17,
        "half": True, "dynamic": True, "simplify": False,
    }
    assert calls["loaded"] == str(output)
    assert calls["checked"] is fake_graph


def test_export_rejects_missing_weights(tmp_path, monkeypatch):
    _fake_ultralytics(monkeypatch, lambda _path: None)
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        export.export_to_onnx(tmp_path / "missing.pt")


def test_quantize_to_int8_writes_dynamic_weight_only_graph(tmp_path, monkeypatch):
    """Dynamic quantization needs no calibration data: weights become QInt8
    and the output lands beside the FP32 graph by default."""
    import onnxruntime.quantization as ortq

    from src.models import quantize

    source = tmp_path / "detector.onnx"
    source.write_bytes(b"fp32 graph")
    calls = {}

    def fake_quantize(model_input, model_output, weight_type, per_channel):
        calls.update(
            model_input=model_input,
            model_output=model_output,
            weight_type=weight_type,
            per_channel=per_channel,
        )
        Path(model_output).write_bytes(b"int8 graph")

    monkeypatch.setattr(ortq, "quantize_dynamic", fake_quantize)

    target = quantize.quantize_to_int8(source)

    assert target == tmp_path / "detector.int8.onnx"
    assert target.read_bytes() == b"int8 graph"
    assert calls["model_input"] == str(source)
    assert calls["weight_type"] == ortq.QuantType.QInt8
    assert calls["per_channel"] is False


def test_quantize_rejects_missing_source_and_in_place_output(tmp_path, monkeypatch):
    import onnxruntime.quantization as ortq

    from src.models import quantize

    monkeypatch.setattr(ortq, "quantize_dynamic", lambda **_kwargs: pytest.fail("must not run"))
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        quantize.quantize_to_int8(tmp_path / "missing.onnx")

    source = tmp_path / "detector.onnx"
    source.write_bytes(b"fp32 graph")
    with pytest.raises(ValueError, match="output must differ"):
        quantize.quantize_to_int8(source, source)


def test_benchmark_rejects_missing_model(tmp_path):
    """Benchmarking a path that is not there fails immediately with the path,
    rather than surfacing an opaque onnxruntime error."""
    from src.models import benchmark

    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        benchmark.ONNXDefectDetector(tmp_path / "missing.onnx")


def _write_frames(folder: Path, count: int = 2) -> list[Path]:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = folder / f"frame_{i}.jpg"
        Image.new("RGB", (32, 32), (i * 10, 0, 0)).save(path, format="JPEG")
        paths.append(path)
    return paths


def test_compare_int8_tradeoff_measures_both_models_on_the_same_frames(tmp_path, monkeypatch):
    """Both graphs run back-to-back over identical frames, so size, latency and
    prediction agreement describe one machine at one moment."""
    from src.models import benchmark, quantize

    fp32 = tmp_path / "detector.onnx"
    int8 = tmp_path / "detector.int8.onnx"
    fp32.write_bytes(b"x" * 400_000)
    int8.write_bytes(b"x" * 100_000)
    frames = _write_frames(tmp_path / "frames")

    class FakeDetector:
        def __init__(self, path):
            self.total_ms = 20.0 if Path(path).name == "detector.onnx" else 10.0

        def predict(self, _img):
            detection = {"class_id": 2, "confidence": 0.5, "box": {}}
            return [detection], {"inference_ms": self.total_ms / 2, "total_ms": self.total_ms}

    monkeypatch.setattr(benchmark, "ONNXDefectDetector", FakeDetector)

    report = quantize.compare_int8_tradeoff(fp32, int8, frames, warmup_runs=1, benchmark_runs=4)

    assert report["frames"] == 2
    assert report["models"]["fp32"] == {
        "file": "detector.onnx",
        "size_mb": 0.4,
        "mean_ms": 20.0,
        "median_ms": 20.0,
        "p95_ms": 20.0,
    }
    assert report["models"]["int8"]["median_ms"] == 10.0
    assert report["int8_speedup_median"] == 2.0
    assert report["detection_agreement"] == {"frames_compared": 2, "identical_classes": 2}


def test_compare_int8_tradeoff_skips_undecodable_frames(tmp_path, monkeypatch):
    from src.models import benchmark, quantize

    class FakeDetector:
        def __init__(self, _path):
            pass

        def predict(self, _img):
            return [], {"inference_ms": 1.0, "total_ms": 2.0}

    monkeypatch.setattr(benchmark, "ONNXDefectDetector", FakeDetector)
    good = _write_frames(tmp_path / "frames", count=1)[0]
    broken = tmp_path / "frames" / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"x")

    report = quantize.compare_int8_tradeoff(model, model, [broken, good], benchmark_runs=2)

    assert report["frames"] == 1


def test_compare_int8_tradeoff_requires_readable_frames(tmp_path):
    from src.models import quantize

    broken = tmp_path / "frame_0.jpg"
    broken.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError, match="no readable frames"):
        quantize.compare_int8_tradeoff(tmp_path / "a.onnx", tmp_path / "b.onnx", [broken])


def test_read_training_args_ignores_noise_and_keeps_first_values(tmp_path):
    """args.yaml is a flat dump: comment lines and non-pairs are skipped, the
    first value wins, and types are inferred (null, int, string)."""
    weights = tmp_path / "runs" / "exp" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    (weights.parent.parent / "args.yaml").write_text(
        "# run summary\n"
        "epochs: 8\n"
        "not a key value pair\n"
        "epochs: 99\n"
        "imgsz: 320\n"
        "optimizer: None\n"
        "model: 1\n"
        "batch: auto\n"
        "lr0: 0.01\n",
        encoding="utf-8",
    )

    assert evaluate.read_training_args(weights) == {
        "epochs": 8,
        "imgsz": 320,
        "optimizer": None,
        "model": 1,
        "batch": "auto",
    }
