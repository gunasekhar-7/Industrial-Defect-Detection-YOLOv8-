"""Unit tests for ONNX preprocessing, postprocessing, and inference routines."""

import json

import numpy as np

from src.models.benchmark import ONNXDefectDetector, benchmark_model, build_benchmark_report


class MockDetector(ONNXDefectDetector):
    """Subclass that skips loading an actual ONNX file for unit testing pure math functions."""
    def __init__(self):
        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.img_h = 640
        self.img_w = 640


def test_preprocessing_output_format():
    detector = MockDetector()
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob, ratio, pad = detector.preprocess(dummy_img)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0
    assert isinstance(ratio, float)
    assert len(pad) == 2


def test_nms_is_per_class():
    """Overlapping boxes of DIFFERENT classes must both survive NMS;
    class-agnostic NMS would silently drop one defect type that sits on
    another — a false-pass risk for inspection."""
    detector = MockDetector()
    preds = np.zeros((1, 10, 2), dtype=np.float32)

    # Anchor 0: center (320,320), 100x100, class 0 score 0.90
    preds[0, 0, 0] = 320.0
    preds[0, 1, 0] = 320.0
    preds[0, 2, 0] = 100.0
    preds[0, 3, 0] = 100.0
    preds[0, 4, 0] = 0.90

    # Anchor 1: center (325,325), 100x100, class 1 score 0.80
    # IoU ~0.82 (> 0.45) but a DIFFERENT class -> must NOT be suppressed.
    preds[0, 0, 1] = 325.0
    preds[0, 1, 1] = 325.0
    preds[0, 2, 1] = 100.0
    preds[0, 3, 1] = 100.0
    preds[0, 5, 1] = 0.80

    detections = detector.postprocess(preds, ratio=1.0, pad=(0, 0), orig_shape=(640, 640))
    classes = sorted(d["class_id"] for d in detections)
    assert classes == [0, 1], f"expected both classes to survive, got {classes}"


def test_benchmark_report_has_required_fields(tmp_path):
    """build_benchmark_report must carry every field the README cites."""
    stats = {
        "mean_inference_ms": 1.0,
        "median_inference_ms": 0.9,
        "p95_inference_ms": 2.0,
        "p99_inference_ms": 3.0,
        "mean_e2e_ms": 4.0,
        "throughput_fps": 250.0,
    }
    report = build_benchmark_report(
        stats=stats,
        onnx_path="models/industrial_yolov8n.onnx",
        provider="CPUExecutionProvider",
        warmup_runs=50,
        benchmark_runs=200,
        model_input=[1, 3, 320, 320],
    )
    for key in (
        "hardware",
        "provider",
        "model",
        "model_input",
        "warmup_runs",
        "benchmark_runs",
        "timestamp",
        "mean_inference_ms",
        "median_inference_ms",
        "p95_inference_ms",
        "p99_inference_ms",
        "mean_e2e_ms",
        "throughput_fps",
    ):
        assert key in report, f"missing report field: {key}"
    assert report["warmup_runs"] == 50
    assert report["benchmark_runs"] == 200
    assert report["provider"] == "CPUExecutionProvider"
    assert report["model"] == "industrial_yolov8n.onnx"


def test_benchmark_cli_json_output(tmp_path):
    """--json path must be written by benchmark_model and parse as JSON."""
    out = tmp_path / "benchmark.json"
    benchmark_model(
        "models/industrial_yolov8n.onnx",
        warmup_runs=1,
        benchmark_runs=2,
        json_path=out,
    )
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["warmup_runs"] == 1
    assert data["benchmark_runs"] == 2
    assert data["p95_inference_ms"] > 0
    assert data["throughput_fps"] > 0


def test_postprocessing_bounding_boxes():
    detector = MockDetector()
    # Mock YOLOv8 output: [1, 10, 3] (3 anchor predictions: 4 bbox coords + 6 class probabilities)
    # Class 5 (scratch) with high confidence 0.95
    preds = np.zeros((1, 10, 2), dtype=np.float32)

    # Box 1: Center (320, 320), width 100, height 50, class 5 conf 0.95
    preds[0, 0, 0] = 320.0
    preds[0, 1, 0] = 320.0
    preds[0, 2, 0] = 100.0
    preds[0, 3, 0] = 50.0
    preds[0, 9, 0] = 0.95  # 4 + class_5 = 9

    # Box 2: Duplicate overlapping box with slightly lower conf (0.85) to test NMS suppression
    preds[0, 0, 1] = 322.0
    preds[0, 1, 1] = 321.0
    preds[0, 2, 1] = 98.0
    preds[0, 3, 1] = 49.0
    preds[0, 9, 1] = 0.85

    detections = detector.postprocess(preds, ratio=1.0, pad=(0, 0), orig_shape=(640, 640))
    # NMS should suppress the duplicate box, retaining only 1 top detection
    assert len(detections) == 1
    det = detections[0]
    assert det["class_id"] == 5
    assert det["confidence"] == 0.95
    assert det["box"]["x_min"] == 270
    assert det["box"]["x_max"] == 370
