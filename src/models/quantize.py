"""INT8 dynamic quantization and the measured FP32-vs-INT8 trade-off.

The FP32 graph is what the API serves; this module produces the smaller
INT8 variant and the evidence used to decide which one ships. Dynamic
(weight-only) quantization needs no calibration set, so the comparison can
be reproduced on any CPU in one command::

    python -m src.models.quantize --onnx models/industrial_yolov8n.onnx \
        --compare data/processed/images/val

The comparison runs both models back-to-back in one process, because
absolute latency drifts between sessions; only the ratio measured on the
same machine at the same moment is comparable.
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def quantize_to_int8(
    source_onnx: str | Path,
    output_onnx: str | Path | None = None,
    per_channel: bool = False,
) -> Path:
    """Quantize a YOLOv8 ONNX graph to INT8 weights with onnxruntime.

    Returns the written path. ``output_onnx`` defaults to
    ``<stem>.int8.onnx`` next to the source, which keeps the FP32 graph
    (what the API loads) and the quantized one side by side.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(source_onnx)
    if not source.exists():
        raise FileNotFoundError(f"ONNX model not found at: {source.resolve()}")

    target = (
        Path(output_onnx)
        if output_onnx is not None
        else source.with_name(f"{source.stem}.int8.onnx")
    )
    if target == source:
        raise ValueError("output must differ from the source model")

    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    print(f"Quantized {source.name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def _latency_summary(latencies: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(latencies, dtype=float)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def compare_int8_tradeoff(
    fp32_onnx: str | Path,
    int8_onnx: str | Path,
    frames: Iterable[str | Path],
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, Any]:
    """Measure FP32 and INT8 on the same frames, in one process.

    Records size, median/p95 end-to-end latency and how often the two
    models return the same detection classes on the sample frames — enough
    to decide whether quantization is actually a win on the target CPU
    rather than assuming it is.
    """
    import cv2

    from src.models.benchmark import ONNXDefectDetector

    images: List[np.ndarray] = []
    for path in frames:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    if not images:
        raise ValueError("no readable frames were supplied for the comparison")

    report: Dict[str, Any] = {
        "hardware": f"{platform.platform()} | {platform.processor() or platform.machine()}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames": len(images),
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "models": {},
    }

    predicted: Dict[str, List[Tuple[int, ...]]] = {}
    for label, model_path in (("fp32", fp32_onnx), ("int8", int8_onnx)):
        detector = ONNXDefectDetector(model_path)
        for i in range(warmup_runs):
            detector.predict(images[i % len(images)])

        latencies: List[float] = []
        classes: List[Tuple[int, ...]] = []
        for i in range(benchmark_runs):
            detections, timings = detector.predict(images[i % len(images)])
            latencies.append(timings["total_ms"])
            if i < len(images):
                classes.append(tuple(sorted(int(d["class_id"]) for d in detections)))

        predicted[label] = classes
        path = Path(model_path)
        report["models"][label] = {
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            **_latency_summary(latencies),
        }

    fp32_ms = report["models"]["fp32"]["median_ms"]
    int8_ms = report["models"]["int8"]["median_ms"]
    identical = sum(1 for a, b in zip(predicted["fp32"], predicted["int8"]) if a == b)
    report["detection_agreement"] = {
        "frames_compared": len(predicted["fp32"]),
        "identical_classes": identical,
    }
    # >1 means INT8 is faster; <1 means it lost (the common case for
    # convolution-heavy graphs on CPUs without VNNI instructions).
    report["int8_speedup_median"] = round(fp32_ms / int8_ms, 2) if int8_ms else 0.0
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m src.models.quantize --onnx models/detector.onnx``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize the YOLOv8 ONNX detector to INT8 and measure the trade-off"
    )
    parser.add_argument("--onnx", type=str, required=True, help="Path to the FP32 .onnx file")
    parser.add_argument(
        "--out", type=str, default=None, help="Output path (default: <stem>.int8.onnx)"
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Per-channel weight quantization (slower to prepare, often slightly more accurate)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="FRAMES_DIR",
        help="Benchmark FP32 vs INT8 on up to 24 .jpg frames from this directory",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/quantization_tradeoff.json",
        help="Where to write the comparison report",
    )
    args = parser.parse_args(argv)

    target = quantize_to_int8(args.onnx, args.out, per_channel=args.per_channel)
    if not args.compare:
        return

    frame_paths = sorted(Path(args.compare).glob("*.jpg"))[:24]
    report = compare_int8_tradeoff(args.onnx, target, frame_paths)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Trade-off report written to: {report_path}")


if __name__ == "__main__":
    main()
