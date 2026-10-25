"""Inference engine and micro-benchmarking harness for ONNX Runtime vs PyTorch."""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import onnxruntime as ort


class ONNXDefectDetector:
    """Production ONNX Runtime inference engine for industrial defect detection."""

    def __init__(
        self,
        onnx_model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: List[str] | None = None,
    ):
        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold

        if providers is None:
            available = ort.get_available_providers()
            # Prefer CUDA/DirectML if available, else CPU
            preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [batch, 3, H, W]
        self.img_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.img_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize and normalize image for YOLOv8 input."""
        orig_h, orig_w = img_bgr.shape[:2]
        r = min(self.img_h / orig_h, self.img_w / orig_w)
        new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))
        dw = (self.img_w - new_unpad[0]) / 2.0
        dh = (self.img_h - new_unpad[1]) / 2.0

        if (orig_w, orig_h) != new_unpad:
            resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_bgr

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert BGR to RGB, normalize [0, 1], transpose HWC -> CHW
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)  # Shape: (1, 3, H, W)
        return blob, r, (int(round(dw)), int(round(dh)))

    def postprocess(
        self,
        outputs: np.ndarray,
        ratio: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Parses YOLOv8 output tensor and applies Non-Maximum Suppression (NMS)."""
        # YOLOv8 output shape: [1, num_classes + 4, num_anchors] -> transpose to [num_anchors, num_classes + 4]
        preds = np.squeeze(outputs, axis=0).T
        orig_h, orig_w = orig_shape
        dw, dh = pad

        boxes_list = []
        confidences = []
        class_ids = []

        num_classes = preds.shape[1] - 4
        for row in preds:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_thresh:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                # Unpad and scale back to original dimensions
                x1 = int(np.clip((xc - w / 2.0 - dw) / ratio, 0, orig_w))
                y1 = int(np.clip((yc - h / 2.0 - dh) / ratio, 0, orig_h))
                bw = int(np.clip(w / ratio, 1, orig_w - x1))
                bh = int(np.clip(h / ratio, 1, orig_h - y1))

                boxes_list.append([x1, y1, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes_list:
            return []

        # Per-class NMS via the class-offset trick: shifting x by
        # class_id * 4096 makes cross-class IoU exactly 0 (max box width
        # 640 << 4096), so only same-class boxes can suppress each other —
        # matching Ultralytics' default class-aware behavior.
        offset_boxes = [
            [x + class_id * 4096, y, w, h]
            for (x, y, w, h), class_id in zip(boxes_list, class_ids)
        ]
        indices = cv2.dnn.NMSBoxes(offset_boxes, confidences, self.conf_thresh, self.iou_thresh)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes_list[idx]
                detections.append({
                    "class_id": int(class_ids[idx]),
                    "confidence": round(float(confidences[idx]), 4),
                    "box": {
                        "x_min": x,
                        "y_min": y,
                        "x_max": min(orig_w, x + w),
                        "y_max": min(orig_h, y + h),
                    },
                })

        return detections

    def predict(self, img_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Runs full inference pipeline and returns detections with latency breakdown in ms."""
        t0 = time.perf_counter()
        blob, ratio, pad = self.preprocess(img_bgr)
        t1 = time.perf_counter()

        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        t2 = time.perf_counter()

        detections = self.postprocess(outputs, ratio, pad, img_bgr.shape[:2])
        t3 = time.perf_counter()

        latencies = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        }
        return detections, latencies


def benchmark_model(
    onnx_path: str | Path,
    img_size: int = 640,
    warmup_runs: int = 50,
    benchmark_runs: int = 200,
) -> Dict[str, float]:
    """Micro-benchmarks ONNX model measuring mean, median, p95, and p99 latency in ms."""
    detector = ONNXDefectDetector(onnx_path)
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    print(f"Warming up ({warmup_runs} iterations)...")
    for _ in range(warmup_runs):
        detector.predict(dummy_img)

    print(f"Benchmarking ({benchmark_runs} iterations)...")
    inference_times = []
    total_times = []
    for _ in range(benchmark_runs):
        _, latencies = detector.predict(dummy_img)
        inference_times.append(latencies["inference_ms"])
        total_times.append(latencies["total_ms"])

    stats = {
        "mean_inference_ms": float(np.mean(inference_times)),
        "median_inference_ms": float(np.median(inference_times)),
        "p95_inference_ms": float(np.percentile(inference_times, 95)),
        "p99_inference_ms": float(np.percentile(inference_times, 99)),
        "mean_e2e_ms": float(np.mean(total_times)),
        "throughput_fps": float(1000.0 / np.mean(total_times)),
    }

    print("\n" + "=" * 65)
    print(f" ONNX RUNTIME LATENCY BENCHMARK: {Path(onnx_path).name}")
    print("=" * 65)
    print(f" Pure Inference Median Latency : {stats['median_inference_ms']:.2f} ms")
    print(f" Pure Inference Mean Latency   : {stats['mean_inference_ms']:.2f} ms")
    print(f" 95th Percentile (p95) Latency : {stats['p95_inference_ms']:.2f} ms")
    print(f" 99th Percentile (p99) Latency : {stats['p99_inference_ms']:.2f} ms")
    print(f" End-to-End Latency (w/ NMS)   : {stats['mean_e2e_ms']:.2f} ms")
    print(f" Throughput (FPS)              : {stats['throughput_fps']:.1f} FPS")
    print("=" * 65 + "\n")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ONNX Model Latency")
    parser.add_argument("--onnx", type=str, required=True, help="Path to .onnx file")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()

    benchmark_model(args.onnx, warmup_runs=args.warmup, benchmark_runs=args.runs)
