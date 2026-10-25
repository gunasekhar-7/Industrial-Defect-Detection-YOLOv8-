"""Model evaluation and per-class diagnostic metrics reporting."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_split_images(data_yaml: str | Path, split: str) -> int:
    """Count the images of a split by reading the dataset YAML's `path:`
    and `split:` entries without pulling in a YAML library (the file uses
    flat `key: value` lines, which is all we need here).

    Returns 0 when an entry is missing or the directory is absent, so an
    unusual layout degrades to 'unknown' instead of failing eval.
    """
    root: Path | None = None
    value: str | None = None
    try:
        lines = Path(data_yaml).read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("path:") and root is None:
            root_value = stripped.split(":", 1)[1].split("#")[0].strip().strip("'\"")
            if root_value:
                root = Path(root_value)
        elif stripped.startswith(f"{split}:") and value is None:
            value = stripped.split(":", 1)[1].split("#")[0].strip().strip("'\"")

    if not value:
        return 0
    directory = Path(value)
    if not directory.is_absolute() and root is not None:
        directory = root / directory
    if not directory.is_dir():
        return 0
    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def read_training_args(weights_path: str | Path) -> Dict[str, Any]:
    """Read the executed training configuration from the run's args.yaml
    (written by Ultralytics next to the weights) without a YAML library —
    flat `key: value` lines only.

    Returned next to the metrics so the report states exactly what was
    trained (epochs, input size, batch, device), not just what was
    measured. Missing file yields an empty dict instead of failing.
    """
    wanted = ("model", "epochs", "batch", "imgsz", "optimizer", "device")
    args_file = Path(weights_path).parent.parent / "args.yaml"
    try:
        lines = args_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found: Dict[str, Any] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in wanted or key in found:
            continue
        value = value.strip()
        if value in {"null", "None", ""}:
            found[key] = None
        else:
            try:
                found[key] = int(value)
            except ValueError:
                found[key] = value
    return found


def evaluate_model(
    weights_path: str | Path,
    data_yaml: str | Path = "configs/dataset.yaml",
    split: str = "val",
    imgsz: int = 640,
    device: str | None = None,
    report_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluates model performance and extracts per-class mAP metrics.

    The JSON report records the executed configuration (weights, data
    config, split, input size, image count, hardware, timestamp) next to
    the metrics, so every number quoted elsewhere can be traced back to
    the exact run that produced it.
    """
    from ultralytics import YOLO

    w_path = Path(weights_path).resolve()
    if not w_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {w_path}")

    model = YOLO(str(w_path))
    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        device=device if device else ("0" if model.device.type == "cuda" else "cpu"),
        plots=True,
    )

    # Extract aggregate metrics
    cpu = platform.processor() or platform.machine() or "unknown cpu"
    summary: Dict[str, Any] = {
        "model": w_path.name,
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "images": count_split_images(data_yaml, split),
        "hardware": f"{platform.platform()} | {cpu} | python {platform.python_version()}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "training": read_training_args(w_path),
        "per_class": {},
    }

    # Per-class breakdown
    class_names = metrics.names
    for cls_idx, cls_name in class_names.items():
        if cls_idx < len(metrics.box.maps):
            summary["per_class"][cls_name] = {
                "mAP50_95": float(metrics.box.maps[cls_idx]),
            }

    print("\n" + "=" * 65)
    print(" INDUSTRIAL DEFECT DETECTION EVALUATION REPORT")
    print("=" * 65)
    print(f" Overall Precision : {summary['precision']:.4f}")
    print(f" Overall Recall    : {summary['recall']:.4f}")
    print(f" Overall mAP@50    : {summary['mAP50']:.4f}")
    print(f" Overall mAP@50-95 : {summary['mAP50_95']:.4f}")
    print("-" * 65)
    print(f" {'Class Name':<20} | {'mAP@50-95':<15}")
    print("-" * 65)
    for cname, cmetrics in summary["per_class"].items():
        print(f" {cname:<20} | {cmetrics['mAP50_95']:<15.4f}")
    print("=" * 65 + "\n")

    # Save summary report to JSON
    report_file = Path(report_path) if report_path else w_path.parent / "eval_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Eval report written to: {report_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Defect Detection Model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to data YAML")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate ('val' or 'test')")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Report destination (default: next to the weights file)",
    )
    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        report_path=args.out,
    )
