"""Model evaluation and per-class diagnostic metrics reporting."""

import json
from pathlib import Path
from typing import Dict, Any


def evaluate_model(
    weights_path: str | Path,
    data_yaml: str | Path = "configs/dataset.yaml",
    split: str = "val",
    imgsz: int = 640,
    device: str | None = None,
) -> Dict[str, Any]:
    """Evaluates model performance and extracts per-class mAP metrics."""
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
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
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
    report_file = w_path.parent / "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Defect Detection Model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to data YAML")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate ('val' or 'test')")
    args = parser.parse_args()

    evaluate_model(weights_path=args.weights, data_yaml=args.data, split=args.split)
