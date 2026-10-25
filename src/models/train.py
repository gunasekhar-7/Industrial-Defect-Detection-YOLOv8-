"""Model training pipeline for YOLOv8 industrial defect detection."""

import os
from pathlib import Path
from typing import Any
import torch


def train_model(
    model_name: str = "yolov8n.pt",
    data_yaml: str | Path = "configs/dataset.yaml",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "",
    name: str = "industrial_defect_yolov8n",
    device: str | None = None,
) -> Any:
    """Trains a YOLOv8 model on the industrial defect dataset.

    Ultralytics resolves the save directory as ``runs/<task>/<project>/<name>``
    (see ``ultralytics.cfg.get_save_dir``), so ``project=""`` yields the
    canonical ``runs/detect/<name>`` layout that evaluate/export defaults use.
    """
    from ultralytics import YOLO

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    # Determine compute device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device} | Model: {model_name} | Epochs: {epochs}")

    # MLflow experiment tracking through Ultralytics' native callback: it
    # registers whenever `mlflow` is importable and starts a run at training
    # start, logging params, per-epoch metrics, and weight artifacts to the
    # local tracking store. MLflow 3.x defaults to a local database backend
    # (the old ./mlruns file store is in maintenance mode and refuses writes
    # without an explicit opt-in), so only the experiment name is pinned
    # here; the store URI stays at mlflow's default and both candidate
    # locations (mlflow.db, mlruns/) are gitignored.
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "industrial-defect")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"Training completed. Weights saved under {project}/{name}/weights/best.pt")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on Industrial Defect Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', '0', etc.)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
