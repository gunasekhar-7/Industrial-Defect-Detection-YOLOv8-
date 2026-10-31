"""ONNX export module with opset and half-precision support."""

from pathlib import Path

import onnx


def export_to_onnx(
    weights_path: str | Path,
    imgsz: int = 640,
    opset: int = 17,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Exports a trained YOLOv8 model to ONNX format and verifies graph integrity."""
    w_path = Path(weights_path).resolve()
    if not w_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {w_path}")

    from ultralytics import YOLO

    print(f"Exporting {w_path.name} to ONNX (opset={opset}, half={half}, dynamic={dynamic})...")
    model = YOLO(str(w_path))
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
    )

    onnx_path = Path(exported_file)
    print(f"Verifying ONNX graph validity at: {onnx_path}...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification SUCCESS.")
    return onnx_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--half", action="store_true", help="Export FP16 half precision")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batching")
    args = parser.parse_args()

    export_to_onnx(
        weights_path=args.weights,
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
    )
